"""
Mean Reversion Strategy
Fades 2-3σ moves away from VWAP on 5m and 15m timeframes.
"""
from typing import Optional
from datetime import datetime
import numpy as np
import time
from src.strategies.base_strategy import BaseStrategy, Signal
from src.infra.nt_bridge import MarketData
from config import MeanReversionConfig
from src.strategies.technical_indicators import TechnicalIndicators
from constants import StrategyConstants, TechnicalAnalysisConstants, TradingConstants, RISK


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy based on VWAP deviation."""
    
    def __init__(self, config: MeanReversionConfig, system_config, price_history_manager=None):
        
        super().__init__("MeanReversion", config, system_config, price_history_manager)
        self.config: MeanReversionConfig = config
        self.vwap_history = []
        self.last_trade_time = None  # Track last trade time for rate limiting
    
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """Generate mean reversion signal based on VWAP deviation."""
        if not self.should_trade(market_data):
            return None
        
        # Check for significant price changes (only log large moves > $1)
        has_change, change_info = self.has_significant_price_change(market_data)
        should_log_price = (has_change and 
                           not change_info.startswith("Price unchanged") and 
                           not change_info.startswith("Stale data") and
                           ("$1." in change_info or "$2." in change_info or "$3." in change_info or "$4." in change_info or "$5." in change_info))
        
        # Price analysis completed
        
        # Update price and volume history (no logging needed)
        self.update_price_history(market_data)
        
        # Check if we have any data for analysis (no hard requirements)
        if self.price_history_manager.get_data_length('5m') < 10:
            # Only log data waiting message every 30 seconds, not on every tick
            if not hasattr(self, '_last_data_wait_log') or (time.time() - self._last_data_wait_log) > 30:
                self.logger.debug("Waiting for sufficient 5m data (need 10 bars)")
                self._last_data_wait_log = time.time()
            return None
        
        # Calculate VWAP and deviation for 5m timeframe
        prices_5m = self.price_history_manager.get_prices('5m')
        volumes_5m = self.price_history_manager.get_volumes('5m')
        signal_5m = self._analyze_timeframe(prices_5m, volumes_5m, "5m", should_log_price)
        
        # Calculate VWAP and deviation for 15m timeframe
        signal_15m = None
        if self.price_history_manager.get_data_length('15m') >= 10:
            prices_15m = self.price_history_manager.get_prices('15m')
            volumes_15m = self.price_history_manager.get_volumes('15m')
            signal_15m = self._analyze_timeframe(prices_15m, volumes_15m, "15m", should_log_price)
        
        # Analyze 1m timeframe for confirmation if enabled
        signal_1m = None
        if (self.config.enable_1m_confirmation and 
            self.price_history_manager.get_data_length('1m') >= 10):
            prices_1m = self.price_history_manager.get_prices('1m')
            volumes_1m = self.price_history_manager.get_volumes('1m')
            signal_1m = self._analyze_1m_confirmation(prices_1m, volumes_1m, "1m")
        
        # Combine signals from all timeframes
        final_signal = self._combine_signals(signal_5m, signal_15m, signal_1m, market_data)
        
        if final_signal:
            self.last_signal_time = datetime.now()
        
        return final_signal
    
    
    def _analyze_timeframe(self, prices: list, volumes: list, timeframe: str, should_log_details: bool = False) -> Optional[Signal]:
        """Analyze a specific timeframe for mean reversion opportunities."""
        # Validate input data
        if not self._validate_timeframe_data(prices, volumes, timeframe):
            return None
        
        current_price = prices[-1]
        current_volume = volumes[-1]
        
        # Skip if volume is too low
        if current_volume < self.config.min_volume_threshold:
            pass  # Volume too low
            return None
        
        # Calculate and validate VWAP
        vwap_data = self._calculate_and_validate_vwap(prices, volumes, timeframe, should_log_details)
        if not vwap_data:
            return None
        
        vwap, vwap_prices, vwap_volumes = vwap_data
        
        # Calculate deviation metrics
        deviation_data = self._calculate_deviation_metrics(prices, vwap, timeframe)
        if not deviation_data:
            return None
        
        z_score, current_deviation, std_dev = deviation_data
        
        # Calculate technical indicators
        rsi = self.calculate_rsi(prices, period=self.system_config.technical_analysis.rsi_period, debug=self.config.rsi_debug_logging)
        atr = self._calculate_atr_for_stops(prices)
        
        # Log analysis details (debug level only)
        if should_log_details:
            self.logger.debug(f"Mean reversion analysis - {timeframe}: price=${current_price:.2f}, z_score={z_score:.2f}, rsi={rsi:.1f}, vwap=${vwap:.2f}")
        
        # Check signal conditions
        oversold, overbought = self._check_signal_conditions(z_score, rsi, current_volume, timeframe)
        
        # Generate signal based on conditions
        signal = None
        if oversold:
            signal = self._create_buy_signal(current_price, vwap, atr, z_score, rsi, timeframe)
            self.logger.info(f"MEAN_REV {timeframe}: BUY @ ${current_price:.2f} | Z-Score: {z_score:.2f} (<-1.5), RSI: {rsi:.1f} (<35), VWAP: ${vwap:.2f}")
        elif overbought:
            signal = self._create_sell_signal(current_price, vwap, atr, z_score, rsi, timeframe)
            self.logger.info(f"MEAN_REV {timeframe}: SELL @ ${current_price:.2f} | Z-Score: {z_score:.2f} (>1.5), RSI: {rsi:.1f} (>65), VWAP: ${vwap:.2f}")
        else:
            # Log hold decision with key metrics (rate limited)
            if should_log_details or self.should_log_detailed_analysis():
                self.logger.info(f"MEAN_REV {timeframe}: HOLD @ ${current_price:.2f} | Z-Score: {z_score:.2f} (need >1.5 or <-1.5), RSI: {rsi:.1f} (need <35 or >65)")
        
        return signal
    
    def _validate_timeframe_data(self, prices: list, volumes: list, timeframe: str) -> bool:
        """Validate input data for timeframe analysis."""
        if len(prices) < 10 or len(volumes) < 10:  # Just need basic data
            if self.should_log_detailed_analysis():
                pass  # Waiting for data
            return False
        return True
    
    def _calculate_and_validate_vwap(self, prices: list, volumes: list, timeframe: str, should_log_details: bool = False) -> Optional[tuple]:
        """Calculate VWAP and perform validation checks."""
        vwap_prices = prices[-self.config.vwap_period:]
        vwap_volumes = volumes[-self.config.vwap_period:]
        
        vwap = self.calculate_vwap(vwap_prices, vwap_volumes)
        
        if vwap == 0:
            return None
        
        # VWAP validation completed
        
        return vwap, vwap_prices, vwap_volumes
    
    def _calculate_deviation_metrics(self, prices: list, vwap: float, timeframe: str) -> Optional[tuple]:
        """Calculate price deviation metrics from VWAP."""
        price_deviations = [(p - vwap) / vwap for p in prices[-self.config.vwap_period:]]
        std_dev = np.std(price_deviations)
        
        if std_dev == 0:
            return None
        
        current_price = prices[-1]
        current_deviation = (current_price - vwap) / vwap
        z_score = current_deviation / std_dev
        
        return z_score, current_deviation, std_dev
    
    def _calculate_atr_for_stops(self, prices: list) -> float:
        """Calculate ATR for stop loss calculations."""
        atr_period = min(self.system_config.technical_analysis.atr_period, len(prices))
        return self.calculate_atr_simple(prices[-atr_period:])
    
    
    def _check_signal_conditions(self, z_score: float, rsi: float, current_volume: float, timeframe: str) -> tuple:
        """Check oversold/overbought signal conditions."""        
        # Signal conditions
        oversold = z_score < -self.config.deviation_threshold and rsi < self.config.rsi_oversold
        overbought = z_score > self.config.deviation_threshold and rsi > self.config.rsi_overbought
        
        return oversold, overbought
    
    def _create_buy_signal(self, current_price: float, vwap: float, atr: float, 
                          z_score: float, rsi: float, timeframe: str) -> Signal:
        """Create a buy signal for oversold conditions."""
        stop_price = current_price - (atr * RISK.STOP_LOSS_ATR_MULTIPLIER)
        target_price = vwap  # Target is VWAP re-touch
        
        # Calculate confidence
        confidence = self._calculate_signal_confidence(z_score, rsi, "BUY")
        
        return Signal(
            action=1,  # Buy
            confidence=confidence,
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            reason=f"Mean reversion buy {timeframe}: z-score={z_score:.2f}, RSI={rsi:.1f}",
            timestamp=datetime.now()
        )
    
    def _create_sell_signal(self, current_price: float, vwap: float, atr: float,
                           z_score: float, rsi: float, timeframe: str) -> Signal:
        """Create a sell signal for overbought conditions."""
        stop_price = current_price + (atr * RISK.STOP_LOSS_ATR_MULTIPLIER)
        target_price = vwap  # Target is VWAP re-touch
        
        # Calculate confidence
        confidence = self._calculate_signal_confidence(z_score, rsi, "SELL")
        
        return Signal(
            action=2,  # Sell
            confidence=confidence,
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            reason=f"Mean reversion sell {timeframe}: z-score={z_score:.2f}, RSI={rsi:.1f}",
            timestamp=datetime.now()
        )
    
    def _calculate_signal_confidence(self, z_score: float, rsi: float, direction: str) -> float:
        """Calculate signal confidence based on z-score and RSI strength."""
        # Scale confidence based on how much z-score exceeds threshold
        excess_z = abs(z_score) - self.config.deviation_threshold
        base_confidence = StrategyConstants.CONFIDENCE_BASE + (excess_z / 2.0) * StrategyConstants.CONFIDENCE_Z_SCORE_MULTIPLIER
        
        # Add RSI contribution (stronger RSI = higher confidence)
        rsi_distance = abs(rsi - TechnicalAnalysisConstants.RSI_NEUTRAL)
        rsi_factor = 1.0 + (rsi_distance / StrategyConstants.RSI_DISTANCE_MAX) * StrategyConstants.RSI_CONFIDENCE_BOOST_MAX
        
        confidence = min(TradingConstants.MAX_SIGNAL_CONFIDENCE, base_confidence * rsi_factor)
        
        # Confidence calculation completed
        
        return confidence
    
    def _analyze_1m_confirmation(self, prices: list, volumes: list, timeframe: str) -> Optional[dict]:
        """Analyze 1m timeframe for signal confirmation (not independent signals)."""
        if len(prices) < 10 or len(volumes) < 10:  # Just need basic data
            return None
        
        current_price = prices[-1]
        current_volume = volumes[-1]
        
        # Skip if volume is too low for 1m
        if current_volume < self.config.min_1m_volume_threshold:
            return None
        
        # Calculate VWAP for the period
        vwap = self.calculate_vwap(
            prices[-self.config.vwap_period:],
            volumes[-self.config.vwap_period:]
        )
        
        if vwap == 0:
            return None
        
        # Calculate price deviation from VWAP
        price_deviations = [(p - vwap) / vwap for p in prices[-self.config.vwap_period:]]
        std_dev = np.std(price_deviations)
        
        if std_dev == 0:
            return None
        
        current_deviation = (current_price - vwap) / vwap
        z_score = current_deviation / std_dev
        
        # Calculate RSI for additional confirmation (configurable period)
        rsi = self.calculate_rsi(prices, period=self.system_config.technical_analysis.rsi_period, debug=self.config.rsi_debug_logging)
        
        # Calculate signal quality scoring for 1m
        quality_score = self._calculate_1m_signal_quality(prices, volumes, z_score, rsi)
        
        # Return confirmation data (not a signal object)
        return {
            'z_score': z_score,
            'rsi': rsi,
            'volume': current_volume,
            'vwap': vwap,
            'price': current_price,
            'quality_score': quality_score,
            'oversold_confirmed': z_score < -self.config.deviation_threshold_1m and rsi < self.config.rsi_oversold,
            'overbought_confirmed': z_score > self.config.deviation_threshold_1m and rsi > self.config.rsi_overbought
        }
    
    def _calculate_1m_signal_quality(self, prices: list, volumes: list, z_score: float, rsi: float) -> float:
        """Calculate signal quality score for 1m confirmation to filter noise."""
        if len(prices) < 10:
            return 0.0
        
        quality_score = 0.0
        
        # 1. Z-score strength relative to 1m threshold
        if abs(z_score) >= self.config.deviation_threshold_1m:
            excess_z = abs(z_score) - self.config.deviation_threshold_1m
            z_strength = min(1.0, 0.5 + (excess_z / 2.0) * 0.5)  # 0.5 to 1.0 scale
        else:
            z_strength = abs(z_score) / self.config.deviation_threshold_1m * 0.5  # 0 to 0.5 scale
        quality_score += z_strength * 0.4
        
        # 2. RSI extremity (more extreme = higher quality)
        rsi_distance = abs(rsi - 50)  # Distance from neutral
        rsi_strength = min(1.0, rsi_distance / 50)  # Normalize to 0-1
        quality_score += rsi_strength * 0.3
        
        # 3. Volume consistency (stable volume = higher quality)
        recent_volumes = volumes[-10:]
        volume_cv = np.std(recent_volumes) / np.mean(recent_volumes) if np.mean(recent_volumes) > 0 else 1.0
        volume_quality = max(0.0, 1.0 - volume_cv)  # Lower coefficient of variation = higher quality
        quality_score += volume_quality * 0.2
        
        # 4. Price action consistency (smooth move = higher quality)
        recent_prices = prices[-10:]
        price_changes = [abs(recent_prices[i] - recent_prices[i-1]) for i in range(1, len(recent_prices))]
        avg_change = np.mean(price_changes) if price_changes else 0
        max_change = max(price_changes) if price_changes else 0
        consistency = 1.0 - (max_change / (avg_change * 3)) if avg_change > 0 else 0.0
        consistency = max(0.0, min(1.0, consistency))
        quality_score += consistency * 0.1
        
        return min(1.0, quality_score)
    
    def _check_rate_limiting(self) -> bool:
        """Check if rate limiting allows a new trade."""
        if not self.last_trade_time:
            return True
        
        time_since_last = datetime.now() - self.last_trade_time
        minutes_since_last = time_since_last.total_seconds() / 60
        
        if minutes_since_last < self.config.min_time_between_trades_minutes:
            return False
        
        return True
    
    
    def _combine_signals(self, signal_5m: Optional[Signal], signal_15m: Optional[Signal], 
                        signal_1m: Optional[dict], market_data: MarketData) -> Optional[Signal]:
        """Combine signals from different timeframes with 1m confirmation."""
        # Priority: 15m signal if available, otherwise 5m
        primary_signal = signal_15m if signal_15m else signal_5m
        
        if not primary_signal:
            return None
        
        # Check if signal meets minimum confidence threshold
        if primary_signal.confidence < self.system_config.risk_management.min_confidence:
            return None
        
        # NFVGS filter: disable signal generation when abs(nfvgs) > 0.5
        nfvgs_value = self._calculate_nfvgs_filter(market_data)
        if abs(nfvgs_value) > 0.5:
            return None
        
        # Check rate limiting
        if not self._check_rate_limiting():
            return None
        
        # Apply 1m confirmation if enabled and available
        if self.config.enable_1m_confirmation and signal_1m:
            confirmation_passed = self._apply_1m_confirmation(primary_signal, signal_1m)
            if not confirmation_passed:
                return None
        
        # Boost confidence if both timeframes agree
        if signal_5m and signal_15m and signal_5m.action == signal_15m.action:
            primary_signal.confidence = min(0.98, primary_signal.confidence * 1.2)
            primary_signal.reason += " (confirmed by both timeframes)"
        
        # Calculate final position size
        if primary_signal.stop_price:
            primary_signal.size = self.calculate_position_size(
                market_data, primary_signal.stop_price
            )
        
        # Update last trade time for rate limiting
        self.last_trade_time = datetime.now()
        
        return primary_signal
    
    def _apply_1m_confirmation(self, primary_signal: Signal, signal_1m: dict) -> bool:
        """Apply 1m confirmation logic to validate primary signal."""
        # Check signal quality threshold
        min_quality = 0.3  # Minimum quality score for 1m confirmation
        if signal_1m['quality_score'] < min_quality:
            return False
        
        # Check if 1m confirms the same direction as primary signal
        if primary_signal.action == 1:  # Buy signal
            if not signal_1m['oversold_confirmed']:
                return False
            
        elif primary_signal.action == 2:  # Sell signal
            if not signal_1m['overbought_confirmed']:
                return False
        
        # Boost confidence with 1m confirmation
        confidence_boost = min(0.15, signal_1m['quality_score'] * 0.2)
        primary_signal.confidence = min(0.98, primary_signal.confidence + confidence_boost)
        primary_signal.reason += f" (1m confirmed, quality={signal_1m['quality_score']:.2f})"
        
        return True
    
    def _calculate_nfvgs_filter(self, market_data: MarketData) -> float:
        """Calculate NFVGS value for signal filtering.
        
        Args:
            market_data: Current market data
            
        Returns:
            NFVGS value (positive = bullish gaps, negative = bearish gaps)
        """
        try:
            # Use 15m timeframe for NFVGS calculation (good balance of responsiveness and stability)
            highs_15m = self.price_history_manager.get_highs('15m')
            lows_15m = self.price_history_manager.get_lows('15m')
            closes_15m = self.price_history_manager.get_prices('15m')
            
            # Need at least 18 bars for NFVGS calculation (14 for ATR + 4 for gap detection)
            if len(closes_15m) < 18 or len(highs_15m) < 18 or len(lows_15m) < 18:
                return 0.0
            
            # Calculate NFVGS using the technical indicators service
            nfvgs_values = TechnicalIndicators.calculate_nfvgs(
                highs=highs_15m,
                lows=lows_15m,
                closes=closes_15m,
                atr_period=14,
                decay_ema=5  
            )
            
            if len(nfvgs_values) == 0:
                return 0.0
            
            # Use the latest NFVGS value
            current_nfvgs = nfvgs_values[-1]
            
            return current_nfvgs
            
        except Exception as e:
            self.logger.error(f"Error calculating NFVGS filter: {e}")
            return 0.0
    
    def get_strategy_metrics(self) -> dict:
        """Get strategy-specific metrics."""
        base_metrics = super().get_strategy_metrics()
        
        base_metrics.update({
            'vwap_period': self.config.vwap_period,
            'deviation_threshold': self.config.deviation_threshold,
            'price_5m_history_length': self.price_history_manager.get_data_length('5m'),
            'price_15m_history_length': self.price_history_manager.get_data_length('15m'),
            'current_vwap_5m': self.calculate_vwap(
                self.price_history_manager.get_prices('5m', self.config.vwap_period),
                self.price_history_manager.get_volumes('5m', self.config.vwap_period)
            ) if self.price_history_manager.get_data_length('5m') >= self.config.vwap_period else 0
        })
        
        return base_metrics