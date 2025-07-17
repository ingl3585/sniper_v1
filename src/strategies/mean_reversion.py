"""
Mean Reversion Strategy
Fades 2-3σ moves away from VWAP on 5m and 15m timeframes.
"""
from typing import Optional
from datetime import datetime
import numpy as np
import logging
from src.strategies.base_strategy import BaseStrategy, Signal
from src.infra.nt_bridge import MarketData
from src.config import MeanReversionConfig


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy based on VWAP deviation."""
    
    def __init__(self, config: MeanReversionConfig, price_history_manager=None):
        
        super().__init__("MeanReversion", config, price_history_manager)
        self.config: MeanReversionConfig = config
        self.logger = logging.getLogger(__name__)
        self.vwap_history = []
        self.last_trade_time = None  # Track last trade time for rate limiting
    
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """Generate mean reversion signal based on VWAP deviation."""
        if not self.should_trade(market_data):
            self.logger.info("Mean Reversion: should_trade() returned False")
            return None
        
        # Check data freshness
        freshness = market_data.get_data_freshness_warning()
        self.logger.info(f"Mean Reversion: {freshness}")
        
        # Update price and volume history
        self.update_price_history(market_data)
        self.logger.info("Mean Reversion: Updated price history, checking data sufficiency...")
        
        # Check if we have sufficient data for analysis
        if not self.price_history_manager.has_sufficient_data('5m', self.config.vwap_period):
            current_5m_length = self.price_history_manager.get_data_length('5m')
            self.logger.info(f"Mean Reversion: Insufficient 5m data - need {self.config.vwap_period}, have {current_5m_length}")
            return None
        
        # Calculate VWAP and deviation for 5m timeframe
        prices_5m = self.price_history_manager.get_prices('5m')
        volumes_5m = self.price_history_manager.get_volumes('5m')
        signal_5m = self._analyze_timeframe(prices_5m, volumes_5m, "5m")
        
        # Calculate VWAP and deviation for 15m timeframe
        signal_15m = None
        if self.price_history_manager.has_sufficient_data('15m', self.config.vwap_period):
            prices_15m = self.price_history_manager.get_prices('15m')
            volumes_15m = self.price_history_manager.get_volumes('15m')
            signal_15m = self._analyze_timeframe(prices_15m, volumes_15m, "15m")
        
        # Analyze 1m timeframe for confirmation if enabled
        signal_1m = None
        if (self.config.enable_1m_confirmation and 
            self.price_history_manager.has_sufficient_data('1m', self.config.vwap_period)):
            prices_1m = self.price_history_manager.get_prices('1m')
            volumes_1m = self.price_history_manager.get_volumes('1m')
            signal_1m = self._analyze_1m_confirmation(prices_1m, volumes_1m, "1m")
        
        # Combine signals from all timeframes
        final_signal = self._combine_signals(signal_5m, signal_15m, signal_1m, market_data)
        
        if final_signal:
            self.last_signal_time = datetime.now()
        
        return final_signal
    
    
    def _analyze_timeframe(self, prices: list, volumes: list, timeframe: str) -> Optional[Signal]:
        """Analyze a specific timeframe for mean reversion opportunities."""
        if len(prices) < self.config.vwap_period or len(volumes) < self.config.vwap_period:
            self.logger.info(f"{timeframe}: Insufficient data - need {self.config.vwap_period}, have {len(prices)} prices, {len(volumes)} volumes")
            return None
        
        current_price = prices[-1]
        current_volume = volumes[-1]
        
        # Skip if volume is too low
        if current_volume < self.config.min_volume_threshold:
            self.logger.debug(f"{timeframe}: Volume too low - {current_volume:.0f} < {self.config.min_volume_threshold}")
            return None
        
        # Calculate VWAP for the period
        vwap_prices = prices[-self.config.vwap_period:]
        vwap_volumes = volumes[-self.config.vwap_period:]
        
        vwap = self.calculate_vwap(vwap_prices, vwap_volumes)
        
        if vwap == 0:
            self.logger.warning(f"{timeframe}: VWAP calculation returned 0")
            return None
        
        # Detailed VWAP validation logging
        total_pv = sum(p * v for p, v in zip(vwap_prices, vwap_volumes))
        total_volume = sum(vwap_volumes)
        manual_vwap = total_pv / total_volume if total_volume > 0 else 0
        simple_avg = sum(vwap_prices) / len(vwap_prices)
        
        self.logger.info(f"{timeframe} VWAP Validation:")
        self.logger.info(f"  VWAP Function Result: ${vwap:.2f}")
        self.logger.info(f"  Manual Calculation: ${manual_vwap:.2f}")
        self.logger.info(f"  Simple Average: ${simple_avg:.2f}")
        self.logger.info(f"  Total Price*Volume: ${total_pv:,.0f}")
        self.logger.info(f"  Total Volume: {total_volume:,.0f}")
        
        # Ensure VWAP is reasonable (within 5% of simple average)
        if abs(vwap - simple_avg) / simple_avg > 0.05:
            self.logger.warning(f"{timeframe}: VWAP ${vwap:.2f} differs significantly from simple average ${simple_avg:.2f}")
        
        # Check for volume concentration (detect if few bars dominate)
        max_volume = max(vwap_volumes)
        if max_volume > total_volume * 0.5:
            self.logger.warning(f"{timeframe}: Single bar dominates volume - max: {max_volume:.0f} vs total: {total_volume:.0f}")
        
        # Log volume distribution
        volume_weights = [v/total_volume for v in vwap_volumes]
        top_3_weights = sorted(volume_weights, reverse=True)[:3]
        self.logger.info(f"  Top 3 Volume Weights: {[f'{w:.1%}' for w in top_3_weights]}")
        
        # Calculate price deviation from VWAP
        price_deviations = [(p - vwap) / vwap for p in prices[-self.config.vwap_period:]]
        std_dev = np.std(price_deviations)
        
        if std_dev == 0:
            self.logger.warning(f"{timeframe}: Standard deviation is 0")
            return None
        
        current_deviation = (current_price - vwap) / vwap
        z_score = current_deviation / std_dev
        
        # Calculate RSI for additional confirmation (configurable period)
        rsi = self.calculate_rsi(prices, period=self.config.rsi_period, debug=self.config.rsi_debug_logging)
        
        # Calculate ATR for stop loss
        atr_period = min(self.config.atr_lookback, len(prices))
        atr = self.calculate_atr_simple(prices[-atr_period:])
        
        # Log all calculated values with detailed breakdown
        self.logger.info(f"=== {timeframe} Mean Reversion Analysis ===")
        self.logger.info(f"Current Price: ${current_price:.2f}")
        self.logger.info(f"Current Volume: {current_volume:.0f}")
        self.logger.info(f"Data Points: {len(prices)} prices, {len(volumes)} volumes")
        
        # VWAP calculation details
        vwap_prices = prices[-self.config.vwap_period:]
        vwap_volumes = volumes[-self.config.vwap_period:]
        self.logger.info(f"VWAP Period: {self.config.vwap_period} bars")
        self.logger.info(f"VWAP Price Range: ${min(vwap_prices):.2f} - ${max(vwap_prices):.2f}")
        self.logger.info(f"VWAP Volume Range: {min(vwap_volumes):.0f} - {max(vwap_volumes):.0f}")
        self.logger.info(f"Calculated VWAP: ${vwap:.2f}")
        
        # Deviation analysis
        self.logger.info(f"Price vs VWAP: ${current_price:.2f} vs ${vwap:.2f}")
        self.logger.info(f"Raw Deviation: {current_deviation:.6f} ({current_deviation*100:.2f}%)")
        self.logger.info(f"Standard Deviation: {std_dev:.6f}")
        self.logger.info(f"Z-Score: {z_score:.3f}")
        
        # RSI detailed calculation logging for validation
        self.logger.info(f"RSI Calculation Details:")
        smoothing_method = "Wilder's exponential smoothing" if self.config.rsi_use_wilder_smoothing else "Simple average"
        self.logger.info(f"RSI Period: {self.config.rsi_period} bars ({smoothing_method})")
        self.logger.info(f"RSI Price Range: ${min(prices):.2f} - ${max(prices):.2f}")
        
        # Add detailed RSI debugging for troubleshooting
        if len(prices) >= 15:  # Need at least 15 for 14-period RSI
            recent_prices = prices[-5:]  # Last 5 prices for context
            deltas = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
            self.logger.info(f"Last 5 prices: {[f'${p:.2f}' for p in recent_prices]}")
            self.logger.info(f"Last 4 price changes: {[f'{d:+.2f}' for d in deltas]}")
            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]
            self.logger.info(f"Recent gains: {[f'{g:.2f}' for g in gains]}")
            self.logger.info(f"Recent losses: {[f'{l:.2f}' for l in losses]}")
        
        self.logger.info(f"Calculated RSI: {rsi:.2f}")
        
        # ATR details
        self.logger.info(f"ATR Period: {atr_period} bars")
        self.logger.info(f"Calculated ATR: ${atr:.2f}")
        
        # Threshold checks
        self.logger.info(f"Thresholds Check:")
        self.logger.info(f"Z-Score {z_score:.3f} vs ±{self.config.deviation_threshold}")
        self.logger.info(f"RSI {rsi:.2f} vs {self.config.rsi_oversold}/{self.config.rsi_overbought}")
        self.logger.info(f"Volume {current_volume:.0f} vs {self.config.min_volume_threshold}")
        
        # Signal conditions
        oversold = z_score < -self.config.deviation_threshold and rsi < self.config.rsi_oversold
        overbought = z_score > self.config.deviation_threshold and rsi > self.config.rsi_overbought
        self.logger.info(f"Signal Conditions:")
        self.logger.info(f"Oversold: {oversold} (z < -{self.config.deviation_threshold} AND rsi < {self.config.rsi_oversold})")
        self.logger.info(f"Overbought: {overbought} (z > {self.config.deviation_threshold} AND rsi > {self.config.rsi_overbought})")
        
        # Mean reversion signals
        signal = None
        
        # Oversold condition: price significantly below VWAP
        if (z_score < -self.config.deviation_threshold and 
            rsi < self.config.rsi_oversold):
            
            stop_price = current_price - (atr * self.config.stop_loss_atr_multiplier)
            target_price = vwap  # Target is VWAP re-touch
            
            # Scale confidence based on how much z-score exceeds threshold
            excess_z = abs(z_score) - self.config.deviation_threshold
            base_confidence = 0.6 + (excess_z / 2.0) * 0.35
            
            # Add RSI contribution (stronger RSI = higher confidence)
            rsi_distance = abs(rsi - 50)  # Distance from neutral (50)
            rsi_factor = 1.0 + (rsi_distance / 50) * 0.1  # Up to 10% boost for extreme RSI
            
            confidence = min(0.95, base_confidence * rsi_factor)
            
            # Log confidence calculation details for validation
            self.logger.info(f"Confidence Calculation (BUY): z_score={z_score:.3f}, excess_z={excess_z:.3f}, base_confidence={base_confidence:.3f}, rsi_distance={rsi_distance:.1f}, rsi_factor={rsi_factor:.3f}, final_confidence={confidence:.3f}")
            
            signal = Signal(
                action=1,  # Buy
                confidence=confidence,
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                reason=f"Mean reversion buy {timeframe}: z-score={z_score:.2f}, RSI={rsi:.1f}",
                timestamp=datetime.now()
            )
        
        # Overbought condition: price significantly above VWAP
        elif (z_score > self.config.deviation_threshold and 
              rsi > self.config.rsi_overbought):
            
            stop_price = current_price + (atr * self.config.stop_loss_atr_multiplier)
            target_price = vwap  # Target is VWAP re-touch
            
            # Scale confidence based on how much z-score exceeds threshold
            excess_z = abs(z_score) - self.config.deviation_threshold
            base_confidence = 0.6 + (excess_z / 2.0) * 0.35
            
            # Add RSI contribution (stronger RSI = higher confidence)
            rsi_distance = abs(rsi - 50)  # Distance from neutral (50)
            rsi_factor = 1.0 + (rsi_distance / 50) * 0.1  # Up to 10% boost for extreme RSI
            
            confidence = min(0.95, base_confidence * rsi_factor)
            
            # Log confidence calculation details for validation
            self.logger.info(f"Confidence Calculation (SELL): z_score={z_score:.3f}, excess_z={excess_z:.3f}, base_confidence={base_confidence:.3f}, rsi_distance={rsi_distance:.1f}, rsi_factor={rsi_factor:.3f}, final_confidence={confidence:.3f}")
            
            signal = Signal(
                action=2,  # Sell
                confidence=confidence,
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                reason=f"Mean reversion sell {timeframe}: z-score={z_score:.2f}, RSI={rsi:.1f}",
                timestamp=datetime.now()
            )
        
        return signal
    
    def _analyze_1m_confirmation(self, prices: list, volumes: list, timeframe: str) -> Optional[dict]:
        """Analyze 1m timeframe for signal confirmation (not independent signals)."""
        if len(prices) < self.config.vwap_period or len(volumes) < self.config.vwap_period:
            self.logger.info(f"{timeframe}: Insufficient data for confirmation - need {self.config.vwap_period}, have {len(prices)} prices")
            return None
        
        current_price = prices[-1]
        current_volume = volumes[-1]
        
        # Skip if volume is too low for 1m
        if current_volume < self.config.min_1m_volume_threshold:
            self.logger.debug(f"{timeframe}: Volume too low for confirmation - {current_volume:.0f} < {self.config.min_1m_volume_threshold}")
            return None
        
        # Calculate VWAP for the period
        vwap = self.calculate_vwap(
            prices[-self.config.vwap_period:],
            volumes[-self.config.vwap_period:]
        )
        
        if vwap == 0:
            self.logger.warning(f"{timeframe}: VWAP calculation returned 0")
            return None
        
        # Calculate price deviation from VWAP
        price_deviations = [(p - vwap) / vwap for p in prices[-self.config.vwap_period:]]
        std_dev = np.std(price_deviations)
        
        if std_dev == 0:
            self.logger.warning(f"{timeframe}: Standard deviation is 0")
            return None
        
        current_deviation = (current_price - vwap) / vwap
        z_score = current_deviation / std_dev
        
        # Calculate RSI for additional confirmation (configurable period)
        rsi = self.calculate_rsi(prices, period=self.config.rsi_period, debug=self.config.rsi_debug_logging)
        
        # Log 1m confirmation analysis
        self.logger.info(f"=== {timeframe} Confirmation Analysis ===")
        self.logger.info(f"Current Price: ${current_price:.2f}")
        self.logger.info(f"Current Volume: {current_volume:.0f}")
        self.logger.info(f"VWAP: ${vwap:.2f}")
        self.logger.info(f"Z-Score: {z_score:.3f} (threshold: ±{self.config.deviation_threshold_1m})")
        
        # 1m RSI validation details
        self.logger.info(f"1m RSI Validation:")
        if len(prices) >= 15:
            recent_prices = prices[-5:]
            deltas = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
            self.logger.info(f"Last 5 prices: {[f'${p:.2f}' for p in recent_prices]}")
            self.logger.info(f"Last 4 changes: {[f'{d:+.2f}' for d in deltas]}")
        
        smoothing_method = "Wilder's" if self.config.rsi_use_wilder_smoothing else "Simple"
        self.logger.info(f"Calculated 1m RSI: {rsi:.2f} ({self.config.rsi_period}-period {smoothing_method})")
        
        # Signal quality scoring for 1m
        quality_score = self._calculate_1m_signal_quality(prices, volumes, z_score, rsi)
        self.logger.info(f"1m Signal Quality Score: {quality_score:.3f}")
        
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
            self.logger.info(f"Rate limit: {minutes_since_last:.1f} minutes since last trade (min: {self.config.min_time_between_trades_minutes})")
            return False
        
        return True
    
    
    def _combine_signals(self, signal_5m: Optional[Signal], signal_15m: Optional[Signal], 
                        signal_1m: Optional[dict], market_data: MarketData) -> Optional[Signal]:
        """Combine signals from different timeframes with 1m confirmation."""
        # Priority: 15m signal if available, otherwise 5m
        primary_signal = signal_15m if signal_15m else signal_5m
        
        if not primary_signal:
            self.logger.info("No primary signal from 5m or 15m timeframes")
            return None
        
        # Check if signal meets minimum confidence threshold
        if primary_signal.confidence < self.config.min_confidence:
            self.logger.info(f"Primary signal confidence {primary_signal.confidence:.3f} below minimum {self.config.min_confidence}")
            return None
        
        # Check rate limiting
        if not self._check_rate_limiting():
            return None
        
        # Apply 1m confirmation if enabled and available
        if self.config.enable_1m_confirmation and signal_1m:
            confirmation_passed = self._apply_1m_confirmation(primary_signal, signal_1m)
            if not confirmation_passed:
                self.logger.info("Signal rejected by 1m confirmation")
                return None
        else:
            self.logger.info("1m confirmation disabled or unavailable")
        
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
        self.logger.info("=== 1m Confirmation Check ===")
        
        # Check signal quality threshold
        min_quality = 0.3  # Minimum quality score for 1m confirmation
        if signal_1m['quality_score'] < min_quality:
            self.logger.info(f"1m quality score {signal_1m['quality_score']:.3f} below threshold {min_quality}")
            return False
        
        # Check if 1m confirms the same direction as primary signal
        if primary_signal.action == 1:  # Buy signal
            if not signal_1m['oversold_confirmed']:
                self.logger.info(f"1m does not confirm oversold condition: z_score={signal_1m['z_score']:.3f}, rsi={signal_1m['rsi']:.2f}")
                return False
            self.logger.info(f"1m CONFIRMS buy signal: z_score={signal_1m['z_score']:.3f}, rsi={signal_1m['rsi']:.2f}")
            
        elif primary_signal.action == 2:  # Sell signal
            if not signal_1m['overbought_confirmed']:
                self.logger.info(f"1m does not confirm overbought condition: z_score={signal_1m['z_score']:.3f}, rsi={signal_1m['rsi']:.2f}")
                return False
            self.logger.info(f"1m CONFIRMS sell signal: z_score={signal_1m['z_score']:.3f}, rsi={signal_1m['rsi']:.2f}")
        
        # Boost confidence with 1m confirmation
        confidence_boost = min(0.15, signal_1m['quality_score'] * 0.2)
        primary_signal.confidence = min(0.98, primary_signal.confidence + confidence_boost)
        primary_signal.reason += f" (1m confirmed, quality={signal_1m['quality_score']:.2f})"
        
        self.logger.info(f"1m confirmation passed, confidence boosted by {confidence_boost:.3f}")
        return True
    
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
            ) if self.price_history_manager.has_sufficient_data('5m', self.config.vwap_period) else 0
        })
        
        return base_metrics