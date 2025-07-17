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
    
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """Generate mean reversion signal based on VWAP deviation."""
        if not self.should_trade(market_data):
            self.logger.info("Mean Reversion: should_trade() returned False")
            return None
        
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
        
        # Combine signals from both timeframes
        final_signal = self._combine_signals(signal_5m, signal_15m, market_data)
        
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
        
        # Calculate RSI for additional confirmation
        rsi = self.calculate_rsi(prices[-20:])
        
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
        
        # RSI details
        rsi_prices = prices[-20:]
        self.logger.info(f"RSI Period: 20 bars (last 20 prices)")
        self.logger.info(f"RSI Price Range: ${min(rsi_prices):.2f} - ${max(rsi_prices):.2f}")
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
            
            confidence = min(0.95, abs(z_score) / 4.0)  # Higher confidence for larger deviations
            
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
            
            confidence = min(0.95, abs(z_score) / 4.0)
            
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
    
    
    def _combine_signals(self, signal_5m: Optional[Signal], signal_15m: Optional[Signal], 
                        market_data: MarketData) -> Optional[Signal]:
        """Combine signals from different timeframes."""
        # Priority: 15m signal if available, otherwise 5m
        primary_signal = signal_15m if signal_15m else signal_5m
        
        if not primary_signal:
            return None
        
        # Check if signal meets minimum confidence threshold
        if primary_signal.confidence < self.config.min_confidence:
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
        
        return primary_signal
    
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