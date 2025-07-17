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
    
    def __init__(self, config: MeanReversionConfig):
        
        super().__init__("MeanReversion", config)
        self.config: MeanReversionConfig = config
        self.logger = logging.getLogger(__name__)
        self.vwap_history = []
        self.price_5m_history = []
        self.volume_5m_history = []
        self.price_15m_history = []
        self.volume_15m_history = []
    
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """Generate mean reversion signal based on VWAP deviation."""
        if not self.should_trade(market_data):
            return None
        
        # Update price and volume history
        self._update_histories(market_data)
        
        self.logger.debug(f"Mean Reversion: 5m history={len(self.price_5m_history)}, 15m history={len(self.price_15m_history)}")
        
        # Need sufficient history for analysis
        if len(self.price_5m_history) < self.config.vwap_period:
            self.logger.debug(f"Mean Reversion: Insufficient 5m data - need {self.config.vwap_period}, have {len(self.price_5m_history)}")
            return None
        
        # Calculate VWAP and deviation for 5m timeframe
        signal_5m = self._analyze_timeframe(
            self.price_5m_history, self.volume_5m_history, "5m"
        )
        
        # Calculate VWAP and deviation for 15m timeframe
        signal_15m = None
        if len(self.price_15m_history) >= self.config.vwap_period:
            signal_15m = self._analyze_timeframe(
                self.price_15m_history, self.volume_15m_history, "15m"
            )
        
        # Combine signals from both timeframes
        final_signal = self._combine_signals(signal_5m, signal_15m, market_data)
        
        if final_signal:
            self.last_signal_time = datetime.now()
        
        return final_signal
    
    def _update_histories(self, market_data: MarketData):
        """Update price and volume histories."""
        # Update 5m data
        if market_data.price_5m:
            self.price_5m_history = market_data.price_5m[-50:]  # Keep last 50 bars
        if market_data.volume_5m:
            self.volume_5m_history = market_data.volume_5m[-50:]
        
        # Update 15m data
        if market_data.price_15m:
            self.price_15m_history = market_data.price_15m[-50:]
        if market_data.volume_15m:
            self.volume_15m_history = market_data.volume_15m[-50:]
    
    def _analyze_timeframe(self, prices: list, volumes: list, timeframe: str) -> Optional[Signal]:
        """Analyze a specific timeframe for mean reversion opportunities."""
        if len(prices) < self.config.vwap_period or len(volumes) < self.config.vwap_period:
            self.logger.debug(f"{timeframe}: Insufficient data - need {self.config.vwap_period}, have {len(prices)} prices, {len(volumes)} volumes")
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
        atr = self._calculate_atr_simple(prices[-atr_period:])
        
        # Log all calculated values
        self.logger.info(f"{timeframe} Mean Reversion Analysis:")
        self.logger.info(f"Price: ${current_price:.2f}, Volume: {current_volume:.0f}")
        self.logger.info(f"VWAP: ${vwap:.2f}, Deviation: {current_deviation:.4f}")
        self.logger.info(f"Z-Score: {z_score:.2f}, RSI: {rsi:.1f}")
        self.logger.info(f"ATR: ${atr:.2f}, Std Dev: {std_dev:.6f}")
        self.logger.info(f"Thresholds: Z±{self.config.deviation_threshold}, RSI {self.config.rsi_oversold}/{self.config.rsi_overbought}")
        
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
    
    def _calculate_atr_simple(self, prices: list) -> float:
        """Calculate simple ATR approximation using price ranges."""
        if len(prices) < 2:
            return prices[0] * 0.01  # 1% default
        
        ranges = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        return np.mean(ranges[-10:])  # 10-period average
    
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
            'price_5m_history_length': len(self.price_5m_history),
            'price_15m_history_length': len(self.price_15m_history),
            'current_vwap_5m': self.calculate_vwap(
                self.price_5m_history[-self.config.vwap_period:],
                self.volume_5m_history[-self.config.vwap_period:]
            ) if len(self.price_5m_history) >= self.config.vwap_period else 0
        })
        
        return base_metrics