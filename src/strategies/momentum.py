"""
Momentum Strategy
Trades trend continuation using EMA crossovers and trend strength.
"""
from typing import Optional
from datetime import datetime
import numpy as np
import logging
from src.strategies.base_strategy import BaseStrategy, Signal
from src.infra.nt_bridge import MarketData
from src.config import MomentumConfig
from src.infra.nt_bridge import TradeSignal


class MomentumStrategy(BaseStrategy):
    """Momentum strategy based on EMA crossovers and trend strength."""
    
    def __init__(self, config: MomentumConfig, price_history_manager=None):
        
        super().__init__("Momentum", config, price_history_manager)
        self.config: MomentumConfig = config
        self.logger = logging.getLogger(__name__)
        self.trend_direction = 0  # 1=up, -1=down, 0=neutral
        self.trend_strength = 0.0
        self.trend_duration = 0
    
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """Generate momentum signal based on EMA crossovers and trend strength."""
        if not self.should_trade(market_data):
            return None
        
        # Update price and volume history
        self.update_price_history(market_data)
        
        # Need sufficient history for EMA calculation
        if not self.price_history_manager.has_sufficient_data('1h', self.config.slow_ema_period):
            return None
        
        # Analyze trend on 1h timeframe
        prices_1h = self.price_history_manager.get_prices('1h')
        volumes_1h = self.price_history_manager.get_volumes('1h')
        signal_1h = self._analyze_trend(prices_1h, volumes_1h, "1h")
        
        # Analyze trend on 30m timeframe for confirmation
        signal_30m = None
        if self.price_history_manager.has_sufficient_data('30m', self.config.slow_ema_period):
            prices_30m = self.price_history_manager.get_prices('30m')
            volumes_30m = self.price_history_manager.get_volumes('30m')
            signal_30m = self._analyze_trend(prices_30m, volumes_30m, "30m")
        
        # Combine signals from both timeframes
        final_signal = self._combine_momentum_signals(signal_1h, signal_30m, market_data)
        
        if final_signal:
            self.last_signal_time = datetime.now()
        
        return final_signal
    
    
    def create_trade_signal(self, signal: Signal, market_data: MarketData) -> 'TradeSignal':
        """Convert strategy signal to trade signal for execution."""
        
        # Calculate position size if not specified
        if signal.size == 1 and signal.stop_price:
            signal.size = self.calculate_position_size(market_data, signal.stop_price)
        
        return TradeSignal(
            action=signal.action,
            position_size=signal.size,
            confidence=signal.confidence,
            use_stop=signal.stop_price is not None,
            stop_price=signal.stop_price or 0.0,
            use_target=signal.target_price is not None,
            target_price=signal.target_price or 0.0
        )
    
    def _analyze_trend(self, prices: list, volumes: list, timeframe: str) -> Optional[Signal]:
        """Analyze trend strength and direction for momentum signals."""
        if len(prices) < self.config.slow_ema_period:
            return None
        
        current_price = prices[-1]
        
        # Calculate EMAs
        fast_ema = self.calculate_ema(prices, self.config.fast_ema_period)
        slow_ema = self.calculate_ema(prices, self.config.slow_ema_period)
        
        # Calculate previous EMAs for crossover detection
        if len(prices) > self.config.slow_ema_period:
            prev_fast_ema = self.calculate_ema(prices[:-1], self.config.fast_ema_period)
            prev_slow_ema = self.calculate_ema(prices[:-1], self.config.slow_ema_period)
        else:
            prev_fast_ema = fast_ema
            prev_slow_ema = slow_ema
        
        # Detect crossovers
        bullish_crossover = (fast_ema > slow_ema and prev_fast_ema <= prev_slow_ema)
        bearish_crossover = (fast_ema < slow_ema and prev_fast_ema >= prev_slow_ema)
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(prices, fast_ema, slow_ema)
        
        # Calculate volume confirmation
        volume_confirmation = self._calculate_volume_confirmation(volumes)
        
        # Update trend tracking
        self._update_trend_tracking(fast_ema, slow_ema, trend_strength)
        
        # Calculate ATR for stop loss
        atr_period = min(self.config.atr_lookback, len(prices))
        atr = self.calculate_atr_simple(prices[-atr_period:])
        
        # Log all calculated values
        self.logger.info(f"{timeframe} Momentum Analysis:")
        self.logger.info(f"Price: ${current_price:.2f}")
        self.logger.info(f"EMA Fast: ${fast_ema:.2f}, Slow: ${slow_ema:.2f}")
        self.logger.info(f"Prev Fast: ${prev_fast_ema:.2f}, Prev Slow: ${prev_slow_ema:.2f}")
        self.logger.info(f"Crossover: Bull={bullish_crossover}, Bear={bearish_crossover}")
        self.logger.info(f"Trend Strength: {trend_strength:.3f}, Volume Conf: {volume_confirmation:.3f}")
        self.logger.info(f"Trend Direction: {self.trend_direction}, Duration: {self.trend_duration}")
        self.logger.info(f"ATR: ${atr:.2f}")
        self.logger.info(f"Thresholds: Trend>{self.config.trend_strength_threshold}, Vol>0.5, Duration>{self.config.min_trend_duration}")
        
        signal = None
        
        # Bullish momentum signal
        if (bullish_crossover and 
            trend_strength > self.config.trend_strength_threshold and
            volume_confirmation > 0.5 and
            self.trend_duration >= self.config.min_trend_duration):
            
            stop_price = current_price - (atr * self.config.trail_stop_atr_multiplier)
            
            # Target is based on trend strength and ATR
            target_distance = atr * (2 + trend_strength * 3)  # 2-5x ATR based on strength
            target_price = current_price + target_distance
            
            confidence = min(0.95, trend_strength * volume_confirmation)
            
            signal = Signal(
                action=1,  # Buy
                confidence=confidence,
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                reason=f"Momentum buy {timeframe}: EMA crossover, strength={trend_strength:.2f}, vol_conf={volume_confirmation:.2f}",
                timestamp=datetime.now()
            )
        
        # Bearish momentum signal
        elif (bearish_crossover and 
              trend_strength > self.config.trend_strength_threshold and
              volume_confirmation > 0.5 and
              self.trend_duration >= self.config.min_trend_duration):
            
            stop_price = current_price + (atr * self.config.trail_stop_atr_multiplier)
            
            # Target is based on trend strength and ATR
            target_distance = atr * (2 + trend_strength * 3)
            target_price = current_price - target_distance
            
            confidence = min(0.95, trend_strength * volume_confirmation)
            
            signal = Signal(
                action=2,  # Sell
                confidence=confidence,
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                reason=f"Momentum sell {timeframe}: EMA crossover, strength={trend_strength:.2f}, vol_conf={volume_confirmation:.2f}",
                timestamp=datetime.now()
            )
        
        return signal
    
    def _calculate_trend_strength(self, prices: list, fast_ema: float, slow_ema: float) -> float:
        """Calculate trend strength based on EMA separation and price action."""
        if len(prices) < 20:
            return 0.0
        
        # EMA separation strength
        ema_separation = abs(fast_ema - slow_ema) / slow_ema
        
        # Price momentum strength
        price_momentum = (prices[-1] - prices[-10]) / prices[-10]
        momentum_strength = abs(price_momentum)
        
        # Consistency of trend direction
        recent_prices = prices[-10:]
        direction_consistency = 0.0
        
        if len(recent_prices) >= 2:
            up_moves = sum(1 for i in range(1, len(recent_prices)) 
                          if recent_prices[i] > recent_prices[i-1])
            direction_consistency = up_moves / (len(recent_prices) - 1)
            
            # Adjust for downtrend
            if price_momentum < 0:
                direction_consistency = 1.0 - direction_consistency
        
        # Combined trend strength
        trend_strength = (ema_separation * 0.4 + 
                         momentum_strength * 0.4 + 
                         direction_consistency * 0.2)
        
        return min(1.0, trend_strength)
    
    def _calculate_volume_confirmation(self, volumes: list) -> float:
        """Calculate volume confirmation for momentum signals."""
        if len(volumes) < self.config.volume_confirmation_period:
            return 0.5  # Neutral if insufficient data
        
        recent_volumes = volumes[-self.config.volume_confirmation_period:]
        avg_volume = np.mean(volumes[-20:] if len(volumes) >= 20 else volumes)
        
        if avg_volume == 0:
            return 0.5
        
        # Check if recent volume is above average
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume
        
        # Volume confirmation strength
        volume_strength = min(1.0, volume_ratio / 2.0)  # Normalize around 2x average
        
        return volume_strength
    
    def _update_trend_tracking(self, fast_ema: float, slow_ema: float, trend_strength: float):
        """Update trend direction and duration tracking."""
        current_direction = 1 if fast_ema > slow_ema else -1
        
        if current_direction == self.trend_direction:
            self.trend_duration += 1
        else:
            self.trend_direction = current_direction
            self.trend_duration = 1
        
        self.trend_strength = trend_strength
    
    
    def _combine_momentum_signals(self, signal_1h: Optional[Signal], signal_30m: Optional[Signal], 
                                market_data: MarketData) -> Optional[Signal]:
        """Combine momentum signals from different timeframes."""
        # Priority: 1h signal as primary, 30m for confirmation
        primary_signal = signal_1h
        
        if not primary_signal:
            return None
        
        # Check if signal meets minimum confidence threshold
        if primary_signal.confidence < self.config.min_confidence:
            return None
        
        # Boost confidence if both timeframes agree
        if signal_30m and signal_1h and signal_30m.action == signal_1h.action:
            primary_signal.confidence = min(0.98, primary_signal.confidence * 1.15)
            primary_signal.reason += " (confirmed by 30m)"
        
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
            'fast_ema_period': self.config.fast_ema_period,
            'slow_ema_period': self.config.slow_ema_period,
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'trend_duration': self.trend_duration,
            'price_1h_history_length': self.price_history_manager.get_data_length('1h'),
            'current_fast_ema': self.calculate_ema(
                self.price_history_manager.get_prices('1h'), self.config.fast_ema_period
            ) if self.price_history_manager.has_sufficient_data('1h', self.config.fast_ema_period) else 0,
            'current_slow_ema': self.calculate_ema(
                self.price_history_manager.get_prices('1h'), self.config.slow_ema_period
            ) if self.price_history_manager.has_sufficient_data('1h', self.config.slow_ema_period) else 0
        })
        
        return base_metrics