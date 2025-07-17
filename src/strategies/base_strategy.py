"""
Base Strategy Framework
Abstract base class for all trading strategies.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from src.infra.nt_bridge import MarketData, TradeSignal
from src.utils.price_history_manager import PriceHistoryManager


# StrategyConfig removed - using centralized config from src.config


@dataclass
class Signal:
    """Trading signal from strategy."""
    action: int  # 1=buy, 2=sell, 0=hold
    confidence: float
    entry_price: float
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    size: int = 1
    reason: str = ""
    timestamp: datetime = None


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, name: str, config, price_history_manager: PriceHistoryManager = None):
        self.name = name
        self.config = config
        self.position_size = 0
        self.last_signal_time = None
        self.atr_values = []
        self.price_history_manager = price_history_manager or PriceHistoryManager()
        
    @abstractmethod
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """Generate trading signal based on market data."""
        pass
    
    def update_price_history(self, market_data: MarketData):
        """Update price history using centralized manager."""
        self.price_history_manager.update_from_market_data(market_data)
    
    def calculate_atr(self, high_prices: List[float], low_prices: List[float], 
                     close_prices: List[float], period: int = 10) -> float:
        """Calculate Average True Range using proper True Range formula."""
        if len(high_prices) < period + 1 or len(low_prices) < period + 1 or len(close_prices) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, min(len(high_prices), len(low_prices), len(close_prices))):
            # True Range = max(H-L, |H-Cp|, |L-Cp|) where Cp = previous close
            high_low = high_prices[i] - low_prices[i]
            high_close_prev = abs(high_prices[i] - close_prices[i-1])
            low_close_prev = abs(low_prices[i] - close_prices[i-1])
            true_range = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return np.mean(true_ranges) if true_ranges else 0.0
        
        return np.mean(true_ranges[-period:])
    
    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        
        # Start with SMA of first 'period' prices (proper EMA initialization)
        ema = np.mean(prices[:period])
        
        # Apply EMA formula to remaining prices
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        # Optional validation logging for debugging
        if hasattr(self, 'logger') and hasattr(self, 'config') and hasattr(self.config, 'ema_debug_logging') and self.config.ema_debug_logging:
            sma_start = np.mean(prices[:period])
            final_ema = ema
            self.logger.debug(f"EMA Calculation: Period={period}, SMA_start={sma_start:.2f}, Final_EMA={final_ema:.2f}, Points={len(prices)}")
        
        return ema
    
    def calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return np.mean(prices)
        
        return np.mean(prices[-period:])
    
    def calculate_vwap(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate Volume Weighted Average Price."""
        if len(prices) != len(volumes) or len(prices) == 0:
            return 0.0
        
        # Validate inputs
        if any(v <= 0 for v in volumes):
            # Handle zero/negative volumes by using equal weights
            return sum(prices) / len(prices)
        
        if any(p <= 0 for p in prices):
            # Log warning for negative prices but continue
            print(f"Warning: Negative price found in VWAP calculation: {[p for p in prices if p <= 0]}")
        
        # Standard VWAP calculation: sum(price * volume) / sum(volume)
        price_volume = [p * v for p, v in zip(prices, volumes)]
        total_pv = sum(price_volume)
        total_volume = sum(volumes)
        
        if total_volume == 0:
            return 0.0
        
        vwap = total_pv / total_volume
        
        # Validate result
        if vwap <= 0:
            print(f"Warning: VWAP calculation resulted in non-positive value: {vwap}")
            return sum(prices) / len(prices)  # Fallback to simple average
        
        return vwap
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, 
                                std_dev: float = 2.0) -> tuple[float, float, float]:
        """Calculate Bollinger Bands (upper, middle, lower)."""
        if len(prices) < period:
            avg = np.mean(prices)
            return avg, avg, avg
        
        sma = self.calculate_sma(prices, period)
        std = np.std(prices[-period:])
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def calculate_rsi(self, prices: List[float], period: int = 14, debug: bool = False) -> float:
        """Calculate Relative Strength Index using Wilder's exponential smoothing method."""
        if len(prices) < period + 1:
            return 50.0
        
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        if debug:
            print(f"RSI Debug - Period: {period}, Price points: {len(prices)}")
            print(f"Last 5 deltas: {deltas[-5:]}")
            print(f"Last 5 gains: {gains[-5:]}")
            print(f"Last 5 losses: {losses[-5:]}")
        
        # Use Wilder's exponential smoothing (standard RSI method)
        # First calculation uses simple average for initial values
        if len(gains) < period:
            return 50.0
            
        # Initial values - simple average of first 'period' values
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if debug:
            print(f"Initial avg_gain: {avg_gain:.6f}")
            print(f"Initial avg_loss: {avg_loss:.6f}")
        
        # Apply Wilder's smoothing for subsequent values
        # Wilder's smoothing: New_avg = ((Previous_avg * (period-1)) + Current_value) / period
        for i in range(period, len(gains)):
            avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
            
            if debug and i >= len(gains) - 3:  # Debug last few calculations
                print(f"Step {i}: gain={gains[i]:.6f}, avg_gain={avg_gain:.6f}, avg_loss={avg_loss:.6f}")
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        if debug:
            print(f"Final: avg_gain={avg_gain:.6f}, avg_loss={avg_loss:.6f}, RS={rs:.6f}, RSI={rsi:.2f}")
        
        return rsi
    
    def calculate_volatility(self, prices: List[float], period: int = 20) -> float:
        """Calculate rolling volatility using standard deviation of returns."""
        if len(prices) < period + 1:
            return 0.02  # Default volatility
        
        # Calculate log returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append(np.log(prices[i] / prices[i-1]))
        
        if len(returns) < period:
            return np.std(returns) if returns else 0.02
        
        # Use the last 'period' returns
        recent_returns = returns[-period:]
        return np.std(recent_returns) * np.sqrt(1440)  # Annualized (1440 minutes per day)
    
    def calculate_atr_simple(self, prices: List[float], period: int = 10) -> float:
        """Calculate ATR approximation using only close prices with improved True Range estimation."""
        if len(prices) < 2:
            return prices[0] * 0.01 if prices else 0.01  # 1% default
        
        # Improved True Range approximation for close-only data
        true_ranges = []
        for i in range(1, len(prices)):
            # Estimate High/Low from close prices using typical intrabar range
            close_prev = prices[i-1]
            close_curr = prices[i]
            
            # Estimate typical intrabar range as percentage of price movement
            # This approximates the missing High/Low information
            price_change = abs(close_curr - close_prev)
            estimated_range = max(price_change, close_curr * 0.001)  # Minimum 0.1% range
            
            # For more volatile moves, increase the range estimate
            if price_change > close_prev * 0.01:  # > 1% move
                estimated_range *= 1.5  # Increase range estimate for large moves
            
            true_ranges.append(estimated_range)
        
        if len(true_ranges) < period:
            return np.mean(true_ranges) if true_ranges else 0.01
        
        # Use exponential smoothing like proper ATR (Wilder's smoothing)
        if len(true_ranges) >= period:
            # Start with simple average of first 'period' values
            atr = np.mean(true_ranges[:period])
            
            # Apply Wilder's smoothing to remaining values
            for tr in true_ranges[period:]:
                atr = ((atr * (period - 1)) + tr) / period
            
            # Optional validation logging for debugging
            if hasattr(self, 'logger') and hasattr(self, 'config') and hasattr(self.config, 'atr_debug_logging') and self.config.atr_debug_logging:
                simple_avg = np.mean(true_ranges)
                self.logger.debug(f"ATR Calculation: Period={period}, Simple_avg={simple_avg:.4f}, Wilder_ATR={atr:.4f}, Points={len(true_ranges)}")
            
            return atr
        else:
            return np.mean(true_ranges)
    
    def calculate_position_size(self, market_data: MarketData, 
                              stop_price: float) -> int:
        """Calculate position size based on risk management."""
        if market_data.account_balance <= 0:
            return 0
        
        current_price = market_data.current_price
        if current_price <= 0 or stop_price <= 0:
            return 1
        
        # Calculate risk per share
        risk_per_share = abs(current_price - stop_price)
        
        # Calculate position size based on risk
        risk_amount = market_data.account_balance * self.config.risk_per_trade
        position_size = int(risk_amount / risk_per_share)
        
        # Apply position limits
        max_size = min(self.config.max_position_size, 
                      int(market_data.buying_power / (current_price * 100)))
        
        return max(1, min(position_size, max_size))
    
    def should_trade(self, market_data: MarketData) -> bool:
        """Check if conditions are suitable for trading."""
        # Check account balance
        if market_data.account_balance <= 0:
            return False
        
        # Check daily loss limit (50% of equity)
        daily_loss_limit = market_data.account_balance * 0.5
        if market_data.daily_pnl < -daily_loss_limit:
            return False
        
        # Check position limits
        if abs(market_data.open_positions) >= self.config.max_position_size:
            return False
        
        # Check volatility (avoid trading in extreme conditions)
        if market_data.volatility > 0.1:  # 10% volatility threshold
            return False
        
        return True
    
    def create_trade_signal(self, signal: Signal, market_data: MarketData) -> TradeSignal:
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
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        return {
            'name': self.name,
            'position_size': self.position_size,
            'last_signal': self.last_signal_time,
            'price_history_status': self.price_history_manager.get_status()
        }