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
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
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
        
        price_volume = [p * v for p, v in zip(prices, volumes)]
        return sum(price_volume) / sum(volumes)
    
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
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
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
        """Calculate simple ATR approximation using price ranges."""
        if len(prices) < 2:
            return prices[0] * 0.01 if prices else 0.01  # 1% default
        
        ranges = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        return np.mean(ranges[-period:]) if len(ranges) >= period else np.mean(ranges)
    
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