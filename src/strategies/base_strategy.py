"""
Base Strategy Framework
Abstract base class for all trading strategies.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import time
from src.infra.nt_bridge import MarketData, TradeSignal
from storage import PriceHistoryManager
from src.strategies.technical_indicators import TechnicalIndicators
from logging_config import get_logger


# StrategyConfig removed - using centralized config from src.core.config


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
    
    def __init__(self, name: str, strategy_config, system_config, price_history_manager: PriceHistoryManager = None):
        self.name = name
        self.config = strategy_config  # Strategy-specific config (MeanReversionConfig, etc.)
        self.system_config = system_config  # Full SystemConfig for shared settings
        self.position_size = 0
        self.last_signal_time = None
        self.atr_values = []
        self.price_history_manager = price_history_manager or PriceHistoryManager(system_config)
        self.logger = get_logger(f"strategies.{name.lower()}")
        self.logger.set_context(strategy=name.lower(), component='strategy')
        self.last_detailed_log_time = 0  # Rate limit detailed analysis logs
        self.log_interval = 30  # Log detailed analysis every 30 seconds
        self.last_signal_log_time = 0  # Rate limit signal condition logs
        self.signal_log_interval = 10  # Log signal conditions every 10 seconds
        
        # Price change tracking
        self.last_price = None
        self.last_timestamp = None
        self.price_unchanged_count = 0
        self.significant_price_change_threshold = 0.25  # $0.25 for MNQ
        
    @abstractmethod
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """Generate trading signal based on market data."""
        pass
    
    def should_log_detailed_analysis(self) -> bool:
        """Check if enough time has passed to log detailed analysis."""
        current_time = time.time()
        if current_time - self.last_detailed_log_time > self.log_interval:
            self.last_detailed_log_time = current_time
            self._enable_detailed_logging = True
            return True
        self._enable_detailed_logging = False
        return False
    
    def should_log_signal_conditions(self) -> bool:
        """Check if enough time has passed to log signal conditions."""
        current_time = time.time()
        if current_time - self.last_signal_log_time > self.signal_log_interval:
            self.last_signal_log_time = current_time
            return True
        return False
    
    def log_if_enabled(self, message: str, level: str = "info"):
        """Log message only if detailed logging is currently enabled."""
        if getattr(self, '_enable_detailed_logging', False):
            getattr(self.logger, level)(message)
    
    def log_detailed_if_time(self, message: str, level: str = "info"):
        """Log detailed analysis only if enough time has passed."""
        if self.should_log_detailed_analysis():
            getattr(self.logger, level)(message)
    
    def has_significant_price_change(self, market_data: MarketData) -> tuple[bool, str]:
        """Check if price has changed significantly since last check."""
        current_price = market_data.current_price
        current_timestamp = market_data.timestamp
        
        if self.last_price is None:
            self.last_price = current_price
            self.last_timestamp = current_timestamp
            return True, f"Initial price: ${current_price:.2f}"
        
        price_change = abs(current_price - self.last_price)
        time_change = current_timestamp - self.last_timestamp if self.last_timestamp else 0
        
        # Check for significant price movement
        if price_change >= self.significant_price_change_threshold:
            direction = "↑" if current_price > self.last_price else "↓"
            change_info = f"Price {direction} ${self.last_price:.2f} → ${current_price:.2f} (${price_change:.2f})"
            self.last_price = current_price
            self.last_timestamp = current_timestamp
            self.price_unchanged_count = 0
            return True, change_info
        
        # Check for timestamp advancement (data freshness)
        elif time_change > 1000:  # More than 1 second
            self.last_timestamp = current_timestamp
            self.price_unchanged_count += 1
            if self.price_unchanged_count % 10 == 0:  # Log every 10th unchanged tick
                return True, f"Price static at ${current_price:.2f} ({self.price_unchanged_count} ticks)"
            return False, f"Price unchanged: ${current_price:.2f}"
        
        # Price and time unchanged - possible stale data
        else:
            self.price_unchanged_count += 1
            if self.price_unchanged_count % 20 == 0:  # Log every 20th stale tick
                return True, f"⚠️ Possible stale data: ${current_price:.2f} ({self.price_unchanged_count} identical ticks)"
            return False, f"Stale data: ${current_price:.2f}"
    
    def update_price_history(self, market_data: MarketData):
        """Update price history using centralized manager."""
        self.price_history_manager.update_from_market_data(market_data)
    
    def calculate_atr(self, high_prices: List[float], low_prices: List[float], 
                     close_prices: List[float], period: int = 14) -> float:
        """Calculate Average True Range using proper True Range formula."""
        return TechnicalIndicators.calculate_atr(high_prices, low_prices, close_prices, period)
    
    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        return TechnicalIndicators.calculate_ema(prices, period)
    
    def calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        return TechnicalIndicators.calculate_sma(prices, period)
    
    def calculate_vwap(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate Volume Weighted Average Price."""
        return TechnicalIndicators.calculate_vwap(prices, volumes)
    
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
        """Calculate Relative Strength Index."""
        # Note: Debug parameter is ignored in the centralized service
        return TechnicalIndicators.calculate_rsi(prices, period)
    
    def calculate_volatility(self, prices: List[float], period: int = 20) -> float:
        """Calculate rolling volatility using standard deviation of returns."""
        return TechnicalIndicators.calculate_volatility(prices, period, annualize=True)
    
    def calculate_atr_simple(self, prices: List[float], period: int = 14) -> float:
        """Calculate ATR approximation using only close prices."""
        return TechnicalIndicators.calculate_atr_from_closes(prices, period)
    
    def calculate_position_size(self, market_data: MarketData, 
                              stop_price: float) -> int:
        """Calculate position size based on risk management."""
        current_price = market_data.current_price
        
        # Debug logging for position sizing
        self.logger.info(f"Position sizing debug: account_balance={market_data.account_balance}, "
                        f"buying_power={market_data.buying_power}, current_price={current_price}, "
                        f"stop_price={stop_price}")
        
        # Basic validation
        if current_price <= 0 or stop_price <= 0:
            self.logger.warning(f"Invalid prices: current={current_price}, stop={stop_price}")
            return 1
        
        # Calculate risk per share
        risk_per_share = abs(current_price - stop_price)
        if risk_per_share <= 0:
            self.logger.warning(f"Invalid risk per share: {risk_per_share}")
            return 1
        
        # For futures like MNQ, use a default account size if balance is 0 or unavailable
        account_balance = market_data.account_balance
        if account_balance <= 0:
            account_balance = 50000.0  # Default $50K account for MNQ
            self.logger.info(f"Using default account balance: ${account_balance}")
        
        # Calculate position size based on risk (0.25% risk per trade)
        risk_amount = account_balance * self.system_config.risk_management.risk_per_trade
        calculated_size = int(risk_amount / risk_per_share)
        
        self.logger.info(f"Risk calculation: risk_amount=${risk_amount:.2f}, "
                        f"risk_per_share=${risk_per_share:.2f}, calculated_size={calculated_size}")
        
        # Apply position limits - for MNQ, max 5 contracts
        max_size = self.system_config.risk_management.max_position_size
        
        # Don't use buying power calculation for futures - it's misleading
        final_size = max(1, min(calculated_size, max_size))
        
        self.logger.info(f"Final position size: {final_size} (max allowed: {max_size})")
        
        return final_size
    
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
        if abs(market_data.open_positions) >= self.system_config.risk_management.max_position_size:
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
    
    def calculate_volatility_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        """Calculate comprehensive volatility metrics for all timeframes."""
        volatility_metrics = {}
        
        timeframes = ['1m', '5m', '15m', '30m', '1h']
        config = self.system_config.technical_analysis
        
        for timeframe in timeframes:
            # Realized volatility for multiple periods
            realized_vols = self.price_history_manager.calculate_realized_volatility(
                timeframe, config.realized_vol_periods
            )
            
            # Volatility regime
            regime = self.price_history_manager.calculate_volatility_regime(
                timeframe, config.vol_regime_threshold
            )
            
            # Volatility percentile
            current_vol = realized_vols.get('20_period', 0.0)
            percentile = self.price_history_manager.calculate_volatility_percentile(
                timeframe, current_vol, config.vol_percentile_lookback
            )
            
            # Volatility breakout detection
            breakout_info = self.price_history_manager.calculate_volatility_breakout(
                timeframe, config.vol_breakout_threshold
            )
            
            volatility_metrics[timeframe] = {
                'realized_volatility': realized_vols,
                'regime': regime,
                'percentile': percentile,
                'breakout': breakout_info
            }
        
        return volatility_metrics
    
    def is_volatility_regime(self, timeframe: str, regime: str) -> bool:
        """Check if current volatility regime matches specified regime."""
        current_regime = self.price_history_manager.calculate_volatility_regime(timeframe)
        return current_regime == regime
    
    def is_volatility_breakout(self, timeframe: str, direction: str = 'any') -> bool:
        """Check if volatility breakout is occurring."""
        breakout_info = self.price_history_manager.calculate_volatility_breakout(timeframe)
        if not breakout_info or not breakout_info.get('is_breakout', False):
            return False
        
        if direction == 'any':
            return True
        elif direction == 'up':
            return breakout_info.get('breakout_direction') == 'up'
        elif direction == 'down':
            return breakout_info.get('breakout_direction') == 'down'
        
        return False
    
    def get_volatility_percentile(self, timeframe: str) -> float:
        """Get current volatility percentile for timeframe."""
        realized_vols = self.price_history_manager.calculate_realized_volatility(timeframe, (20,))
        if not realized_vols:
            return 0.5
        
        current_vol = realized_vols.get('20_period', 0.0)
        return self.price_history_manager.calculate_volatility_percentile(timeframe, current_vol)
    
    def get_volatility_z_score(self, timeframe: str) -> float:
        """Get volatility z-score for mean reversion analysis."""
        breakout_info = self.price_history_manager.calculate_volatility_breakout(timeframe)
        return breakout_info.get('z_score', 0.0) if breakout_info else 0.0