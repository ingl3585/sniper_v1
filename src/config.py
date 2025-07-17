"""
System Configuration
Centralized configuration for the entire trading system.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TradingConfig:
    """Trading system configuration."""
    # TCP Connection
    data_port: int = 5556
    signal_port: int = 5557
    
    # Risk Management
    max_daily_loss_pct: float = 0.5  # 50% daily loss limit
    max_position_size: int = 10
    min_account_balance: float = 1000.0
    
    # Strategy Settings
    enable_ml_allocator: bool = True
    enable_rl_execution: bool = True
    
    # System Settings
    log_level: str = "INFO"
    data_dir: str = "data"
    
    # Trading Hours (24-hour format)
    trading_blackout_start: int = 16  # 4 PM
    trading_blackout_end: int = 17    # 5 PM
    
    @classmethod
    def from_env(cls) -> 'TradingConfig':
        """Create config from environment variables."""
        defaults = cls()
        return cls(
            data_port=int(os.getenv('DATA_PORT', defaults.data_port)),
            signal_port=int(os.getenv('SIGNAL_PORT', defaults.signal_port)),
            max_daily_loss_pct=float(os.getenv('MAX_DAILY_LOSS_PCT', defaults.max_daily_loss_pct)),
            max_position_size=int(os.getenv('MAX_POSITION_SIZE', defaults.max_position_size)),
            min_account_balance=float(os.getenv('MIN_ACCOUNT_BALANCE', defaults.min_account_balance)),
            enable_ml_allocator=os.getenv('ENABLE_ML_ALLOCATOR', str(defaults.enable_ml_allocator).lower()).lower() == 'true',
            enable_rl_execution=os.getenv('ENABLE_RL_EXECUTION', str(defaults.enable_rl_execution).lower()).lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', defaults.log_level),
            data_dir=os.getenv('DATA_DIR', defaults.data_dir),
            trading_blackout_start=int(os.getenv('TRADING_BLACKOUT_START', defaults.trading_blackout_start)),
            trading_blackout_end=int(os.getenv('TRADING_BLACKOUT_END', defaults.trading_blackout_end))
        )


@dataclass
class MeanReversionConfig:
    """Mean reversion strategy configuration."""
    # Risk Management
    max_position_size: int = 10
    risk_per_trade: float = 0.02
    atr_lookback: int = 10
    stop_loss_atr_multiplier: float = 2.0
    min_confidence: float = 0.6
    
    # Strategy Parameters
    vwap_period: int = 20
    # deviation_threshold: float = 2.0  # Original conservative threshold
    deviation_threshold: float = 1.5     # Adjusted for more trading opportunities
    min_volume_threshold: float = 1000
    # rsi_oversold: float = 30          # Original conservative threshold
    # rsi_overbought: float = 70        # Original conservative threshold
    rsi_oversold: float = 35             # Adjusted for more trading opportunities
    rsi_overbought: float = 65           # Adjusted for more trading opportunities
    profit_target_multiplier: float = 1.5
    
    # 1-Minute Timeframe Confirmation Settings
    enable_1m_confirmation: bool = True
    deviation_threshold_1m: float = 1.0  # Tighter threshold for 1m due to noise
    min_1m_volume_threshold: float = 500  # Lower threshold for 1m bars
    max_trades_per_hour: int = 6         # Rate limiting for 1m-triggered trades
    min_time_between_trades_minutes: int = 5  # Minimum gap between 1m signals
    
    # RSI Calculation Settings
    rsi_period: int = 14                 # Standard RSI period
    rsi_use_wilder_smoothing: bool = True  # Use Wilder's exponential smoothing (standard)
    rsi_debug_logging: bool = False      # Enable detailed RSI calculation logging


@dataclass
class MomentumConfig:
    """Momentum strategy configuration."""
    # Risk Management
    max_position_size: int = 10
    risk_per_trade: float = 0.02
    atr_lookback: int = 10
    stop_loss_atr_multiplier: float = 2.0
    min_confidence: float = 0.6
    
    # Strategy Parameters
    fast_ema_period: int = 20
    # slow_ema_period: int = 100         # Original - requires 100 hours (~4+ days)
    slow_ema_period: int = 50            # Adjusted for faster testing - requires 50 hours (~2 days)
    # trend_strength_threshold: float = 0.6  # Original conservative threshold
    # trend_strength_threshold: float = 0.4   # Previously adjusted for more trading opportunities
    trend_strength_threshold: float = 0.15    # Adjusted for range-bound market conditions (July 2025)
    volume_confirmation_period: int = 10
    trail_stop_atr_multiplier: float = 3.0
    min_trend_duration: int = 5


@dataclass
class MetaAllocatorConfig:
    """Meta allocator ML model configuration."""
    model_path: str = "data/meta_allocator_model.pkl"
    lookback_period: int = 60
    retrain_interval: int = 1000
    feature_history_size: int = 5000


@dataclass
class PPOExecutionConfig:
    """PPO execution agent configuration."""
    model_path: str = "data/ppo_execution_model.zip"
    retrain_interval: int = 10000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64


@dataclass
class SystemConfig:
    """Complete system configuration."""
    trading: TradingConfig
    mean_reversion: MeanReversionConfig
    momentum: MomentumConfig
    meta_allocator: MetaAllocatorConfig
    ppo_execution: PPOExecutionConfig
    
    @classmethod
    def default(cls) -> 'SystemConfig':
        """Create default system configuration."""
        return cls(
            trading=TradingConfig(),
            mean_reversion=MeanReversionConfig(),
            momentum=MomentumConfig(),
            meta_allocator=MetaAllocatorConfig(),
            ppo_execution=PPOExecutionConfig()
        )
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Create system configuration from environment variables."""
        return cls(
            trading=TradingConfig.from_env(),
            mean_reversion=MeanReversionConfig(),
            momentum=MomentumConfig(),
            meta_allocator=MetaAllocatorConfig(),
            ppo_execution=PPOExecutionConfig()
        )