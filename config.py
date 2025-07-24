"""
System Configuration
Centralized configuration for the entire trading system.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataBufferConfig:
    """Data buffer sizes for different timeframes."""
    buffer_1m: int = 6000    # ~100 hours
    buffer_5m: int = 1200    # ~4.2 days
    buffer_15m: int = 400    # ~4.2 days
    buffer_30m: int = 200    # ~4.2 days  
    buffer_1h: int = 100     # ~4.2 days



@dataclass
class NetworkConfig:
    """Network and connection settings."""
    host: str = "localhost"
    data_port: int = 5556
    signal_port: int = 5557
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 5.0
    connection_timeout: float = 60.0
    signal_timeout: float = 1.0
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    
    # Data validation thresholds
    stale_data_threshold: int = 300    # 5 minutes
    aging_data_threshold: int = 120    # 2 minutes
    fresh_data_threshold: int = 60     # 1 minute
    timestamp_epoch_threshold: int = 1_000_000_000  # year 2001 cutoff


@dataclass
class RiskManagementConfig:
    """Shared risk management configuration."""
    max_position_size: int = 5
    risk_per_trade: float = 0.25
    stop_loss_atr_multiplier: float = 2.0
    min_confidence: float = 0.6


@dataclass
class TechnicalAnalysisConfig:
    """Technical analysis default parameters."""
    rsi_period: int = 14
    bollinger_period: int = 20
    volatility_period: int = 20
    atr_period: int = 10
    lookback_bars: int = 20
    default_volatility: float = 0.02
    rsi_neutral: float = 50.0
    
    # Volatility calculation parameters
    realized_vol_periods: tuple = (20, 60, 240)  # 20min, 1hr, 4hr lookbacks
    vol_percentile_lookback: int = 480  # 8 hours for percentile calculation
    vol_regime_threshold: float = 0.5  # 50th percentile threshold for regime detection
    vol_breakout_threshold: float = 2.0  # Standard deviations for breakout detection
    vol_annualization_factor: int = 525600  # minutes per year (365 * 24 * 60)
    vol_smoothing_alpha: float = 0.1  # EMA smoothing factor for volatility
    
    # Financial constants
    annualization_factor: int = 1440  # minutes per trading day
    position_size_divisor: int = 100
    volume_normalization_1k: float = 1000.0
    volume_normalization_5k: float = 5000.0
    percentage_multiplier: int = 100


@dataclass
class TradingConfig:
    """Trading system configuration."""
    # Risk Management
    max_daily_loss_pct: float = 0.5  # 95% daily loss limit
    min_account_balance: float = 500.0
    
    # Strategy Settings
    enable_ml_allocator: bool = True
    enable_rl_execution: bool = True
    force_market_orders: bool = True  # Force market orders for guaranteed fills
    
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
            max_daily_loss_pct=float(os.getenv('MAX_DAILY_LOSS_PCT', defaults.max_daily_loss_pct)),
            min_account_balance=float(os.getenv('MIN_ACCOUNT_BALANCE', defaults.min_account_balance)),
            enable_ml_allocator=os.getenv('ENABLE_ML_ALLOCATOR', str(defaults.enable_ml_allocator).lower()).lower() == 'true',
            enable_rl_execution=os.getenv('ENABLE_RL_EXECUTION', str(defaults.enable_rl_execution).lower()).lower() == 'true',
            force_market_orders=os.getenv('FORCE_MARKET_ORDERS', str(defaults.force_market_orders).lower()).lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', defaults.log_level),
            data_dir=os.getenv('DATA_DIR', defaults.data_dir),
            trading_blackout_start=int(os.getenv('TRADING_BLACKOUT_START', defaults.trading_blackout_start)),
            trading_blackout_end=int(os.getenv('TRADING_BLACKOUT_END', defaults.trading_blackout_end))
        )


@dataclass
class MeanReversionConfig:
    """Mean reversion strategy configuration."""
    
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
    
    # RSI Calculation Settings (uses TechnicalAnalysisConfig.rsi_period)
    rsi_use_wilder_smoothing: bool = True  # Use Wilder's exponential smoothing (standard)
    rsi_debug_logging: bool = False      # Enable detailed RSI calculation logging


@dataclass
class MomentumConfig:
    """Momentum strategy configuration."""
    
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
class VolCarryConfig:
    """Volatility carry strategy configuration."""
    
    # Term Structure Thresholds
    contango_threshold: float = 0.15        # 15% slope for contango signal
    backwardation_threshold: float = 0.15   # 15% slope for backwardation signal
    
    # Carry Opportunity Validation
    min_carry_confidence: float = 0.5       # Minimum confidence for carry trades
    target_atr_multiplier: float = 2.0      # Target distance in ATR multiples
    
    # Volatility Regime Filters
    avoid_low_vol_sells: bool = True        # Don't sell vol in low vol regimes
    avoid_high_vol_buys: bool = True        # Don't buy vol in high vol regimes
    max_breakout_strength: float = 2.5      # Avoid trades during strong breakouts
    
    # Risk Management
    max_position_holding_hours: int = 24    # Max time to hold carry positions
    volatility_stop_multiplier: float = 1.5 # Stop based on volatility expansion


@dataclass
class VolBreakoutConfig:
    """Volatility breakout strategy configuration."""
    
    # Breakout Detection
    breakout_z_threshold: float = 2.0       # Z-score threshold for breakout detection
    min_regime_strength: float = 1.5        # Minimum strength for regime transition
    
    # Momentum and Price Thresholds
    momentum_threshold: float = 0.01        # 1% price momentum threshold
    price_extension_threshold: float = 1.5  # Price extension z-score threshold
    
    # Signal Validation
    min_volume_confirmation: float = 1.2    # 20% above average volume for confirmation
    target_atr_multiplier: float = 2.5      # Target distance in ATR multiples
    
    # Cooldown and Rate Limiting
    cooldown_minutes: int = 30              # Cooldown period between signals
    max_trades_per_session: int = 8         # Max breakout trades per session
    
    # Risk Management
    max_volatility_exposure: float = 0.25   # Max % of portfolio exposed to vol plays
    regime_transition_timeout: int = 60     # Minutes to wait for regime confirmation


@dataclass
class MetaAllocatorConfig:
    """Meta allocator ML model configuration."""
    model_path: str = "data/meta_allocator_model.pkl"
    lookback_period: int = 60
    retrain_interval: int = 1000
    feature_history_size: int = 5000
    
    # Strategy allocation parameters
    strategy_count: int = 4  # MeanReversion, Momentum, VolCarry, VolBreakout
    strategy_names: tuple = ("MeanReversion", "Momentum", "VolCarry", "VolBreakout")
    
    # Allocation constraints
    min_strategy_weight: float = 0.05    # Minimum 5% allocation per strategy
    max_strategy_weight: float = 0.60    # Maximum 60% allocation per strategy
    rebalance_threshold: float = 0.10    # Rebalance when allocation drift > 10%
    
    # Volatility strategy specific parameters
    vol_strategy_max_combined: float = 0.40  # Max combined weight for vol strategies
    vol_regime_adjustment: bool = True        # Adjust vol strategy weights based on regime
    
    # Feature engineering for 4-strategy model
    feature_lookback_bars: int = 240         # 4 hours of 1-minute bars
    volatility_feature_weight: float = 0.3   # Weight for volatility-based features
    momentum_feature_weight: float = 0.25    # Weight for momentum-based features
    mean_reversion_feature_weight: float = 0.25  # Weight for mean reversion features
    market_regime_feature_weight: float = 0.2    # Weight for market regime features


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
    network: NetworkConfig
    data_buffers: DataBufferConfig
    technical_analysis: TechnicalAnalysisConfig
    risk_management: RiskManagementConfig
    mean_reversion: MeanReversionConfig
    momentum: MomentumConfig
    vol_carry: VolCarryConfig
    vol_breakout: VolBreakoutConfig
    meta_allocator: MetaAllocatorConfig
    ppo_execution: PPOExecutionConfig
    
    @classmethod
    def default(cls) -> 'SystemConfig':
        """Create default system configuration."""
        return cls(
            trading=TradingConfig(),
            network=NetworkConfig(),
            data_buffers=DataBufferConfig(),
            technical_analysis=TechnicalAnalysisConfig(),
            risk_management=RiskManagementConfig(),
            mean_reversion=MeanReversionConfig(),
            momentum=MomentumConfig(),
            vol_carry=VolCarryConfig(),
            vol_breakout=VolBreakoutConfig(),
            meta_allocator=MetaAllocatorConfig(),
            ppo_execution=PPOExecutionConfig()
        )
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Create system configuration from environment variables."""
        return cls(
            trading=TradingConfig.from_env(),
            network=NetworkConfig(),
            data_buffers=DataBufferConfig(),
            technical_analysis=TechnicalAnalysisConfig(),
            risk_management=RiskManagementConfig(),
            mean_reversion=MeanReversionConfig(),
            momentum=MomentumConfig(),
            vol_carry=VolCarryConfig(),
            vol_breakout=VolBreakoutConfig(),
            meta_allocator=MetaAllocatorConfig(),
            ppo_execution=PPOExecutionConfig()
        )
    
