"""
Application Constants
Centralized constants to eliminate magic numbers throughout the codebase.
"""
from typing import Final

# Time Constants
class TimeConstants:
    """Time-related constants in various units."""
    
    # Milliseconds
    MILLISECONDS_PER_SECOND: Final[int] = 1000
    MILLISECONDS_PER_MINUTE: Final[int] = 60 * 1000
    MILLISECONDS_PER_HOUR: Final[int] = 60 * 60 * 1000
    MILLISECONDS_PER_DAY: Final[int] = 24 * 60 * 60 * 1000
    
    # Seconds
    SECONDS_PER_MINUTE: Final[int] = 60
    SECONDS_PER_HOUR: Final[int] = 3600
    SECONDS_PER_DAY: Final[int] = 86400
    
    # Minutes
    MINUTES_PER_HOUR: Final[int] = 60
    MINUTES_PER_DAY: Final[int] = 1440
    
    # Hours
    HOURS_PER_DAY: Final[int] = 24
    
    # Trading-specific time constants
    TRADING_DAYS_PER_YEAR: Final[int] = 252
    TRADING_HOURS_PER_DAY: Final[int] = 24  # Futures trade 24/5
    TRADING_MINUTES_PER_HOUR: Final[int] = 60
    
    # Data freshness thresholds (seconds)
    DATA_FRESH_THRESHOLD: Final[int] = 60      # 1 minute
    DATA_STALE_THRESHOLD: Final[int] = 120     # 2 minutes
    DATA_VERY_STALE_THRESHOLD: Final[int] = 300  # 5 minutes
    
    # Tick data thresholds
    TICK_MAX_AGE_SECONDS: Final[float] = 10.0
    TICK_VALIDATION_WINDOW: Final[int] = 10  # seconds


class NetworkConstants:
    """Network and communication constants."""
    
    # Message size limits
    MAX_MESSAGE_SIZE_BYTES: Final[int] = 1024 * 1024  # 1MB
    HEADER_SIZE_BYTES: Final[int] = 4  # 4-byte message length header
    
    # Connection timeouts
    DEFAULT_CONNECTION_TIMEOUT: Final[float] = 30.0  # seconds
    SOCKET_TIMEOUT: Final[float] = 1.0  # seconds
    
    # Retry constants
    MAX_RECONNECT_ATTEMPTS: Final[int] = 3
    RECONNECT_DELAY_SECONDS: Final[float] = 2.0
    QUEUE_TIMEOUT_SECONDS: Final[float] = 1.0


class TechnicalAnalysisConstants:
    """Technical analysis and indicator constants."""
    
    # Default periods for indicators
    DEFAULT_RSI_PERIOD: Final[int] = 14
    DEFAULT_ATR_PERIOD: Final[int] = 14
    DEFAULT_EMA_FAST_PERIOD: Final[int] = 12
    DEFAULT_EMA_SLOW_PERIOD: Final[int] = 26
    DEFAULT_SMA_PERIOD: Final[int] = 20
    DEFAULT_BOLLINGER_PERIOD: Final[int] = 20
    
    # Volatility calculation constants
    MIN_PERIODS_FOR_VOLATILITY: Final[int] = 20
    VOLATILITY_LOOKBACK_PERIODS: Final[int] = 20
    ANNUALIZATION_FACTOR_1M: Final[float] = 252 * 24 * 60  # 1-minute bars
    ANNUALIZATION_FACTOR_5M: Final[float] = 252 * 24 * 12  # 5-minute bars  
    ANNUALIZATION_FACTOR_15M: Final[float] = 252 * 24 * 4  # 15-minute bars
    ANNUALIZATION_FACTOR_1H: Final[float] = 252 * 24       # 1-hour bars
    ANNUALIZATION_FACTOR_1D: Final[float] = 252            # Daily bars
    
    # Default volatility values
    DEFAULT_VOLATILITY: Final[float] = 0.02  # 2% annualized
    MIN_VOLATILITY: Final[float] = 0.005     # 0.5% minimum
    MAX_VOLATILITY: Final[float] = 0.1       # 10% maximum
    
    # Volatility regime thresholds
    LOW_VOLATILITY_THRESHOLD: Final[float] = 0.015   # 1.5%
    HIGH_VOLATILITY_THRESHOLD: Final[float] = 0.04   # 4.0%
    
    # Bollinger Bands
    BOLLINGER_STD_DEV_MULTIPLIER: Final[float] = 2.0
    
    # RSI boundaries
    RSI_OVERSOLD_THRESHOLD: Final[float] = 30.0
    RSI_OVERBOUGHT_THRESHOLD: Final[float] = 70.0
    RSI_NEUTRAL: Final[float] = 50.0


class ValidationConstants:
    """Data validation constants."""
    
    # Price validation
    MIN_VALID_PRICE: Final[float] = 0.1      # Minimum valid price
    MAX_VALID_PRICE: Final[float] = 100000.0 # Maximum valid price
    MIN_VALID_VOLUME: Final[float] = 0.0     # Minimum valid volume
    
    # Timestamp validation
    TIMESTAMP_FORMAT_LENGTH: Final[int] = 14  # 'yyyyMMddHHmmss' format
    MAX_TIMESTAMP_DEVIATION_HOURS: Final[float] = 24.0  # Maximum age in hours
    
    # Array validation
    MIN_ARRAY_LENGTH: Final[int] = 1
    MAX_ARRAY_LENGTH: Final[int] = 10000


class TradingConstants:
    """Trading and execution constants."""
    
    # Position sizing
    DEFAULT_POSITION_SIZE: Final[int] = 1
    MAX_POSITION_SIZE: Final[int] = 10
    
    # Risk management
    DEFAULT_RISK_PER_TRADE: Final[float] = 0.02     # 2% risk per trade
    MAX_DAILY_LOSS_PERCENT: Final[float] = 0.5      # 50% max daily loss
    DEFAULT_STOP_LOSS_ATR_MULTIPLIER: Final[float] = 1.2  # 1.2x ATR for stops
    
    # Confidence thresholds
    MIN_SIGNAL_CONFIDENCE: Final[float] = 0.6
    MAX_SIGNAL_CONFIDENCE: Final[float] = 0.95
    
    # Execution constants
    DEFAULT_SLIPPAGE_ESTIMATE: Final[float] = 0.0005  # 0.05% base slippage
    MAX_SLIPPAGE_ESTIMATE: Final[float] = 0.002       # 0.2% maximum slippage
    MARKET_IMPACT_MULTIPLIER: Final[float] = 0.001    # Market impact factor
    
    # MNQ-specific constants
    MNQ_TICK_SIZE: Final[float] = 0.25          # MNQ minimum tick
    MNQ_MIN_SPREAD: Final[float] = 0.25         # Minimum bid-ask spread
    MNQ_MAX_SPREAD: Final[float] = 1.00         # Maximum reasonable spread
    MNQ_TYPICAL_ATR_PERCENT: Final[float] = 0.0015  # 0.15% of price
    MNQ_MIN_ATR_PERCENT: Final[float] = 0.001        # 0.1% of price
    MNQ_MAX_ATR_PERCENT: Final[float] = 0.0025       # 0.25% of price


class StrategyConstants:
    """Strategy-specific constants."""
    
    # Mean reversion
    DEFAULT_VWAP_PERIOD: Final[int] = 20
    DEFAULT_DEVIATION_THRESHOLD: Final[float] = 2.0  # Z-score threshold
    CONFIDENCE_BASE: Final[float] = 0.6
    CONFIDENCE_Z_SCORE_MULTIPLIER: Final[float] = 0.35
    RSI_DISTANCE_MAX: Final[int] = 50  # Distance from neutral (50)
    RSI_CONFIDENCE_BOOST_MAX: Final[float] = 0.1  # 10% max boost
    
    # Momentum
    DEFAULT_TREND_STRENGTH_THRESHOLD: Final[float] = 0.5
    MIN_TREND_DURATION: Final[int] = 3  # Minimum bars for trend
    VOLUME_CONFIRMATION_MULTIPLIER: Final[float] = 2.0  # 2x average volume
    TREND_TARGET_ATR_MULTIPLIER_MIN: Final[float] = 2.0  # Minimum target
    TREND_TARGET_ATR_MULTIPLIER_MAX: Final[float] = 5.0  # Maximum target
    
    # Volatility
    MIN_VOLATILITY_PERIODS: Final[int] = 100  # Minimum data points
    VOLATILITY_BREAKOUT_Z_THRESHOLD: Final[float] = 2.5
    MOMENTUM_THRESHOLD: Final[float] = 0.02  # 2% price movement
    PRICE_EXTENSION_THRESHOLD: Final[float] = 2.0  # Z-score for extension
    
    # General
    DEBUG_PRICE_HISTORY_LENGTH: Final[int] = 5   # Last N prices for debugging
    CONFIDENCE_CALCULATION_PRECISION: Final[int] = 3  # Decimal places for logging


class LoggingConstants:
    """Logging and monitoring constants."""
    
    # Log levels and formatting
    MAX_LOG_MESSAGE_LENGTH: Final[int] = 1000
    PRICE_DECIMAL_PLACES: Final[int] = 2
    PERCENTAGE_DECIMAL_PLACES: Final[int] = 3
    CONFIDENCE_DECIMAL_PLACES: Final[int] = 3
    VOLUME_DECIMAL_PLACES: Final[int] = 0
    
    # Performance monitoring
    PERFORMANCE_WARNING_THRESHOLD_MS: Final[float] = 100.0  # 100ms
    MEMORY_WARNING_THRESHOLD_MB: Final[float] = 500.0       # 500MB


class RISK:
    """Risk management multipliers and sizing constants."""
    
    # Position sizing
    PER_TRADE: Final[float] = 0.02         # 2% risk per trade
    MAX_POSITION_SIZE: Final[float] = 0.1  # 10% max position size
    MAX_DAILY_LOSS: Final[float] = 0.05    # 5% max daily loss
    
    # ATR-based multipliers
    STOP_LOSS_ATR_MULTIPLIER: Final[float] = 1.2    # 1.2x ATR for stops
    TARGET_ATR_MULTIPLIER: Final[float] = 2.5       # 2.5x ATR for targets
    TRAIL_STOP_ATR_MULTIPLIER: Final[float] = 1.5   # 1.5x ATR for trailing stops
    
    # Portfolio-level constraints
    MAX_CORRELATED_POSITIONS: Final[int] = 3        # Max correlated positions
    MAX_TOTAL_EXPOSURE: Final[float] = 0.3          # 30% max total exposure


# Convenience access to common constants
COMMON_PERIODS = TechnicalAnalysisConstants
TIME = TimeConstants  
NETWORK = NetworkConstants
VALIDATION = ValidationConstants
TRADING = TradingConstants
STRATEGY = StrategyConstants