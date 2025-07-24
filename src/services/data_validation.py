"""
Data Validation Service
Pydantic validators for inbound socket JSON and market data.
"""
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, validator, Field
import numpy as np

from logging_config import get_logger

log = get_logger(__name__)


class MarketDataValidator(BaseModel):
    """Validator for incoming market data from socket connections."""
    
    timestamp: float = Field(..., description="Unix timestamp")
    symbol: str = Field(..., description="Trading symbol")
    price: float = Field(..., gt=0, description="Current price")
    volume: float = Field(..., ge=0, description="Current volume")
    
    # Optional OHLC data
    open_price: Optional[float] = Field(None, gt=0, description="Opening price")
    high_price: Optional[float] = Field(None, gt=0, description="High price")
    low_price: Optional[float] = Field(None, gt=0, description="Low price")
    
    # Optional timeframe data
    price_1m: Optional[List[float]] = Field(None, description="1-minute price history")
    volume_1m: Optional[List[float]] = Field(None, description="1-minute volume history")
    price_5m: Optional[List[float]] = Field(None, description="5-minute price history")
    volume_5m: Optional[List[float]] = Field(None, description="5-minute volume history")
    price_15m: Optional[List[float]] = Field(None, description="15-minute price history")
    volume_15m: Optional[List[float]] = Field(None, description="15-minute volume history")
    price_30m: Optional[List[float]] = Field(None, description="30-minute price history")
    volume_30m: Optional[List[float]] = Field(None, description="30-minute volume history")
    price_1h: Optional[List[float]] = Field(None, description="1-hour price history")
    volume_1h: Optional[List[float]] = Field(None, description="1-hour volume history")
    
    @validator('price', 'open_price', 'high_price', 'low_price')
    def validate_prices(cls, v):
        """Validate that prices are positive and not NaN."""
        if v is not None:
            if np.isnan(v) or np.isinf(v):
                raise ValueError("Price cannot be NaN or infinite")
            if v <= 0:
                raise ValueError("Price must be positive")
        return v
    
    @validator('volume')
    def validate_volume(cls, v):
        """Validate volume is non-negative and not NaN."""
        if np.isnan(v) or np.isinf(v):
            raise ValueError("Volume cannot be NaN or infinite")
        if v < 0:
            raise ValueError("Volume cannot be negative")
        return v
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp is reasonable."""
        if v <= 0:
            raise ValueError("Timestamp must be positive")
        # Check if timestamp is reasonable (between 2020 and 2050)
        if v < 1577836800 or v > 2524608000:  # 2020-01-01 to 2050-01-01
            raise ValueError("Timestamp appears unreasonable")
        return v
    
    @validator('price_1m', 'price_5m', 'price_15m', 'price_30m', 'price_1h')
    def validate_price_arrays(cls, v):
        """Validate price arrays contain valid data."""
        if v is not None:
            if len(v) == 0:
                return v
            # Check for NaN/inf values
            if any(np.isnan(x) or np.isinf(x) or x <= 0 for x in v):
                raise ValueError("Price arrays cannot contain NaN, infinite, or non-positive values")
        return v
    
    @validator('volume_1m', 'volume_5m', 'volume_15m', 'volume_30m', 'volume_1h')
    def validate_volume_arrays(cls, v):
        """Validate volume arrays contain valid data."""
        if v is not None:
            if len(v) == 0:
                return v
            # Check for NaN/inf values
            if any(np.isnan(x) or np.isinf(x) or x < 0 for x in v):
                raise ValueError("Volume arrays cannot contain NaN, infinite, or negative values")
        return v


class TradeSignalValidator(BaseModel):
    """Validator for outgoing trade signals."""
    
    action: int = Field(..., ge=1, le=2, description="Trade action: 1=Buy, 2=Sell")
    position_size: int = Field(..., gt=0, description="Position size in contracts")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence")
    use_stop: bool = Field(False, description="Whether to use stop loss")
    stop_price: float = Field(0.0, ge=0.0, description="Stop loss price")
    use_target: bool = Field(False, description="Whether to use target price")
    target_price: float = Field(0.0, ge=0.0, description="Target price")
    
    @validator('stop_price')
    def validate_stop_price(cls, v, values):
        """Validate stop price is reasonable if stop is used."""
        if values.get('use_stop', False) and v <= 0:
            raise ValueError("Stop price must be positive when use_stop is True")
        return v
    
    @validator('target_price')
    def validate_target_price(cls, v, values):
        """Validate target price is reasonable if target is used."""
        if values.get('use_target', False) and v <= 0:
            raise ValueError("Target price must be positive when use_target is True")
        return v


class TechnicalIndicatorValidator(BaseModel):
    """Validator for technical indicator inputs."""
    
    prices: List[float] = Field(..., min_items=1, description="Price array")
    period: int = Field(..., gt=0, le=200, description="Calculation period")
    
    @validator('prices')
    def validate_prices(cls, v):
        """Validate price array contains valid data."""
        if not v:
            raise ValueError("Price array cannot be empty")
        if any(np.isnan(x) or np.isinf(x) or x <= 0 for x in v):
            raise ValueError("Prices cannot contain NaN, infinite, or non-positive values")
        return v


class OrderFlowValidator(BaseModel):
    """Validator for order flow data used in FVDR calculation."""
    
    buys: List[float] = Field(..., description="Buy volume array")
    sells: List[float] = Field(..., description="Sell volume array")
    highs: List[float] = Field(..., description="High price array")
    lows: List[float] = Field(..., description="Low price array")
    closes: List[float] = Field(..., description="Close price array")
    
    @validator('buys', 'sells')
    def validate_volumes(cls, v):
        """Validate volume arrays."""
        if any(np.isnan(x) or np.isinf(x) or x < 0 for x in v):
            raise ValueError("Volume arrays cannot contain NaN, infinite, or negative values")
        return v
    
    @validator('highs', 'lows', 'closes')
    def validate_prices(cls, v):
        """Validate price arrays."""
        if any(np.isnan(x) or np.isinf(x) or x <= 0 for x in v):
            raise ValueError("Price arrays cannot contain NaN, infinite, or non-positive values")
        return v
    
    @validator('highs', 'lows', 'closes')
    def validate_array_lengths(cls, v, values):
        """Validate all arrays have the same length."""
        if 'buys' in values and len(v) != len(values['buys']):
            raise ValueError("All arrays must have the same length")
        return v


def validate_market_data(data: Dict[str, Any]) -> MarketDataValidator:
    """Validate incoming market data.
    
    Args:
        data: Raw market data dictionary
        
    Returns:
        Validated MarketDataValidator instance
        
    Raises:
        ValidationError: If data is invalid
    """
    try:
        return MarketDataValidator(**data)
    except Exception as e:
        log.error(f"Market data validation failed: {e}")
        raise


def validate_trade_signal(data: Dict[str, Any]) -> TradeSignalValidator:
    """Validate outgoing trade signal.
    
    Args:
        data: Raw trade signal dictionary
        
    Returns:
        Validated TradeSignalValidator instance
        
    Raises:
        ValidationError: If data is invalid
    """
    try:
        return TradeSignalValidator(**data)
    except Exception as e:
        log.error(f"Trade signal validation failed: {e}")
        raise


def validate_indicator_input(prices: List[float], period: int) -> TechnicalIndicatorValidator:
    """Validate technical indicator inputs.
    
    Args:
        prices: Price array
        period: Calculation period
        
    Returns:
        Validated TechnicalIndicatorValidator instance
        
    Raises:
        ValidationError: If data is invalid
    """
    try:
        return TechnicalIndicatorValidator(prices=prices, period=period)
    except Exception as e:
        log.error(f"Technical indicator validation failed: {e}")
        raise


def validate_order_flow(buys: List[float], sells: List[float], highs: List[float], 
                       lows: List[float], closes: List[float]) -> OrderFlowValidator:
    """Validate order flow data for FVDR calculation.
    
    Args:
        buys: Buy volume array
        sells: Sell volume array  
        highs: High price array
        lows: Low price array
        closes: Close price array
        
    Returns:
        Validated OrderFlowValidator instance
        
    Raises:
        ValidationError: If data is invalid
    """
    try:
        return OrderFlowValidator(
            buys=buys, sells=sells, highs=highs, lows=lows, closes=closes
        )
    except Exception as e:
        log.error(f"Order flow validation failed: {e}")
        raise