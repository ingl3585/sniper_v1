"""
Core Data Models
Frozen Pydantic models for shared domain models.
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from logging_config import get_logger

log = get_logger(__name__)


class Bar(BaseModel):
    """OHLCV bar data with timestamp.
    
    Represents a complete price bar for any timeframe with full OHLCV data.
    """
    model_config = ConfigDict(frozen=True)
    
    timestamp: float = Field(..., description="Unix timestamp")
    open_price: float = Field(..., gt=0, description="Opening price")
    high_price: float = Field(..., gt=0, description="High price")
    low_price: float = Field(..., gt=0, description="Low price") 
    close_price: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Volume traded")
    
    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)."""
        return (self.high_price + self.low_price + self.close_price) / 3.0
    
    @property
    def range_price(self) -> float:
        """Calculate price range (High - Low).""" 
        return self.high_price - self.low_price
    
    @property
    def body_size(self) -> float:
        """Calculate candle body size (abs(Close - Open))."""
        return abs(self.close_price - self.open_price)
    
    @property
    def is_bullish(self) -> bool:
        """Check if bar is bullish (Close > Open)."""
        return self.close_price > self.open_price
    
    @property 
    def is_bearish(self) -> bool:
        """Check if bar is bearish (Close < Open)."""
        return self.close_price < self.open_price
    
    @property
    def is_doji(self, threshold: float = 0.1) -> bool:
        """Check if bar is a doji (small body relative to range).
        
        Args:
            threshold: Maximum body/range ratio for doji classification
        """
        if self.range_price == 0:
            return True
        return (self.body_size / self.range_price) <= threshold


class Tick(BaseModel):
    """Individual tick data point.
    
    Represents a single price/volume observation at a point in time.
    """
    model_config = ConfigDict(frozen=True)
    
    timestamp: float = Field(..., description="Unix timestamp")
    price: float = Field(..., gt=0, description="Tick price")
    volume: float = Field(..., ge=0, description="Tick volume")
    bid: Optional[float] = Field(None, gt=0, description="Bid price")
    ask: Optional[float] = Field(None, gt=0, description="Ask price")
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread if both are available."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price if bid/ask are available."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return None


class TradeSignal(BaseModel):
    """Trading signal with entry/exit parameters.
    
    Represents a complete trading signal with risk management parameters.
    """
    model_config = ConfigDict(frozen=True)
    
    timestamp: float = Field(..., description="Signal generation timestamp")
    action: int = Field(..., ge=1, le=2, description="Action: 1=Buy, 2=Sell")
    entry_price: float = Field(..., gt=0, description="Intended entry price")
    position_size: int = Field(..., gt=0, description="Position size in contracts")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence (0-1)")
    
    # Risk management
    stop_price: Optional[float] = Field(None, gt=0, description="Stop loss price")
    target_price: Optional[float] = Field(None, gt=0, description="Target price")
    use_stop: bool = Field(True, description="Whether to use stop loss")
    use_target: bool = Field(True, description="Whether to use target")
    
    # Metadata
    strategy_name: str = Field(..., description="Originating strategy name")
    timeframe: str = Field(..., description="Primary analysis timeframe")
    reason: str = Field(..., description="Signal generation reason")
    
    @property
    def is_long(self) -> bool:
        """Check if signal is for long position."""
        return self.action == 1
    
    @property
    def is_short(self) -> bool:
        """Check if signal is for short position."""
        return self.action == 2
    
    @property
    def risk_per_contract(self) -> Optional[float]:
        """Calculate risk per contract if stop is set."""
        if self.stop_price is not None:
            return abs(self.entry_price - self.stop_price)
        return None
    
    @property
    def reward_per_contract(self) -> Optional[float]:
        """Calculate reward per contract if target is set.""" 
        if self.target_price is not None:
            return abs(self.target_price - self.entry_price)
        return None
    
    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk/reward ratio."""
        if self.risk_per_contract is not None and self.reward_per_contract is not None:
            if self.risk_per_contract > 0:
                return self.reward_per_contract / self.risk_per_contract
        return None
    
    @property
    def total_risk(self) -> Optional[float]:
        """Calculate total dollar risk for position."""
        if self.risk_per_contract is not None:
            return self.risk_per_contract * self.position_size
        return None
    
    def validate_prices(self) -> bool:
        """Validate that stop and target prices make sense given entry and direction."""
        if self.is_long:
            # Long: stop should be below entry, target above
            if self.stop_price is not None and self.stop_price >= self.entry_price:
                return False
            if self.target_price is not None and self.target_price <= self.entry_price:
                return False
        elif self.is_short:
            # Short: stop should be above entry, target below
            if self.stop_price is not None and self.stop_price <= self.entry_price:
                return False
            if self.target_price is not None and self.target_price >= self.entry_price:
                return False
        return True


class Position(BaseModel):
    """Active trading position.
    
    Represents a current market position with P&L tracking.
    """
    model_config = ConfigDict(frozen=True)
    
    symbol: str = Field(..., description="Trading symbol")
    side: int = Field(..., ge=1, le=2, description="Position side: 1=Long, 2=Short")
    size: int = Field(..., ne=0, description="Position size (positive for long, negative for short)")
    entry_price: float = Field(..., gt=0, description="Average entry price")
    current_price: float = Field(..., gt=0, description="Current market price")
    entry_timestamp: float = Field(..., description="Position entry timestamp")
    
    # Risk management
    stop_price: Optional[float] = Field(None, gt=0, description="Stop loss price")
    target_price: Optional[float] = Field(None, gt=0, description="Target price")
    
    # P&L tracking
    realized_pnl: float = Field(0.0, description="Realized P&L from partial closes")
    commission_paid: float = Field(0.0, ge=0, description="Total commission paid")
    
    # Metadata
    strategy_name: str = Field(..., description="Originating strategy")
    original_signal_id: Optional[str] = Field(None, description="Original signal ID")
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == 1 and self.size > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == 2 and self.size < 0
    
    @property
    def abs_size(self) -> int:
        """Get absolute position size."""
        return abs(self.size)
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.is_long:
            return (self.current_price - self.entry_price) * self.abs_size
        else:  # short
            return (self.entry_price - self.current_price) * self.abs_size
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def net_pnl(self) -> float:
        """Calculate net P&L after commissions."""
        return self.total_pnl - self.commission_paid
    
    @property
    def pnl_percent(self) -> float:
        """Calculate P&L as percentage of position value."""
        position_value = self.entry_price * self.abs_size
        if position_value > 0:
            return (self.net_pnl / position_value) * 100.0
        return 0.0
    
    @property
    def duration_seconds(self) -> float:
        """Calculate position duration in seconds."""
        from time import time
        return time() - self.entry_timestamp
    
    @property
    def duration_minutes(self) -> float:
        """Calculate position duration in minutes."""
        return self.duration_seconds / 60.0
    
    def update_current_price(self, new_price: float) -> 'Position':
        """Create new Position instance with updated current price.
        
        Args:
            new_price: New current market price
            
        Returns:
            New Position instance with updated price
        """
        # Since model is frozen, create new instance
        return Position(
            symbol=self.symbol,
            side=self.side,
            size=self.size,
            entry_price=self.entry_price,
            current_price=new_price,
            entry_timestamp=self.entry_timestamp,
            stop_price=self.stop_price,
            target_price=self.target_price,
            realized_pnl=self.realized_pnl,
            commission_paid=self.commission_paid,
            strategy_name=self.strategy_name,
            original_signal_id=self.original_signal_id
        )
    
    def should_stop_out(self) -> bool:
        """Check if position should be stopped out."""
        if self.stop_price is None:
            return False
            
        if self.is_long:
            return self.current_price <= self.stop_price
        else:  # short
            return self.current_price >= self.stop_price
    
    def should_take_profit(self) -> bool:
        """Check if position should take profit."""
        if self.target_price is None:
            return False
            
        if self.is_long:
            return self.current_price >= self.target_price
        else:  # short
            return self.current_price <= self.target_price