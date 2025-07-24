"""
Message Types for NinjaTrader Communication
Dataclasses for market data, signals, and trade completions.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class MarketData:
    """Market data structure for multi-timeframe bars."""
    price_1m: list[float]
    price_5m: list[float]
    price_15m: list[float]
    price_30m: list[float]
    price_1h: list[float]
    volume_1m: list[float]
    volume_5m: list[float]
    volume_15m: list[float]
    volume_30m: list[float]
    volume_1h: list[float]
    account_balance: float
    buying_power: float
    daily_pnl: float
    unrealized_pnl: float
    open_positions: int
    current_price: float
    timestamp: int
    
    # Volatility fields
    volatility_1m: float = 0.0
    volatility_5m: float = 0.0
    volatility_15m: float = 0.0
    volatility_30m: float = 0.0
    volatility_1h: float = 0.0
    volatility: float = 0.0
    volatility_regime: str = 'medium'
    volatility_percentile: float = 0.5
    volatility_breakout: Optional[dict] = None
    
    # Tick-level data
    current_tick_price: float = 0.0
    tick_timestamp: int = 0
    tick_volume: float = 0.0
    
    def get_data_freshness_warning(self) -> str:
        """Get data freshness status."""
        import time
        current_time = int(time.time() * 1000)
        age = (current_time - self.timestamp) / 1000.0
        
        if age > 300:  # 5 minutes
            return f"Data is {age:.0f}s old - very stale"
        elif age > 120:  # 2 minutes
            return f"Data is {age:.0f}s old - stale"
        elif age > 60:  # 1 minute
            return f"Data is {age:.0f}s old - getting stale"
        else:
            return f"Data is {age:.0f}s old - fresh"


@dataclass
class TradeSignal:
    """Trading signal to send to NinjaScript."""
    action: int  # 1=buy, 2=sell, 0=hold
    position_size: int
    confidence: float
    use_stop: bool = False
    stop_price: float = 0.0
    use_target: bool = False
    target_price: float = 0.0
    entry_price: float = 0.0
    reason: str = ""
    timestamp: datetime = None
    order_id: str = ""
    order_type: str = "market"
    expected_slippage: float = 0.0
    urgency_score: float = 0.0


@dataclass
class TradeCompletion:
    """Trade completion notification from NinjaScript."""
    order_id: str
    symbol: str
    quantity: int
    price: float
    pnl: float
    timestamp: int