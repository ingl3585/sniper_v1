"""
Price History Manager
Centralized management of price history data across all timeframes.
"""
from collections import deque
from typing import Dict, List, Optional
import threading
from dataclasses import dataclass
from src.infra.nt_bridge import MarketData


@dataclass
class TimeframeData:
    """Data structure for a specific timeframe."""
    prices: deque
    volumes: deque
    max_length: int
    
    def __post_init__(self):
        """Initialize deques with maxlen for efficient memory usage."""
        self.prices = deque(maxlen=self.max_length)
        self.volumes = deque(maxlen=self.max_length)
    
    def add_data(self, price: float, volume: float):
        """Add new price and volume data."""
        self.prices.append(price)
        self.volumes.append(volume)
    
    def get_prices(self, length: Optional[int] = None) -> List[float]:
        """Get price list, optionally limited to specific length."""
        if length is None:
            return list(self.prices)
        return list(self.prices)[-length:] if length <= len(self.prices) else list(self.prices)
    
    def get_volumes(self, length: Optional[int] = None) -> List[float]:
        """Get volume list, optionally limited to specific length."""
        if length is None:
            return list(self.volumes)
        return list(self.volumes)[-length:] if length <= len(self.volumes) else list(self.volumes)
    
    def has_sufficient_data(self, min_length: int) -> bool:
        """Check if we have sufficient data for analysis."""
        return len(self.prices) >= min_length and len(self.volumes) >= min_length


class PriceHistoryManager:
    """Centralized manager for price history across all timeframes."""
    
    def __init__(self):
        """Initialize the price history manager."""
        # Define max lengths for each timeframe (optimized for memory)
        self.timeframes = {
            '1m': TimeframeData(deque(), deque(), 1000),   # ~16.7 hours
            '5m': TimeframeData(deque(), deque(), 500),    # ~1.7 days  
            '15m': TimeframeData(deque(), deque(), 300),   # ~3.1 days
            '30m': TimeframeData(deque(), deque(), 150),   # ~3.1 days
            '1h': TimeframeData(deque(), deque(), 100)     # ~4.2 days
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Last update timestamp for each timeframe
        self.last_update = {tf: 0 for tf in self.timeframes.keys()}
    
    def update_from_market_data(self, market_data: MarketData):
        """Update all timeframes from market data."""
        with self._lock:
            # Update 1m data
            if market_data.price_1m and market_data.volume_1m:
                self._update_timeframe('1m', market_data.price_1m, market_data.volume_1m)
            
            # Update 5m data
            if market_data.price_5m and market_data.volume_5m:
                self._update_timeframe('5m', market_data.price_5m, market_data.volume_5m)
            
            # Update 15m data
            if market_data.price_15m and market_data.volume_15m:
                self._update_timeframe('15m', market_data.price_15m, market_data.volume_15m)
            
            # Update 30m data
            if market_data.price_30m and market_data.volume_30m:
                self._update_timeframe('30m', market_data.price_30m, market_data.volume_30m)
            
            # Update 1h data
            if market_data.price_1h and market_data.volume_1h:
                self._update_timeframe('1h', market_data.price_1h, market_data.volume_1h)
    
    def _update_timeframe(self, timeframe: str, price_list: List[float], volume_list: List[float], market_data: MarketData):
        """Update a specific timeframe with new data."""
        if timeframe not in self.timeframes:
            return
        
        tf_data = self.timeframes[timeframe]
        
        # Add all new data points (NinjaTrader sends full history)
        # But we only want to add the latest points that we haven't seen
        if len(price_list) > len(tf_data.prices):
            # Add only the new data points
            start_index = len(tf_data.prices)
            for i in range(start_index, len(price_list)):
                if i < len(volume_list):
                    tf_data.add_data(price_list[i], volume_list[i])
        elif len(price_list) > 0 and len(volume_list) > 0:
            # Update the latest point if it's different
            if (len(tf_data.prices) == 0 or 
                tf_data.prices[-1] != price_list[-1] or 
                tf_data.volumes[-1] != volume_list[-1]):
                tf_data.add_data(price_list[-1], volume_list[-1])
        
        self.last_update[timeframe] = market_data.timestamp if hasattr(market_data, 'timestamp') else 0
    
    def get_prices(self, timeframe: str, length: Optional[int] = None) -> List[float]:
        """Get price data for a specific timeframe."""
        with self._lock:
            if timeframe not in self.timeframes:
                return []
            return self.timeframes[timeframe].get_prices(length)
    
    def get_volumes(self, timeframe: str, length: Optional[int] = None) -> List[float]:
        """Get volume data for a specific timeframe."""
        with self._lock:
            if timeframe not in self.timeframes:
                return []
            return self.timeframes[timeframe].get_volumes(length)
    
    def has_sufficient_data(self, timeframe: str, min_length: int) -> bool:
        """Check if timeframe has sufficient data for analysis."""
        with self._lock:
            if timeframe not in self.timeframes:
                return False
            return self.timeframes[timeframe].has_sufficient_data(min_length)
    
    def get_data_length(self, timeframe: str) -> int:
        """Get the number of data points for a timeframe."""
        with self._lock:
            if timeframe not in self.timeframes:
                return 0
            return len(self.timeframes[timeframe].prices)
    
    def get_current_price(self, timeframe: str) -> Optional[float]:
        """Get the most recent price for a timeframe."""
        with self._lock:
            if timeframe not in self.timeframes:
                return None
            prices = self.timeframes[timeframe].prices
            return prices[-1] if len(prices) > 0 else None
    
    def get_current_volume(self, timeframe: str) -> Optional[float]:
        """Get the most recent volume for a timeframe."""
        with self._lock:
            if timeframe not in self.timeframes:
                return None
            volumes = self.timeframes[timeframe].volumes
            return volumes[-1] if len(volumes) > 0 else None
    
    def clear_timeframe(self, timeframe: str):
        """Clear data for a specific timeframe."""
        with self._lock:
            if timeframe in self.timeframes:
                self.timeframes[timeframe].prices.clear()
                self.timeframes[timeframe].volumes.clear()
                self.last_update[timeframe] = 0
    
    def clear_all(self):
        """Clear all timeframe data."""
        with self._lock:
            for timeframe in self.timeframes:
                self.clear_timeframe(timeframe)
    
    def get_status(self) -> Dict[str, Dict[str, int]]:
        """Get status information for all timeframes."""
        with self._lock:
            status = {}
            for timeframe in self.timeframes:
                status[timeframe] = {
                    'data_points': len(self.timeframes[timeframe].prices),
                    'max_length': self.timeframes[timeframe].max_length,
                    'last_update': self.last_update[timeframe]
                }
            return status