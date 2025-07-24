"""
Storage Management
DuckDB helpers and data persistence functionality for trading system.
"""
import logging
import os
import json
from collections import deque
from typing import Dict, List, Optional, Any
import threading
from dataclasses import dataclass
import numpy as np

from logging_config import get_logger

log = get_logger(__name__)


@dataclass
class TimeframeData:
    """Data structure for a specific timeframe."""
    prices: deque
    volumes: deque
    highs: deque
    lows: deque
    opens: deque
    max_length: int
    
    def __post_init__(self):
        """Initialize deques with maxlen for efficient memory usage."""
        self.prices = deque(maxlen=self.max_length)  # closes
        self.volumes = deque(maxlen=self.max_length)
        self.highs = deque(maxlen=self.max_length)
        self.lows = deque(maxlen=self.max_length)
        self.opens = deque(maxlen=self.max_length)
    
    def add_data(self, price: float, volume: float, high: float = None, low: float = None, open_price: float = None):
        """Add new OHLCV data."""
        self.prices.append(price)  # close
        self.volumes.append(volume)
        self.highs.append(high if high is not None else price)
        self.lows.append(low if low is not None else price)
        self.opens.append(open_price if open_price is not None else price)
    
    def get_prices(self, length: Optional[int] = None) -> List[float]:
        """Get price list (closes), optionally limited to specific length."""
        if length is None:
            return list(self.prices)
        return list(self.prices)[-length:] if length <= len(self.prices) else list(self.prices)
    
    def get_volumes(self, length: Optional[int] = None) -> List[float]:
        """Get volume list, optionally limited to specific length."""
        if length is None:
            return list(self.volumes)
        return list(self.volumes)[-length:] if length <= len(self.volumes) else list(self.volumes)
    
    def get_highs(self, length: Optional[int] = None) -> List[float]:
        """Get high prices, optionally limited to specific length."""
        if length is None:
            return list(self.highs)
        return list(self.highs)[-length:] if length <= len(self.highs) else list(self.highs)
    
    def get_lows(self, length: Optional[int] = None) -> List[float]:
        """Get low prices, optionally limited to specific length."""
        if length is None:
            return list(self.lows)
        return list(self.lows)[-length:] if length <= len(self.lows) else list(self.lows)
    
    def get_opens(self, length: Optional[int] = None) -> List[float]:
        """Get opening prices, optionally limited to specific length."""
        if length is None:
            return list(self.opens)
        return list(self.opens)[-length:] if length <= len(self.opens) else list(self.opens)
    
    def has_sufficient_data(self, min_length: int) -> bool:
        """Check if we have sufficient data for analysis."""
        return len(self.prices) >= min_length and len(self.volumes) >= min_length


class DataManager:
    """Simple data manager for storing and retrieving trading data."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize data manager with storage directory.
        
        Args:
            data_dir: Directory for data storage
        """
        self.data_dir = data_dir
        self.logger = get_logger(__name__)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def store_historical_data(self, historical_data: Dict[str, Any]) -> None:
        """Store historical data to disk.
        
        Args:
            historical_data: Dictionary containing historical market data
        """
        try:
            if not historical_data:
                log.warning("No historical data to store")
                return
            
            # Simple JSON storage for now
            file_path = os.path.join(self.data_dir, "historical_data.json")
            with open(file_path, 'w') as f:
                json.dump(historical_data, f, indent=2, default=str)
            
            log.info(f"Successfully stored historical data to {file_path}")
            
        except Exception as e:
            log.error(f"Error storing historical data: {e}")
    
    def load_historical_data(self) -> Dict[str, Any]:
        """Load historical data from disk.
        
        Returns:
            Dictionary containing historical market data
        """
        try:
            file_path = os.path.join(self.data_dir, "historical_data.json")
            if not os.path.exists(file_path):
                log.info(f"No historical data file found at {file_path}")
                return {}
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            log.info(f"Successfully loaded historical data from {file_path}")
            return data
            
        except Exception as e:
            log.error(f"Error loading historical data: {e}")
            return {}


class PriceHistoryManager:
    """Centralized manager for price history across all timeframes."""
    
    def __init__(self, config=None):
        """Initialize the price history manager.
        
        Args:
            config: System configuration object
        """
        from config import SystemConfig
        
        if config is None:
            config = SystemConfig.default()
        
        # Define max lengths for each timeframe from config
        buffers = config.data_buffers
        self.timeframes = {
            '1m': TimeframeData(deque(), deque(), deque(), deque(), deque(), buffers.buffer_1m),
            '5m': TimeframeData(deque(), deque(), deque(), deque(), deque(), buffers.buffer_5m),
            '15m': TimeframeData(deque(), deque(), deque(), deque(), deque(), buffers.buffer_15m),
            '30m': TimeframeData(deque(), deque(), deque(), deque(), deque(), buffers.buffer_30m),
            '1h': TimeframeData(deque(), deque(), deque(), deque(), deque(), buffers.buffer_1h)
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Last update timestamp for each timeframe
        self.last_update = {tf: 0 for tf in self.timeframes.keys()}
    
    def update_from_market_data(self, market_data) -> None:
        """Update all timeframes from market data.
        
        Args:
            market_data: MarketData object containing price/volume information
        """
        with self._lock:
            # Update 1m data
            if hasattr(market_data, 'price_1m') and market_data.price_1m and market_data.volume_1m:
                self._update_timeframe('1m', market_data.price_1m, market_data.volume_1m)
                self.last_update['1m'] = market_data.timestamp
            
            # Update 5m data
            if hasattr(market_data, 'price_5m') and market_data.price_5m and market_data.volume_5m:
                self._update_timeframe('5m', market_data.price_5m, market_data.volume_5m)
                self.last_update['5m'] = market_data.timestamp
            
            # Update 15m data
            if hasattr(market_data, 'price_15m') and market_data.price_15m and market_data.volume_15m:
                self._update_timeframe('15m', market_data.price_15m, market_data.volume_15m)
                self.last_update['15m'] = market_data.timestamp
            
            # Update 30m data
            if hasattr(market_data, 'price_30m') and market_data.price_30m and market_data.volume_30m:
                self._update_timeframe('30m', market_data.price_30m, market_data.volume_30m)
                self.last_update['30m'] = market_data.timestamp
            
            # Update 1h data
            if hasattr(market_data, 'price_1h') and market_data.price_1h and market_data.volume_1h:
                self._update_timeframe('1h', market_data.price_1h, market_data.volume_1h)
                self.last_update['1h'] = market_data.timestamp
    
    def _update_timeframe(self, timeframe: str, price_list: List[float], volume_list: List[float]) -> None:
        """Update a specific timeframe with new data.
        
        Args:
            timeframe: Timeframe identifier (e.g., '1m', '5m')
            price_list: List of price values
            volume_list: List of volume values
        """
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
    
    def get_prices(self, timeframe: str, length: Optional[int] = None) -> List[float]:
        """Get price data for a specific timeframe.
        
        Args:
            timeframe: Timeframe identifier
            length: Optional limit on number of data points
            
        Returns:
            List of price values
        """
        with self._lock:
            if timeframe not in self.timeframes:
                return []
            return self.timeframes[timeframe].get_prices(length)
    
    def get_volumes(self, timeframe: str, length: Optional[int] = None) -> List[float]:
        """Get volume data for a specific timeframe.
        
        Args:
            timeframe: Timeframe identifier
            length: Optional limit on number of data points
            
        Returns:
            List of volume values
        """
        with self._lock:
            if timeframe not in self.timeframes:
                return []
            return self.timeframes[timeframe].get_volumes(length)
    
    def get_highs(self, timeframe: str, length: Optional[int] = None) -> List[float]:
        """Get high price data for a specific timeframe.
        
        Args:
            timeframe: Timeframe identifier
            length: Optional limit on number of data points
            
        Returns:
            List of high price values
        """
        with self._lock:
            if timeframe not in self.timeframes:
                return []
            return self.timeframes[timeframe].get_highs(length)
    
    def get_lows(self, timeframe: str, length: Optional[int] = None) -> List[float]:
        """Get low price data for a specific timeframe.
        
        Args:
            timeframe: Timeframe identifier
            length: Optional limit on number of data points
            
        Returns:
            List of low price values
        """
        with self._lock:
            if timeframe not in self.timeframes:
                return []
            return self.timeframes[timeframe].get_lows(length)
    
    def get_opens(self, timeframe: str, length: Optional[int] = None) -> List[float]:
        """Get opening price data for a specific timeframe.
        
        Args:
            timeframe: Timeframe identifier
            length: Optional limit on number of data points
            
        Returns:
            List of opening price values
        """
        with self._lock:
            if timeframe not in self.timeframes:
                return []
            return self.timeframes[timeframe].get_opens(length)
    
    def get_data_length(self, timeframe: str) -> int:
        """Get the number of data points for a timeframe.
        
        Args:
            timeframe: Timeframe identifier
            
        Returns:
            Number of data points available
        """
        with self._lock:
            if timeframe not in self.timeframes:
                return 0
            return len(self.timeframes[timeframe].prices)
    
    def has_sufficient_data(self, timeframe: str, min_length: int) -> bool:
        """Check if timeframe has sufficient data for analysis.
        
        Args:
            timeframe: Timeframe identifier
            min_length: Minimum number of data points required
            
        Returns:
            True if sufficient data is available
        """
        with self._lock:
            if timeframe not in self.timeframes:
                return False
            return self.timeframes[timeframe].has_sufficient_data(min_length)
    
    def clear_all(self) -> None:
        """Clear all timeframe data."""
        with self._lock:
            for timeframe in self.timeframes:
                self.timeframes[timeframe].prices.clear()
                self.timeframes[timeframe].volumes.clear()
                self.timeframes[timeframe].highs.clear()
                self.timeframes[timeframe].lows.clear()
                self.timeframes[timeframe].opens.clear()
                self.last_update[timeframe] = 0
    
    def get_status(self) -> Dict[str, Dict[str, int]]:
        """Get status information for all timeframes.
        
        Returns:
            Dictionary containing status information for each timeframe
        """
        with self._lock:
            status = {}
            for timeframe in self.timeframes:
                status[timeframe] = {
                    'data_points': len(self.timeframes[timeframe].prices),
                    'max_length': self.timeframes[timeframe].max_length,
                    'last_update': self.last_update[timeframe]
                }
            return status
    
    def calculate_realized_volatility(self, timeframe: str, periods: tuple = (20, 60)) -> Dict[str, float]:
        """Calculate realized volatility for different periods.
        
        Args:
            timeframe: Timeframe identifier (e.g., '15m')
            periods: Tuple of periods to calculate (short_period, long_period)
            
        Returns:
            Dictionary with volatility calculations
        """
        with self._lock:
            if timeframe not in self.timeframes:
                return {}
            
            prices = self.timeframes[timeframe].get_prices()
            if len(prices) < max(periods):
                return {}
            
            results = {}
            
            for period in periods:
                if len(prices) >= period:
                    # Calculate log returns
                    recent_prices = prices[-period:]
                    returns = []
                    for i in range(1, len(recent_prices)):
                        if recent_prices[i-1] > 0 and recent_prices[i] > 0:
                            ret = np.log(recent_prices[i] / recent_prices[i-1])
                            returns.append(ret)
                    
                    if returns:
                        # Annualized volatility
                        vol = np.std(returns) * np.sqrt(252)  # Assuming 252 trading days
                        results[f'{period}_period'] = vol
            
            return results
    
    def calculate_volatility_breakout(self, timeframe: str, z_threshold: float = 2.0) -> Dict[str, Any]:
        """Calculate volatility breakout signals.
        
        Args:
            timeframe: Timeframe identifier
            z_threshold: Z-score threshold for breakout detection
            
        Returns:
            Dictionary with breakout analysis
        """
        with self._lock:
            if timeframe not in self.timeframes:
                return {}
            
            prices = self.timeframes[timeframe].get_prices()
            if len(prices) < 50:  # Need sufficient data
                return {'is_breakout': False, 'reason': 'insufficient_data'}
            
            # Calculate rolling volatility (using 20-period windows)
            volatilities = []
            for i in range(20, len(prices)):
                window = prices[i-20:i]
                returns = []
                for j in range(1, len(window)):
                    if window[j-1] > 0 and window[j] > 0:
                        ret = np.log(window[j] / window[j-1])
                        returns.append(ret)
                
                if returns:
                    vol = np.std(returns)
                    volatilities.append(vol)
            
            if len(volatilities) < 20:
                return {'is_breakout': False, 'reason': 'insufficient_volatility_data'}
            
            # Current volatility vs historical average
            current_vol = volatilities[-1]
            mean_vol = np.mean(volatilities[:-1])  # Exclude current
            std_vol = np.std(volatilities[:-1])
            
            if std_vol == 0:
                z_score = 0
            else:
                z_score = (current_vol - mean_vol) / std_vol
            
            # Determine breakout
            is_breakout = abs(z_score) > z_threshold
            direction = 'up' if z_score > z_threshold else 'down' if z_score < -z_threshold else 'none'
            
            return {
                'is_breakout': is_breakout,
                'breakout_direction': direction,
                'breakout_strength': abs(z_score),
                'z_score': z_score,
                'current_vol': current_vol,
                'mean_vol': mean_vol,
                'std_vol': std_vol
            }
    
    def calculate_volatility_regime(self, timeframe: str) -> str:
        """Calculate current volatility regime.
        
        Args:
            timeframe: Timeframe identifier
            
        Returns:
            Volatility regime: 'low', 'medium', 'high'
        """
        with self._lock:
            if timeframe not in self.timeframes:
                return 'medium'
            
            prices = self.timeframes[timeframe].get_prices()
            if len(prices) < 100:
                return 'medium'
            
            # Calculate recent volatility
            recent_returns = []
            for i in range(1, min(21, len(prices))):  # Last 20 periods
                if prices[-i-1] > 0 and prices[-i] > 0:
                    ret = abs(np.log(prices[-i] / prices[-i-1]))
                    recent_returns.append(ret)
            
            if not recent_returns:
                return 'medium'
            
            current_vol = np.mean(recent_returns)
            
            # Calculate historical volatility percentiles
            all_returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0 and prices[i] > 0:
                    ret = abs(np.log(prices[i] / prices[i-1]))
                    all_returns.append(ret)
            
            if not all_returns:
                return 'medium'
            
            percentile_33 = np.percentile(all_returns, 33)
            percentile_67 = np.percentile(all_returns, 67)
            
            if current_vol <= percentile_33:
                return 'low'
            elif current_vol >= percentile_67:
                return 'high'
            else:
                return 'medium'
    
    def calculate_atr(self, timeframe: str, period: int = 14, length: int = 50) -> float:
        """Calculate Average True Range (ATR).
        
        Args:
            timeframe: Timeframe identifier
            period: ATR period
            length: Number of bars to use for calculation
            
        Returns:
            ATR value
        """
        with self._lock:
            if timeframe not in self.timeframes:
                return 0.0
            
            tf_data = self.timeframes[timeframe]
            if len(tf_data.prices) < period + 1:
                return 0.0
            
            # Get OHLC data
            highs = tf_data.get_highs(length)
            lows = tf_data.get_lows(length)
            closes = tf_data.get_prices(length)
            
            if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
                return 0.0
            
            # Calculate True Range
            true_ranges = []
            for i in range(1, len(highs)):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            
            if len(true_ranges) < period:
                return 0.0
            
            # Calculate ATR using Wilder's smoothing
            atr = np.mean(true_ranges[:period])  # Initial ATR
            
            for i in range(period, len(true_ranges)):
                atr = ((atr * (period - 1)) + true_ranges[i]) / period
            
            return atr