"""
Price History Manager
Centralized management of price history data across all timeframes.
"""
from collections import deque
from typing import Dict, List, Optional
import threading
from dataclasses import dataclass
import numpy as np
from src.infra.nt_bridge import MarketData


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


class PriceHistoryManager:
    """Centralized manager for price history across all timeframes."""
    
    def __init__(self, config=None):
        """Initialize the price history manager."""
        from src.config import SystemConfig
        
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
    
    def update_from_market_data(self, market_data: MarketData):
        """Update all timeframes from market data."""
        with self._lock:
            # Update 1m data
            if market_data.price_1m and market_data.volume_1m:
                self._update_timeframe('1m', market_data.price_1m, market_data.volume_1m)
                self.last_update['1m'] = market_data.timestamp
            
            # Update 5m data
            if market_data.price_5m and market_data.volume_5m:
                self._update_timeframe('5m', market_data.price_5m, market_data.volume_5m)
                self.last_update['5m'] = market_data.timestamp
            
            # Update 15m data
            if market_data.price_15m and market_data.volume_15m:
                self._update_timeframe('15m', market_data.price_15m, market_data.volume_15m)
                self.last_update['15m'] = market_data.timestamp
            
            # Update 30m data
            if market_data.price_30m and market_data.volume_30m:
                self._update_timeframe('30m', market_data.price_30m, market_data.volume_30m)
                self.last_update['30m'] = market_data.timestamp
            
            # Update 1h data
            if market_data.price_1h and market_data.volume_1h:
                self._update_timeframe('1h', market_data.price_1h, market_data.volume_1h)
                self.last_update['1h'] = market_data.timestamp
    
    def _update_timeframe(self, timeframe: str, price_list: List[float], volume_list: List[float]):
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
        
        # Timestamp will be updated by the calling update_from_market_data method
    
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
    
    def get_highs(self, timeframe: str, length: Optional[int] = None) -> List[float]:
        """Get high price data for a specific timeframe."""
        with self._lock:
            if timeframe not in self.timeframes:
                return []
            return self.timeframes[timeframe].get_highs(length)
    
    def get_lows(self, timeframe: str, length: Optional[int] = None) -> List[float]:
        """Get low price data for a specific timeframe."""
        with self._lock:
            if timeframe not in self.timeframes:
                return []
            return self.timeframes[timeframe].get_lows(length)
    
    def get_opens(self, timeframe: str, length: Optional[int] = None) -> List[float]:
        """Get opening price data for a specific timeframe."""
        with self._lock:
            if timeframe not in self.timeframes:
                return []
            return self.timeframes[timeframe].get_opens(length)
    
    def calculate_atr(self, timeframe: str, period: int = 14, length: Optional[int] = None) -> float:
        """Calculate proper ATR using OHLC data: ATR = SMA(TR) where TR = max(H-L, |H-C₍ₜ₋₁₎|, |L-C₍ₜ₋₁₎|)"""
        with self._lock:
            if timeframe not in self.timeframes:
                return 0.0
            
            highs = self.get_highs(timeframe, length)
            lows = self.get_lows(timeframe, length)
            closes = self.get_prices(timeframe, length)
            
            if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
                return closes[-1] * 0.01 if closes else 0.01
            
            true_ranges = []
            for i in range(1, min(len(highs), len(lows), len(closes))):
                # True Range formula: TR = max(H-L, |H-C₍ₜ₋₁₎|, |L-C₍ₜ₋₁₎|)
                high_low = highs[i] - lows[i]
                high_prev_close = abs(highs[i] - closes[i-1])
                low_prev_close = abs(lows[i] - closes[i-1])
                
                true_range = max(high_low, high_prev_close, low_prev_close)
                true_ranges.append(true_range)
            
            # ATR = Simple Moving Average of True Range over specified period
            if len(true_ranges) >= period:
                return np.mean(true_ranges[-period:])
            else:
                return np.mean(true_ranges) if true_ranges else 0.0
    
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
    
    def calculate_realized_volatility(self, timeframe: str, periods: tuple = (20, 60, 240)) -> Dict[str, float]:
        """Calculate realized volatility for multiple lookback periods."""
        with self._lock:
            if timeframe not in self.timeframes:
                return {}
                
            prices = list(self.timeframes[timeframe].prices)
            if len(prices) < max(periods):
                return {}
            
            # Calculate log returns
            log_returns = []
            for i in range(1, len(prices)):
                if prices[i] > 0 and prices[i-1] > 0:
                    log_returns.append(np.log(prices[i] / prices[i-1]))
            
            if len(log_returns) < max(periods):
                return {}
            
            # Calculate realized volatility for each period
            realized_vols = {}
            for period in periods:
                if len(log_returns) >= period:
                    recent_returns = log_returns[-period:]
                    # Calculate standard deviation and annualize
                    std_dev = np.std(recent_returns, ddof=1)
                    # Annualize based on timeframe
                    timeframe_multiplier = self._get_annualization_multiplier(timeframe)
                    annualized_vol = std_dev * np.sqrt(timeframe_multiplier)
                    realized_vols[f'{period}_period'] = annualized_vol
            
            return realized_vols
    
    def calculate_volatility_percentile(self, timeframe: str, current_vol: float, lookback: int = 480) -> float:
        """Calculate volatility percentile based on historical volatility."""
        with self._lock:
            if timeframe not in self.timeframes:
                return 0.5
                
            prices = list(self.timeframes[timeframe].prices)
            if len(prices) < lookback + 20:  # Need extra data for vol calculation
                return 0.5
            
            # Calculate rolling volatility
            vol_history = []
            for i in range(20, min(len(prices), lookback + 20)):
                window_prices = prices[i-20:i]
                if len(window_prices) >= 20:
                    log_returns = []
                    for j in range(1, len(window_prices)):
                        if window_prices[j] > 0 and window_prices[j-1] > 0:
                            log_returns.append(np.log(window_prices[j] / window_prices[j-1]))
                    
                    if len(log_returns) >= 19:
                        vol = np.std(log_returns, ddof=1)
                        timeframe_multiplier = self._get_annualization_multiplier(timeframe)
                        annualized_vol = vol * np.sqrt(timeframe_multiplier)
                        vol_history.append(annualized_vol)
            
            if len(vol_history) < 10:
                return 0.5
                
            # Calculate percentile
            vol_history = np.array(vol_history)
            percentile = np.mean(vol_history <= current_vol)
            return percentile
    
    def calculate_volatility_regime(self, timeframe: str, threshold: float = 0.5) -> str:
        """Determine volatility regime (low, medium, high)."""
        realized_vols = self.calculate_realized_volatility(timeframe, (20,))
        if not realized_vols:
            return 'medium'
            
        current_vol = realized_vols.get('20_period', 0.0)
        percentile = self.calculate_volatility_percentile(timeframe, current_vol)
        
        if percentile < 0.3:
            return 'low'
        elif percentile > 0.7:
            return 'high'
        else:
            return 'medium'
    
    def calculate_volatility_breakout(self, timeframe: str, threshold: float = 2.0) -> Dict[str, float]:
        """Detect volatility breakouts based on recent volatility expansion."""
        with self._lock:
            if timeframe not in self.timeframes:
                return {}
                
            prices = list(self.timeframes[timeframe].prices)
            if len(prices) < 100:  # Need sufficient data
                return {}
            
            # Calculate recent volatility (last 20 periods)
            recent_vol = self.calculate_realized_volatility(timeframe, (20,))
            if not recent_vol:
                return {}
            
            current_vol = recent_vol['20_period']
            
            # Calculate longer-term volatility distribution (last 100 periods)
            long_term_vols = []
            for i in range(40, min(len(prices), 120)):
                window_prices = prices[i-20:i]
                if len(window_prices) >= 20:
                    log_returns = []
                    for j in range(1, len(window_prices)):
                        if window_prices[j] > 0 and window_prices[j-1] > 0:
                            log_returns.append(np.log(window_prices[j] / window_prices[j-1]))
                    
                    if len(log_returns) >= 19:
                        vol = np.std(log_returns, ddof=1)
                        timeframe_multiplier = self._get_annualization_multiplier(timeframe)
                        annualized_vol = vol * np.sqrt(timeframe_multiplier)
                        long_term_vols.append(annualized_vol)
            
            if len(long_term_vols) < 20:
                return {}
            
            # Calculate z-score for breakout detection
            mean_vol = np.mean(long_term_vols)
            std_vol = np.std(long_term_vols, ddof=1)
            
            if std_vol > 0:
                z_score = (current_vol - mean_vol) / std_vol
                is_breakout = abs(z_score) > threshold
                breakout_direction = 'up' if z_score > threshold else 'down' if z_score < -threshold else 'none'
                
                return {
                    'current_vol': current_vol,
                    'mean_vol': mean_vol,
                    'z_score': z_score,
                    'is_breakout': is_breakout,
                    'breakout_direction': breakout_direction,
                    'breakout_strength': abs(z_score)
                }
            
            return {}
    
    def _get_annualization_multiplier(self, timeframe: str) -> float:
        """Get annualization multiplier based on timeframe."""
        # For financial markets, we typically use trading days (252) and trading hours (6.5h/day)
        # This gives us more realistic volatility numbers
        timeframe_multipliers = {
            '1m': 252 * 6.5 * 60,    # 98,280 minutes per trading year
            '5m': 252 * 6.5 * 12,    # 19,656 five-minute bars per trading year
            '15m': 252 * 6.5 * 4,    # 6,552 fifteen-minute bars per trading year
            '30m': 252 * 6.5 * 2,    # 3,276 thirty-minute bars per trading year
            '1h': 252 * 6.5          # 1,638 hourly bars per trading year
        }
        return timeframe_multipliers.get(timeframe, 1638)