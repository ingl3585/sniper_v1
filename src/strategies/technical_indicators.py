"""
Technical Indicators
Centralized calculation of all technical indicators to eliminate code duplication.
"""
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np

from logging_config import get_logger


class TechnicalIndicators:
    """Centralized technical indicator calculations."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.logger.set_context(component="technical_indicators")
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index).
        
        Args:
            prices: List of prices (typically close prices)
            period: RSI calculation period (default: 14)
            
        Returns:
            RSI value between 0 and 100
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        if len(gains) < period:
            return 50.0
            
        # Calculate initial averages
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        if avg_gain == 0:
            return 0.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return max(0.0, min(100.0, rsi))
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average.
        
        Args:
            prices: List of prices
            period: EMA calculation period
            
        Returns:
            Current EMA value
        """
        if len(prices) < period:
            return np.mean(prices) if prices else 0.0
        
        multiplier = 2 / (period + 1)
        
        # Start with SMA of first 'period' prices (proper EMA initialization)
        ema = np.mean(prices[:period])
        
        # Apply EMA formula for remaining prices
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average.
        
        Args:
            prices: List of prices
            period: SMA calculation period
            
        Returns:
            Current SMA value
        """
        if len(prices) < period:
            return np.mean(prices) if prices else 0.0
        
        return np.mean(prices[-period:])
    
    @staticmethod
    def calculate_atr(highs: List[float], lows: List[float], closes: List[float], 
                     period: int = 14) -> float:
        """Calculate Average True Range using proper True Range formula.
        
        Args:
            highs: List of high prices
            lows: List of low prices  
            closes: List of close prices
            period: ATR calculation period (default: 14)
            
        Returns:
            Current ATR value
        """
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return closes[-1] * 0.01 if closes else 0.01
        
        true_ranges = []
        for i in range(1, min(len(highs), len(lows), len(closes))):
            # True Range = max(H-L, |H-C₍t₋₁₎|, |L-C₍t₋₁₎|)
            high_low = highs[i] - lows[i]
            high_prev_close = abs(highs[i] - closes[i-1])
            low_prev_close = abs(lows[i] - closes[i-1])
            
            true_range = max(high_low, high_prev_close, low_prev_close)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return np.mean(true_ranges) if true_ranges else 0.01
        
        atr14 = np.mean(true_ranges[-period:])
        
        # Sanity check: ATR14 should match manual calculation of last 14 true ranges
        if period == 14 and len(true_ranges) >= 14:
            last14_tr = true_ranges[-14:]
            manual_atr14 = np.mean(last14_tr)
            assert abs(atr14 - manual_atr14) < 1e-6, f"ATR14 sanity check failed: {atr14} != {manual_atr14}"
        
        return atr14
    
    @staticmethod
    def calculate_atr_from_closes(prices: List[float], period: int = 14) -> float:
        """Calculate ATR approximation using only close prices (fallback method).
        
        Args:
            prices: List of close prices
            period: ATR calculation period
            
        Returns:
            Approximated ATR value
        """
        if len(prices) < 2:
            return prices[0] * 0.01 if prices else 0.01
        
        true_ranges = []
        for i in range(1, len(prices)):
            close_prev = prices[i-1]
            close_curr = prices[i]
            
            # MNQ-specific True Range approximation
            price_change = abs(close_curr - close_prev)
            
            # Base intrabar range for MNQ (typically 0.05-0.15% of price)
            base_range = close_curr * 0.0008  # 0.08% baseline
            
            # Gap component (inter-bar price change)
            gap_component = price_change
            
            # True Range = max of gap and typical intrabar range
            # Cap at reasonable levels for MNQ
            estimated_range = min(max(base_range, gap_component), close_curr * 0.002)  # Cap at 0.2%
            
            true_ranges.append(estimated_range)
        
        if len(true_ranges) < period:
            return np.mean(true_ranges) if true_ranges else 0.01
        
        # Use exponential smoothing like proper ATR (Wilder's smoothing)
        if len(true_ranges) >= period:
            # Start with simple average of first 'period' values
            atr = np.mean(true_ranges[:period])
            
            # Apply Wilder's smoothing to remaining values
            for tr in true_ranges[period:]:
                atr = ((atr * (period - 1)) + tr) / period
            
            return atr
        else:
            return np.mean(true_ranges[-period:])
    
    @staticmethod
    def calculate_vwap(prices: List[float], volumes: List[float], 
                      length: Optional[int] = None) -> float:
        """Calculate Volume Weighted Average Price.
        
        Args:
            prices: List of prices
            volumes: List of volumes
            length: Number of periods to use (None for all available)
            
        Returns:
            VWAP value
        """
        if not prices or not volumes or len(prices) != len(volumes):
            return np.mean(prices) if prices else 0.0
        
        # Use specified length or all available data
        if length and len(prices) > length:
            prices = prices[-length:]
            volumes = volumes[-length:]
        
        if not volumes or sum(volumes) == 0:
            return np.mean(prices)
        
        # VWAP = Σ(Price × Volume) / Σ(Volume)
        price_volume_sum = sum(p * v for p, v in zip(prices, volumes))
        volume_sum = sum(volumes)
        
        return price_volume_sum / volume_sum if volume_sum > 0 else np.mean(prices)
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, 
                                std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands.
        
        Args:
            prices: List of prices
            period: SMA calculation period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (lower_band, middle_band, upper_band)
        """
        if len(prices) < period:
            avg = np.mean(prices) if prices else 0.0
            return avg, avg, avg
        
        # Calculate middle band (SMA)
        middle_band = np.mean(prices[-period:])
        
        # Calculate standard deviation
        std = np.std(prices[-period:])
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return lower_band, middle_band, upper_band
    
    @staticmethod
    def calculate_momentum(prices: List[float], period: int = 10) -> float:
        """Calculate price momentum.
        
        Args:
            prices: List of prices
            period: Momentum calculation period
            
        Returns:
            Momentum value (current price / price N periods ago)
        """
        if len(prices) < period + 1:
            return 1.0  # No momentum
        
        current = prices[-1]
        past = prices[-(period + 1)]
        
        return current / past if past > 0 else 1.0
    
    @staticmethod
    def calculate_volatility(prices: List[float], period: int = 20, 
                           annualize: bool = True) -> float:
        """Calculate realized volatility.
        
        Args:
            prices: List of prices
            period: Calculation period
            annualize: Whether to annualize the volatility
            
        Returns:
            Volatility value
        """
        if len(prices) < 2:
            return 0.02  # Default volatility
        
        # Calculate log returns
        returns = []
        for i in range(1, min(len(prices), period + 1)):
            if prices[i-1] > 0:
                returns.append(np.log(prices[i] / prices[i-1]))
        
        if len(returns) < 2:
            return 0.02
        
        volatility = np.std(returns)
        
        if annualize:
            # Annualize based on trading timeframe
            volatility *= np.sqrt(252)  # 252 trading days
        
        return volatility
    
    @staticmethod
    def calculate_fvdr(buys: Union[List[float], np.ndarray], 
                      sells: Union[List[float], np.ndarray],
                      highs: Union[List[float], np.ndarray], 
                      lows: Union[List[float], np.ndarray],
                      closes: Union[List[float], np.ndarray],
                      atr_period: int = 14) -> np.ndarray:
        """Calculate Flow-Vol Drift Ratio (FVDR).
        
        Flow-Vol Drift Ratio measures the net order flow relative to price range,
        providing a momentum proxy based on order flow dynamics.
        
        Args:
            buys: Array of buy volume, shape (N,)
            sells: Array of sell volume, shape (N,)
            highs: Array of high prices, shape (N,)  
            lows: Array of low prices, shape (N,)
            closes: Array of close prices, shape (N,)
            atr_period: Period for Wilder's EMA smoothing (default: 14)
            
        Returns:
            Array of FVDR values, shape (N,)
            
        Logic:
            - net_flow[t] = buys[t] - sells[t]
            - tr[t] = max(high[t] - low[t], abs(high[t] - close[t-1]), abs(low[t] - close[t-1]))
            - clip tr > 0.0001 to avoid division by zero
            - raw_ratio = net_flow / tr  
            - Return: Wilder EMA(raw_ratio, period=14)
        """
        # Convert inputs to numpy arrays
        buys = np.asarray(buys, dtype=np.float64)
        sells = np.asarray(sells, dtype=np.float64)
        highs = np.asarray(highs, dtype=np.float64)
        lows = np.asarray(lows, dtype=np.float64)
        closes = np.asarray(closes, dtype=np.float64)
        
        # Validate input arrays
        if not all(len(arr) == len(buys) for arr in [sells, highs, lows, closes]):
            raise ValueError("All input arrays must have the same length")
        
        if len(buys) < 2:
            raise ValueError("Need at least 2 data points for FVDR calculation")
        
        # Calculate net flow
        net_flow = buys - sells
        
        # Calculate True Range (TR)
        true_ranges = np.zeros(len(highs))
        true_ranges[0] = highs[0] - lows[0]  # First bar: just high-low
        
        for i in range(1, len(highs)):
            # TR = max(H-L, |H-C₍t₋₁₎|, |L-C₍t₋₁₎|)
            high_low = highs[i] - lows[i]
            high_prev_close = abs(highs[i] - closes[i-1])
            low_prev_close = abs(lows[i] - closes[i-1])
            true_ranges[i] = max(high_low, high_prev_close, low_prev_close)
        
        # Clip true ranges to avoid division by zero
        true_ranges = np.maximum(true_ranges, 0.0001)
        
        # Calculate raw ratio
        raw_ratio = net_flow / true_ranges
        
        # Apply Wilder's EMA smoothing
        # Wilder's EMA: alpha = 1/period, more conservative than standard EMA
        alpha = 1.0 / atr_period
        fvdr = np.zeros_like(raw_ratio)
        fvdr[0] = raw_ratio[0]  # Initialize with first value
        
        for i in range(1, len(raw_ratio)):
            fvdr[i] = alpha * raw_ratio[i] + (1 - alpha) * fvdr[i-1]
        
        # Sanity check: FVDR should not contain NaN values
        assert not np.isnan(fvdr).any(), "FVDR contains NaN values"
        
        return fvdr
    
    @staticmethod  
    def calculate_nfvgs(highs: Union[List[float], np.ndarray],
                       lows: Union[List[float], np.ndarray], 
                       closes: Union[List[float], np.ndarray],
                       atr_period: int = 14,
                       decay_ema: int = 5) -> np.ndarray:
        """Calculate Normalized Fair Value Gap Strength (NFVGS).
        
        NFVGS identifies and scores fair value gaps (price gaps between candles),
        normalizing them by ATR and tracking their persistence over time.
        
        Args:
            highs: Array of high prices, shape (N,)
            lows: Array of low prices, shape (N,)
            closes: Array of close prices, shape (N,)
            atr_period: Period for ATR calculation (default: 14)
            decay_ema: Period for EMA smoothing of gap scores (default: 5)
            
        Returns:
            Array of NFVGS values, shape (N,). Positive = bullish gaps, Negative = bearish gaps
            
        Logic:
            - Bullish gap: low[t-1] > high[t-3] (gap up)
            - Bearish gap: high[t-1] < low[t-3] (gap down)
            - gap_size = |low[t-1] - high[t-3]| for bullish, |high[t-1] - low[t-3]| for bearish
            - score = gap_size / ATR14
            - Divide score by (age+1) while gap remains unfilled
            - Smooth with EMA(alpha=1/decay_ema)
            - Output: +bullish signal, -bearish signal
        """
        # Convert inputs to numpy arrays
        highs = np.asarray(highs, dtype=np.float64)  
        lows = np.asarray(lows, dtype=np.float64)
        closes = np.asarray(closes, dtype=np.float64)
        
        # Validate input arrays
        if not all(len(arr) == len(highs) for arr in [lows, closes]):
            raise ValueError("All input arrays must have the same length")
            
        if len(highs) < atr_period + 4:
            raise ValueError(f"Need at least {atr_period + 4} data points for NFVGS calculation")
        
        # Calculate ATR using True Range
        atr_values = np.zeros(len(closes))
        true_ranges = np.zeros(len(closes))
        
        # Calculate True Range
        true_ranges[0] = highs[0] - lows[0]
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_prev_close = abs(highs[i] - closes[i-1])
            low_prev_close = abs(lows[i] - closes[i-1])
            true_ranges[i] = max(high_low, high_prev_close, low_prev_close)
        
        # Calculate ATR using simple moving average
        for i in range(len(closes)):
            start_idx = max(0, i - atr_period + 1)
            atr_values[i] = np.mean(true_ranges[start_idx:i+1])
        
        # Initialize arrays for gap tracking
        nfvgs = np.zeros(len(closes))
        gap_scores = np.zeros(len(closes))
        
        # Track active gaps (up to last 50 bars to manage memory)
        max_gap_tracking = 50
        active_gaps = []  # List of (start_idx, gap_size, is_bullish, age)
        
        # Process each bar starting from index 3 (need t-3 lookback)
        for i in range(3, len(closes)):
            current_atr = max(atr_values[i], closes[i] * 0.001)  # Minimum ATR = 0.1% of price
            
            # Check for new bullish gap: low[t-1] > high[t-3]
            if lows[i-1] > highs[i-3]:
                gap_size = lows[i-1] - highs[i-3]
                gap_score = gap_size / current_atr
                active_gaps.append((i, gap_score, True, 0))  # (index, score, is_bullish, age)
                
            # Check for new bearish gap: high[t-1] < low[t-3]  
            elif highs[i-1] < lows[i-3]:
                gap_size = lows[i-3] - highs[i-1]  # Always positive
                gap_score = gap_size / current_atr
                active_gaps.append((i, gap_score, False, 0))  # (index, score, is_bearish, age)
            
            # Update existing gaps and check if filled
            updated_gaps = []
            total_gap_score = 0.0
            
            for gap_start, gap_score, is_bullish, age in active_gaps:
                # Check if gap is filled
                gap_filled = False
                
                if is_bullish:
                    # Bullish gap filled if current low trades back into the gap
                    original_high = highs[gap_start-3]
                    if lows[i] <= original_high:
                        gap_filled = True
                else:
                    # Bearish gap filled if current high trades back into the gap  
                    original_low = lows[gap_start-3]
                    if highs[i] >= original_low:
                        gap_filled = True
                
                if not gap_filled and age < max_gap_tracking:
                    # Gap still active, age it and add to score
                    aged_score = gap_score / (age + 1)
                    if is_bullish:
                        total_gap_score += aged_score
                    else:
                        total_gap_score -= aged_score
                    updated_gaps.append((gap_start, gap_score, is_bullish, age + 1))
            
            active_gaps = updated_gaps
            gap_scores[i] = total_gap_score
        
        # Apply EMA smoothing to gap scores
        alpha = 1.0 / decay_ema
        nfvgs[0] = gap_scores[0]
        
        for i in range(1, len(gap_scores)):
            nfvgs[i] = alpha * gap_scores[i] + (1 - alpha) * nfvgs[i-1]
        
        return nfvgs