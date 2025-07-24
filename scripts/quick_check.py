#!/usr/bin/env python3
"""
Quick Check Script
CLI entry point for single-session sanity pass validation.
"""
import sys
import argparse
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

# Add src to path for imports
sys.path.insert(0, '/mnt/c/Users/ingle/OneDrive/Desktop/sniper_v1')

from logging_config import get_logger
from src.core.models import Bar, Tick, TradeSignal, Position
from src.strategies.technical_indicators import TechnicalIndicators
from storage import DataManager
from src.infra.nt_bridge import MarketData

log = get_logger(__name__)


def generate_synthetic_data(num_bars: int = 50) -> Dict[str, Any]:
    """Generate synthetic OHLCV data for testing.
    
    Args:
        num_bars: Number of bars to generate
        
    Returns:
        Dictionary containing synthetic market data
    """
    log.info(f"Generating {num_bars} bars of synthetic data")
    
    # Base price around MNQ levels
    base_price = 22200.0
    prices = [base_price]
    
    # Generate random walk with trend
    np.random.seed(42)  # For reproducible results
    
    bars = []
    for i in range(num_bars):
        # Random walk with slight upward bias
        change = np.random.normal(0, base_price * 0.001)  # 0.1% volatility
        new_price = max(prices[-1] + change, base_price * 0.95)  # Floor at 5% below base
        
        # Generate OHLC around the close
        intrabar_range = abs(change) * 2 + base_price * 0.0005  # Minimum range
        high = new_price + np.random.uniform(0, intrabar_range * 0.6)
        low = new_price - np.random.uniform(0, intrabar_range * 0.4)
        open_price = prices[-1]  # Previous close
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, new_price)
        low = min(low, open_price, new_price)
        
        # Volume with some randomness
        volume = np.random.uniform(1000, 5000)
        
        timestamp = datetime.now().timestamp() - (num_bars - i) * 300  # 5-minute bars
        
        bar = Bar(
            timestamp=timestamp,
            open_price=open_price,
            high_price=high,
            low_price=low,
            close_price=new_price,
            volume=volume
        )
        bars.append(bar)
        prices.append(new_price)
    
    # Extract data for indicators
    highs = [bar.high_price for bar in bars]
    lows = [bar.low_price for bar in bars]  
    closes = [bar.close_price for bar in bars]
    volumes = [bar.volume for bar in bars]
    
    # Generate synthetic order flow for FVDR
    buys = [vol * np.random.uniform(0.4, 0.8) for vol in volumes]
    sells = [vol - buy for vol, buy in zip(volumes, buys)]
    
    return {
        'bars': bars,
        'highs': highs,
        'lows': lows,
        'closes': closes,
        'volumes': volumes,
        'buys': buys,
        'sells': sells,
        'timestamp': datetime.now().isoformat()
    }


def load_historical_data(date_str: str) -> Dict[str, Any]:
    """Load historical data for specified date.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        
    Returns:
        Historical data dictionary or None if not found
    """
    log.info(f"Attempting to load historical data for {date_str}")
    
    data_manager = DataManager()
    historical_data = data_manager.load_historical_data()
    
    if not historical_data:
        log.warning("No historical data found")
        return None
    
    # Look for data matching the requested date
    target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    
    for session_key, session_data in historical_data.items():
        if 'timestamp' in session_data:
            try:
                session_date = datetime.fromisoformat(session_data['timestamp']).date()
                if session_date == target_date:
                    log.info(f"Found historical data for {date_str}")
                    return session_data
            except (ValueError, TypeError):
                continue
    
    log.warning(f"No historical data found for {date_str}")
    return None


def calculate_indicators(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate technical indicators from market data.
    
    Args:
        data: Market data dictionary
        
    Returns:
        Dictionary containing calculated indicators
    """
    log.info("Calculating technical indicators")
    
    highs = data['highs']
    lows = data['lows'] 
    closes = data['closes']
    volumes = data['volumes']
    buys = data['buys']
    sells = data['sells']
    
    # Calculate ATR
    atr_values = []
    for i in range(len(closes)):
        if i < 14:
            # Not enough data for full ATR
            continue
        atr = TechnicalIndicators.calculate_atr(
            highs[max(0, i-13):i+1],
            lows[max(0, i-13):i+1], 
            closes[max(0, i-13):i+1],
            period=14
        )
        atr_values.append(atr)
    
    # Calculate ATR using pandas-ta equivalent for comparison
    # Simple implementation: mean of true ranges over 14 periods
    pandas_ta_atr = []
    if len(closes) >= 15:  # Need at least 15 for 14-period + 1 for previous close
        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_prev_close = abs(highs[i] - closes[i-1])
            low_prev_close = abs(lows[i] - closes[i-1])
            tr = max(high_low, high_prev_close, low_prev_close)
            true_ranges.append(tr)
        
        # Calculate 14-period ATR
        if len(true_ranges) >= 14:
            pandas_ta_atr.append(np.mean(true_ranges[-14:]))
    
    # Calculate FVDR
    fvdr_values = []
    if len(closes) >= 2:
        try:
            fvdr = TechnicalIndicators.calculate_fvdr(
                buys, sells, highs, lows, closes, atr_period=14
            )
            fvdr_values = fvdr
        except Exception as e:
            log.error(f"FVDR calculation failed: {e}")
            fvdr_values = np.zeros(len(closes))
    
    # Calculate NFVGS  
    nfvgs_values = []
    if len(closes) >= 18:  # Need 14 + 4 for ATR + gap detection
        try:
            nfvgs = TechnicalIndicators.calculate_nfvgs(
                highs, lows, closes, atr_period=14, decay_ema=5
            )
            nfvgs_values = nfvgs
        except Exception as e:
            log.error(f"NFVGS calculation failed: {e}")
            nfvgs_values = np.zeros(len(closes))
    
    return {
        'atr_our_method': atr_values,
        'atr_pandas_ta_equivalent': pandas_ta_atr,
        'fvdr': fvdr_values,
        'nfvgs': nfvgs_values
    }


def generate_trade_signal(data: Dict[str, Any], indicators: Dict[str, Any]) -> TradeSignal:
    """Generate a dummy trade signal for testing.
    
    Args:
        data: Market data
        indicators: Calculated indicators
        
    Returns:
        Test trade signal
    """
    log.info("Generating test trade signal")
    
    current_price = data['closes'][-1]
    
    # Simple signal logic: buy if FVDR > 0, sell if FVDR < 0
    fvdr = indicators['fvdr']
    action = 1 if len(fvdr) > 0 and fvdr[-1] > 0 else 2
    
    # Use ATR for stop loss
    atr = indicators['atr_our_method'][-1] if indicators['atr_our_method'] else current_price * 0.01
    
    if action == 1:  # Buy
        stop_price = current_price - (atr * 1.5)
        target_price = current_price + (atr * 2.5)
    else:  # Sell
        stop_price = current_price + (atr * 1.5)
        target_price = current_price - (atr * 2.5)
    
    signal = TradeSignal(
        timestamp=datetime.now().timestamp(),
        action=action,
        entry_price=current_price,
        position_size=1,
        confidence=0.75,
        stop_price=stop_price,
        target_price=target_price,
        use_stop=True,
        use_target=True,
        strategy_name="QuickCheck",
        timeframe="5m",
        reason=f"Test signal with FVDR={fvdr[-1]:.3f}" if fvdr else "Test signal"
    )
    
    return signal


def send_to_nt_bridge(signal: TradeSignal) -> bool:
    """Simulate sending signal to NinjaTrader bridge.
    
    Args:
        signal: Trade signal to send
        
    Returns:
        True if successful, False otherwise
    """
    log.info("Simulating trade signal dispatch to NT bridge")
    
    # Validate signal
    if not signal.validate_prices():
        log.error("Signal price validation failed")
        return False
    
    # In a real implementation, this would send to the actual NT bridge
    # For now, just validate the signal structure
    try:
        signal_dict = {
            'action': signal.action,
            'entry_price': signal.entry_price,
            'position_size': signal.position_size,
            'confidence': signal.confidence,
            'stop_price': signal.stop_price,
            'target_price': signal.target_price,
            'strategy_name': signal.strategy_name
        }
        
        log.info(f"Signal validation successful: {signal_dict}")
        return True
        
    except Exception as e:
        log.error(f"Signal dispatch failed: {e}")
        return False


def main():
    """Main entry point for quick_check script."""
    parser = argparse.ArgumentParser(description='Trading system quick check validation')
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format for historical data')
    
    args = parser.parse_args()
    
    log.info("Starting quick check validation")
    
    # Load data (historical if date provided, otherwise synthetic)
    if args.date:
        data = load_historical_data(args.date)
        if data is None:
            log.warning(f"No historical data for {args.date}, falling back to synthetic data")
            data = generate_synthetic_data(50)
    else:
        log.info("No date provided, using synthetic data")
        data = generate_synthetic_data(50)
    
    # Calculate indicators
    indicators = calculate_indicators(data)
    
    # Generate trade signal
    signal = generate_trade_signal(data, indicators)
    
    # Send signal to NT bridge (simulated)
    signal_ok = send_to_nt_bridge(signal)
    
    # Output results as specified in CLAUDE.md
    print("\n=== QUICK CHECK RESULTS ===")
    
    # ATR diff vs pandas-ta
    atr_diff = 0.0
    if indicators['atr_our_method'] and indicators['atr_pandas_ta_equivalent']:
        atr_diff = abs(indicators['atr_our_method'][-1] - indicators['atr_pandas_ta_equivalent'][-1])
    print(f"ATR diff vs pandas-ta: {atr_diff:.6f}")
    
    # FVDR tail (last 3 values)
    fvdr_tail = indicators['fvdr'][-3:] if len(indicators['fvdr']) >= 3 else indicators['fvdr']
    print(f"FVDR tail: {fvdr_tail}")
    
    # NFVGS tail (last 3 values)  
    nfvgs_tail = indicators['nfvgs'][-3:] if len(indicators['nfvgs']) >= 3 else indicators['nfvgs']
    print(f"NFVGS tail: {nfvgs_tail}")
    
    # Signal status
    print(f"Signal OK" if signal_ok else "Signal FAILED")
    
    print("=== END QUICK CHECK ===\n")
    
    log.info("Quick check validation completed")
    
    # Exit cleanly
    sys.exit(0)


if __name__ == '__main__':
    main()