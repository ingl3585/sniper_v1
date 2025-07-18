#!/usr/bin/env python3
"""
Integration Test for Volatility Strategies
Tests configuration, strategy instantiation, and basic functionality.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_basic_integration():
    """Test basic integration without external dependencies."""
    print("=" * 50)
    print("INTEGRATION TEST - Volatility Strategies")
    print("=" * 50)
    
    # Test 1: Configuration Loading
    print("\n1. Testing Configuration Loading...")
    try:
        from src.config import SystemConfig, VolCarryConfig, VolBreakoutConfig
        
        config = SystemConfig.default()
        print(f"✓ SystemConfig loaded successfully")
        print(f"✓ Vol Carry Config available: {hasattr(config, 'vol_carry')}")
        print(f"✓ Vol Breakout Config available: {hasattr(config, 'vol_breakout')}")
        
        # Test config parameters
        print(f"✓ Vol Carry contango threshold: {config.vol_carry.contango_threshold}")
        print(f"✓ Vol Breakout z-threshold: {config.vol_breakout.breakout_z_threshold}")
        print(f"✓ Meta Allocator strategy count: {config.meta_allocator.strategy_count}")
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False
    
    # Test 2: Strategy Class Definitions
    print("\n2. Testing Strategy Class Definitions...")
    try:
        # Mock numpy for testing
        import sys
        class MockNumpy:
            def log(self, x): return x * 0.001  # Mock log function
            def std(self, x, ddof=1): return 0.02  # Mock std function
            def sqrt(self, x): return x ** 0.5  # Mock sqrt function
            def mean(self, x): return sum(x) / len(x) if x else 0  # Mock mean function
            def array(self, x): return x  # Mock array function
        
        sys.modules['numpy'] = MockNumpy()
        
        from src.strategies.volatility_carry import VolatilityCarryStrategy
        from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
        from src.utils.price_history_manager import PriceHistoryManager
        
        print(f"✓ VolatilityCarryStrategy imported successfully")
        print(f"✓ VolatilityBreakoutStrategy imported successfully")
        print(f"✓ PriceHistoryManager imported successfully")
        
        # Test strategy instantiation
        price_history = PriceHistoryManager(config)
        vol_carry = VolatilityCarryStrategy(config.vol_carry, config, price_history)
        vol_breakout = VolatilityBreakoutStrategy(config.vol_breakout, config, price_history)
        
        print(f"✓ Vol Carry Strategy instantiated: {vol_carry.name}")
        print(f"✓ Vol Breakout Strategy instantiated: {vol_breakout.name}")
        
    except Exception as e:
        print(f"✗ Strategy class test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Signal Processor Integration
    print("\n3. Testing Signal Processor Integration...")
    try:
        from src.orchestration.signal_processor import SignalProcessor
        
        # Mock existing strategies
        class MockStrategy:
            def __init__(self, name):
                self.name = name
            def generate_signal(self, market_data):
                return None
        
        mean_reversion = MockStrategy("MeanReversion")
        momentum = MockStrategy("Momentum")
        
        signal_processor = SignalProcessor(
            mean_reversion, momentum, vol_carry, vol_breakout
        )
        
        print(f"✓ SignalProcessor instantiated with 4 strategies")
        print(f"✓ Vol Carry Strategy: {signal_processor.vol_carry.name}")
        print(f"✓ Vol Breakout Strategy: {signal_processor.vol_breakout.name}")
        
    except Exception as e:
        print(f"✗ Signal Processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Price History Manager Volatility Methods
    print("\n4. Testing Price History Manager Volatility Methods...")
    try:
        # Test with mock data
        import collections
        
        # Create some mock price data
        mock_prices = [100 + i * 0.1 for i in range(100)]  # Simple trending prices
        
        # Test annualization multiplier
        multiplier_1m = price_history._get_annualization_multiplier('1m')
        multiplier_15m = price_history._get_annualization_multiplier('15m')
        
        print(f"✓ Annualization multiplier (1m): {multiplier_1m}")
        print(f"✓ Annualization multiplier (15m): {multiplier_15m}")
        
        # Test that multipliers are reasonable (not too large)
        assert multiplier_1m < 200000, "1m multiplier too large"
        assert multiplier_15m < 10000, "15m multiplier too large"
        
        print(f"✓ Annualization multipliers are reasonable")
        
    except Exception as e:
        print(f"✗ Price History Manager test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✓ ALL INTEGRATION TESTS PASSED!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_basic_integration()
    sys.exit(0 if success else 1)