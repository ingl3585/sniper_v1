#!/usr/bin/env python3
"""
MNQ Algo Trading System
Entry point for the algorithmic trading system.
"""
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import SystemConfig
from src.core.trading_system import TradingSystem
from logging_config import get_logger


def main():
    """Main entry point for the trading system."""
    logger = None
    try:
        # Load configuration
        config = SystemConfig.default()
        
        # Create and run trading system
        trading_system = TradingSystem(config)
        logger = trading_system.logger
        
        trading_system.run()
        
    except KeyboardInterrupt:
        if logger:
            logger.info("Trading system interrupted by user")
        sys.exit(0)
    except Exception as e:
        if logger:
            logger.critical(f"Fatal error in main: {e}", exc_info=True)
        else:
            print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()