"""
Connection Management
Handles NinjaTrader connection lifecycle and initialization.
"""
import logging
import threading
import time
from typing import Dict, Any


class ConnectionManager:
    """Manages NinjaTrader bridge connection lifecycle."""
    
    def __init__(self, bridge, data_manager, price_history_manager=None):
        self.bridge = bridge
        self.data_manager = data_manager
        self.price_history_manager = price_history_manager
        self.logger = logging.getLogger(__name__)
    
    def start_bridge(self):
        """Start the TCP bridge."""
        self.bridge.start()
        self.logger.info("Bridge started")
    
    def wait_for_connection(self, shutdown_event: threading.Event):
        """Wait for NinjaTrader connection."""
        self.logger.info("Waiting for NinjaTrader connection...")
        
        while not self.bridge.is_connected() and not shutdown_event.is_set():
            if shutdown_event.wait(1):
                raise ConnectionError("Shutdown requested while waiting for connection")
            
            if self.bridge.data_socket is not None:
                self.logger.info("Data connection established")
                break
        
        if shutdown_event.is_set():
            raise ConnectionError("Shutdown requested")
            
        self.logger.info("NinjaTrader connection established")
    
    def initialize_strategies(self, historical_data: Dict[str, Any]):
        """Initialize strategies with historical data."""
        try:
            # Store historical data in database
            self.data_manager.store_historical_data(historical_data)
            
            # Load historical data into PriceHistoryManager for strategies
            if self.price_history_manager:
                self._load_historical_data_to_manager(historical_data)
                self.logger.info(f"Historical data loaded into PriceHistoryManager: "
                               f"1m={self.price_history_manager.get_data_length('1m')}, "
                               f"5m={self.price_history_manager.get_data_length('5m')}, "
                               f"15m={self.price_history_manager.get_data_length('15m')}, "
                               f"30m={self.price_history_manager.get_data_length('30m')}, "
                               f"1h={self.price_history_manager.get_data_length('1h')}")
            
            self.logger.info("Historical data stored and strategies initialized")
        except Exception as e:
            self.logger.error(f"Error initializing strategies: {e}")
    
    def _load_historical_data_to_manager(self, historical_data: Dict[str, Any]):
        """Load historical data into PriceHistoryManager."""
        timeframes = ['1m', '5m', '15m', '30m', '1h']
        
        for timeframe in timeframes:
            bars_key = f'bars_{timeframe}'
            if bars_key in historical_data:
                bars = historical_data[bars_key]
                self.logger.info(f"Loading {len(bars)} bars for {timeframe}")
                
                for bar in bars:
                    # Create MarketData object for each historical bar
                    market_data = self._create_market_data_from_bar(bar, timeframe)
                    if market_data:
                        self.price_history_manager.update_from_market_data(market_data)
    
    def _create_market_data_from_bar(self, bar: Dict[str, Any], timeframe: str):
        """Create MarketData object from historical bar data."""
        try:
            # Initialize with zeros for all timeframes
            price_data = {
                'price_1m': [],
                'price_5m': [],
                'price_15m': [],
                'price_30m': [],
                'price_1h': [],
                'volume_1m': [],
                'volume_5m': [],
                'volume_15m': [],
                'volume_30m': [],
                'volume_1h': []
            }
            
            # Set the specific timeframe data
            price_data[f'price_{timeframe}'] = [bar.get('close', 0)]
            price_data[f'volume_{timeframe}'] = [bar.get('volume', 0)]
            
            from src.infra.nt_bridge import MarketData
            return MarketData(
                current_price=bar.get('close', 0),
                account_balance=0,  # Not available in historical data
                buying_power=0,
                daily_pnl=0,
                unrealized_pnl=0,
                open_positions=0,
                timestamp=int(bar.get('timestamp', int(time.time() * 1000))),  # Use bar timestamp or current time
                **price_data
            )
        except Exception as e:
            self.logger.error(f"Error creating MarketData from bar: {e}")
            return None
    
    def stop_bridge(self):
        """Stop the TCP bridge."""
        try:
            self.bridge.stop()
            self.logger.info("Bridge stopped")
        except Exception as e:
            self.logger.error(f"Error stopping bridge: {e}")