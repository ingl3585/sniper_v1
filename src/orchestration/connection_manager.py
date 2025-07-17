"""
Connection Management
Handles NinjaTrader connection lifecycle and initialization.
"""
import logging
import threading
from typing import Dict, Any


class ConnectionManager:
    """Manages NinjaTrader bridge connection lifecycle."""
    
    def __init__(self, bridge, data_manager):
        self.bridge = bridge
        self.data_manager = data_manager
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
            # Store historical data
            self.data_manager.store_historical_data(historical_data)
            self.logger.info("Historical data stored and strategies initialized")
        except Exception as e:
            self.logger.error(f"Error initializing strategies: {e}")
    
    def stop_bridge(self):
        """Stop the TCP bridge."""
        try:
            self.bridge.stop()
            self.logger.info("Bridge stopped")
        except Exception as e:
            self.logger.error(f"Error stopping bridge: {e}")