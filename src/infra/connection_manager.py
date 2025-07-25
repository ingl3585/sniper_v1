"""
Connection Management
Handles NinjaTrader connection lifecycle and initialization.
"""
import threading
import time
from typing import Dict, Any
from logging_config import get_logger


class ConnectionManager:
    """Manages NinjaTrader bridge connection lifecycle."""
    
    def __init__(self, bridge, data_manager, price_history_manager=None):
        self.bridge = bridge
        self.data_manager = data_manager
        self.price_history_manager = price_history_manager
        self.logger = get_logger(__name__)
        self.logger.set_context(component='connection_manager')
        self.historical_data_received = False
    
    def start_bridge(self):
        """Start the TCP bridge."""
        self.bridge.start()
        self.logger.info("Bridge started")
    
    def wait_for_connection(self, shutdown_event: threading.Event):
        """Wait for NinjaTrader connection."""
        self.logger.info("Waiting for NinjaTrader connection...")
        
        while not shutdown_event.is_set():
            status = self.bridge.get_connection_status()
            if status['data_connected']:
                self.logger.info("Data connection established")
                break
                
            if shutdown_event.wait(1):
                raise ConnectionError("Shutdown requested while waiting for connection")
        
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
            
            # Mark historical data as received
            self.historical_data_received = True
            self.logger.info("Historical data stored and strategies initialized - Trading enabled")
        except Exception as e:
            self.logger.error(f"Error initializing strategies: {e}")
    
    def is_ready_for_trading(self) -> bool:
        """Check if system has received historical data and is ready for trading."""
        return self.historical_data_received
    
    def create_market_data_from_tick(self, tick_data: Dict[str, Any]):
        """Create MarketData object from tick data for real-time processing."""
        try:
            from src.infra.nt_bridge import MarketData
            import time
            
            # Create minimal MarketData with tick information
            # Use existing price history data and add current tick
            current_time_ms = int(time.time() * 1000)
            
            # Get current price from tick
            current_price = float(tick_data.get('current_tick_price', 0.0))
            if current_price <= 0:
                return None
                
            # Create MarketData with minimal required fields
            market_data = MarketData(
                price_1m=[],  # Would need historical data integration
                price_5m=[],
                price_15m=[],
                price_30m=[],
                price_1h=[],
                volume_1m=[],
                volume_5m=[],
                volume_15m=[],
                volume_30m=[],
                volume_1h=[],
                account_balance=float(tick_data.get('account_balance', 100000.0)),
                buying_power=float(tick_data.get('buying_power', 100000.0)),
                daily_pnl=float(tick_data.get('daily_pnl', 0.0)),
                unrealized_pnl=float(tick_data.get('unrealized_pnl', 0.0)),
                open_positions=int(tick_data.get('open_positions', 0)),
                current_price=current_price,
                timestamp=current_time_ms,
                current_tick_price=current_price,
                tick_timestamp=tick_data.get('tick_timestamp', current_time_ms),
                tick_volume=float(tick_data.get('tick_volume', 0.0))
            )
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error creating market data from tick: {e}")
            return None
    
    def process_historical_data(self, historical_data: Dict[str, Any]):
        """Process historical data received from NinjaTrader."""
        try:
            # Processing historical data structure
            
            # Initialize strategies with historical data
            self.initialize_strategies(historical_data)
            
        except Exception as e:
            self.logger.error(f"Error processing historical data: {e}")
    
    def _load_historical_data_to_manager(self, historical_data: Dict[str, Any]):
        """Load historical data into PriceHistoryManager."""
        timeframes = ['1m', '5m', '15m', '30m', '1h']
        
        # Accumulate all bars for all timeframes first
        accumulated_data = {
            'price_1m': [],
            'price_5m': [],
            'price_15m': [],
            'price_30m': [],
            'price_1h': [],
            'volume_1m': [],
            'volume_5m': [],
            'volume_15m': [],
            'volume_30m': [],
            'volume_1h': [],
            # Add OHLC data for technical indicators
            'open_1m': [],
            'open_5m': [],
            'open_15m': [],
            'open_30m': [],
            'open_1h': [],
            'high_1m': [],
            'high_5m': [],
            'high_15m': [],
            'high_30m': [],
            'high_1h': [],
            'low_1m': [],
            'low_5m': [],
            'low_15m': [],
            'low_30m': [],
            'low_1h': []
        }
        
        for timeframe in timeframes:
            bars_key = f'bars_{timeframe}'
            if bars_key in historical_data:
                bars = historical_data[bars_key]
                # Loading bars for timeframe
                
                # Accumulate all bars for this timeframe
                for bar in bars:
                    accumulated_data[f'price_{timeframe}'].append(bar.get('close', 0))
                    accumulated_data[f'volume_{timeframe}'].append(bar.get('volume', 0))
                    # Extract full OHLC data for technical indicators
                    accumulated_data[f'open_{timeframe}'].append(bar.get('open', 0))
                    accumulated_data[f'high_{timeframe}'].append(bar.get('high', 0))
                    accumulated_data[f'low_{timeframe}'].append(bar.get('low', 0))
        
        # Create single MarketData object with all accumulated historical data
        if any(len(data) > 0 for data in accumulated_data.values()):
            from src.infra.nt_bridge import MarketData
            market_data = MarketData(
                current_price=0,  # Not relevant for historical data load
                account_balance=0,
                buying_power=0,
                daily_pnl=0,
                unrealized_pnl=0,
                open_positions=0,
                timestamp=int(time.time() * 1000),
                **accumulated_data
            )
            
            # Single call to update with all historical data
            self.price_history_manager.update_from_market_data(market_data)
            # Historical data loaded
    
    
    def stop_bridge(self):
        """Stop the TCP bridge."""
        try:
            self.bridge.stop()
            self.logger.info("Bridge stopped")
        except Exception as e:
            self.logger.error(f"Error stopping bridge: {e}")