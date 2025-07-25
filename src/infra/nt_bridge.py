"""
NinjaTrader Bridge v2 - Clean Architecture
Orchestrates communication with NinjaScript using composition pattern.
"""
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import logging

# Import message types
from src.infra.message_types import MarketData, TradeSignal, TradeCompletion

# Import the networking components
from src.infra.network_manager import NetworkManager
from src.infra.data_parser import DataParser
from src.infra.message_handler_factory import MessageHandlerFactory
from logging_config import get_logger


class NinjaTradeBridge:
    """Clean, focused bridge to NinjaTrader using composition pattern."""
    
    def __init__(self, config=None):
        from config import SystemConfig
        
        if config is None:
            config = SystemConfig.default()
        
        self.config = config
        self.logger = get_logger(__name__)
        self.logger.set_context(component='ninja_bridge')
        
        # Composition: Use focused components
        self.network_manager = NetworkManager(
            config.network.data_port, 
            config.network.signal_port, 
            config
        )
        self.data_parser = DataParser()
        
        # Data storage
        self.historical_data: Dict[str, Any] = {}
        self.latest_market_data: Optional[MarketData] = None
        
        # Thread safety
        self._data_lock = threading.RLock()
        
        # Message handling
        self.message_handler_factory = MessageHandlerFactory(self.logger, self._data_lock)
        
        # Event callbacks
        self.on_historical_data: Optional[Callable] = None
        self.on_market_data: Optional[Callable] = None
        self.on_realtime_tick: Optional[Callable] = None
        self.on_trade_completion: Optional[Callable] = None
        
        # Wire up network callback
        self.network_manager.on_message_received = self._handle_message
    
    def start(self):
        """Start the bridge."""
        self.logger.info("Starting NinjaTrader Bridge v2...", extra={'ports': {'data': self.config.network.data_port, 'signal': self.config.network.signal_port}})
        try:
            self.network_manager.start()
            self.logger.info("NinjaTrader Bridge v2 started successfully")
        except Exception as e:
            self.logger.error("Failed to start NinjaTrader Bridge", extra={'error': str(e)})
            raise
    
    def stop(self):
        """Stop the bridge."""
        self.logger.info("Stopping NinjaTrader Bridge v2...")
        self.network_manager.stop()
        self.logger.info("NinjaTrader Bridge v2 stopped")
    
    def send_signal(self, trade_signal: TradeSignal):
        """Send trading signal to NinjaScript."""
        signal_data = {
            "type": "trade_signal",
            "action": trade_signal.action,
            "position_size": trade_signal.position_size,  # Fixed: was "size", now "position_size"
            "confidence": trade_signal.confidence,
            "use_stop": trade_signal.use_stop,
            "stop_price": trade_signal.stop_price,
            "use_target": trade_signal.use_target,
            "target_price": trade_signal.target_price,
            "entry_price": trade_signal.entry_price,
            "reason": trade_signal.reason,
            "order_id": trade_signal.order_id,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        
        # Signal prepared for transmission
        
        try:
            self.network_manager.send_signal(signal_data)
        except Exception as e:
            self.logger.error("Failed to send trade signal", extra={'error': str(e), 'order_id': trade_signal.order_id})
    
    def get_latest_market_data(self) -> Optional[MarketData]:
        """Get the latest market data (thread-safe)."""
        with self._data_lock:
            return self.latest_market_data
    
    def get_historical_data(self) -> Dict[str, Any]:
        """Get historical data (thread-safe)."""
        with self._data_lock:
            return self.historical_data.copy()
    
    def is_connected(self) -> bool:
        """Check if bridge is connected."""
        status = self.network_manager.get_connection_status()
        return status['is_running'] and status['data_connected']
    
    def is_data_connected(self) -> bool:
        """Check if data connection is established."""
        status = self.network_manager.get_connection_status()
        return status['data_connected']
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status."""
        return self.network_manager.get_connection_status()
    
    def register_custom_message_handler(self, message_type: str, 
                                      handler_func: Callable[[Dict[str, Any], 'NinjaTradeBridge'], None]):
        """Register a custom message handler."""
        custom_handler = self.message_handler_factory.create_custom_handler(message_type, handler_func)
        self.message_handler_factory.register_handler(custom_handler)
        # Custom handler registered
    
    def get_message_handler_stats(self) -> Dict[str, Any]:
        """Get message handler statistics."""
        return self.message_handler_factory.get_handler_stats()
    
    def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming message using factory pattern."""
        message_type = message.get('type', 'unknown')
        # Processing message
        
        try:
            # Use factory pattern for message handling (replaces direct handling)
            self.message_handler_factory.handle_message(message, self)
            
        except Exception as e:
            self.logger.error("Error handling message", extra={'error': str(e), 'message_type': message_type})
    
    def _handle_historical_data(self, message: Dict[str, Any]):
        """Handle historical data from NinjaTrader."""
        self.logger.info("Received historical data from NinjaTrader")
        
        # Parse historical data
        historical_data = self.data_parser.parse_historical_data(message)
        
        with self._data_lock:
            self.historical_data = historical_data
        
        # Notify callback
        if self.on_historical_data:
            self.on_historical_data(historical_data)
    
    def _handle_live_data(self, message: Dict[str, Any]):
        """Handle live market data from NinjaTrader."""
        # Parse market data
        market_data = self._parse_market_data(message)
        
        with self._data_lock:
            self.latest_market_data = market_data
        
        # Notify callback
        if self.on_market_data:
            self.on_market_data(market_data)
    
    def _handle_realtime_tick(self, message: Dict[str, Any]):
        """Handle real-time tick data from NinjaTrader."""
        # Parse tick data
        tick_data = self._parse_realtime_tick(message)
        
        # Notify callback
        if self.on_realtime_tick:
            self.on_realtime_tick(tick_data)
    
    def _handle_trade_completion(self, message: Dict[str, Any]):
        """Handle trade completion from NinjaTrader."""
        # Parse trade completion
        trade_completion = self._parse_trade_completion(message)
        
        # Notify callback
        if self.on_trade_completion:
            self.on_trade_completion(trade_completion)
    
    def _parse_market_data(self, message: Dict[str, Any]) -> MarketData:
        """Parse market data message."""
        return self.data_parser.parse_market_data(message)
    
    def _parse_trade_completion(self, message: Dict[str, Any]) -> TradeCompletion:
        """Parse trade completion message."""
        return self.data_parser.parse_trade_completion(message)
    
    def _parse_realtime_tick(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Parse real-time tick data."""
        return self.data_parser.parse_realtime_tick(message)