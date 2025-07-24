"""
Message Handler Factory
Factory pattern for processing different types of messages from NinjaTrader.
"""
from typing import Dict, Any, Callable, Optional
from abc import ABC, abstractmethod
from threading import Lock

from logging_config import get_logger, TradingLogger


class MessageHandler(ABC):
    """Abstract base class for message handlers."""
    
    def __init__(self, logger: TradingLogger, data_lock: Lock):
        self.logger = logger
        self.data_lock = data_lock
    
    @abstractmethod
    def handle(self, message: Dict[str, Any], bridge_context: 'NinjaTradeBridge') -> None:
        """Handle a specific message type."""
        pass
    
    @abstractmethod
    def get_message_type(self) -> str:
        """Return the message type this handler processes."""
        pass


class HistoricalDataHandler(MessageHandler):
    """Handler for historical data messages."""
    
    def get_message_type(self) -> str:
        return 'historical_data'
    
    def handle(self, message: Dict[str, Any], bridge_context: 'NinjaTradeBridge') -> None:
        """Handle historical data message."""
        try:
            with self.data_lock:
                bridge_context.historical_data = message
            
            if bridge_context.on_historical_data:
                try:
                    bridge_context.on_historical_data(message)
                except Exception as e:
                    self.logger.error(f"Error in historical data callback: {e}")
            
            self.logger.info("Historical data loaded")
            
        except Exception as e:
            self.logger.error(f"Error handling historical data: {e}")


class LiveDataHandler(MessageHandler):
    """Handler for live market data messages."""
    
    def get_message_type(self) -> str:
        return 'live_data'
    
    def handle(self, message: Dict[str, Any], bridge_context: 'NinjaTradeBridge') -> None:
        """Handle live market data message."""
        try:
            market_data = bridge_context._parse_market_data(message)
            
            with self.data_lock:
                bridge_context.latest_market_data = market_data
            
            if bridge_context.on_market_data:
                try:
                    bridge_context.on_market_data(market_data)
                except Exception as e:
                    self.logger.error(f"Error in market data callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error handling live data: {e}")


class RealtimeTickHandler(MessageHandler):
    """Handler for real-time tick data messages."""
    
    def get_message_type(self) -> str:
        return 'realtime_tick'
    
    def handle(self, message: Dict[str, Any], bridge_context: 'NinjaTradeBridge') -> None:
        """Handle real-time tick data message."""
        try:
            tick_data = bridge_context._parse_realtime_tick(message)
            
            # Skip processing if validation failed
            if tick_data is None:
                self.logger.debug("Realtime tick validation failed - skipping")
                return
            
            with self.data_lock:
                # Update latest market data with tick information
                if bridge_context.latest_market_data:
                    bridge_context.latest_market_data.current_tick_price = tick_data.get('current_tick_price', 0.0)
                    bridge_context.latest_market_data.tick_timestamp = tick_data.get('tick_timestamp', 0)
                    bridge_context.latest_market_data.tick_volume = tick_data.get('tick_volume', 0.0)
            
            # Call real-time tick callback for immediate strategy decisions
            if bridge_context.on_realtime_tick:
                try:
                    bridge_context.on_realtime_tick(tick_data)
                except Exception as e:
                    self.logger.error(f"Error in realtime tick callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error handling realtime tick: {e}")


class TradeCompletionHandler(MessageHandler):
    """Handler for trade completion messages."""
    
    def get_message_type(self) -> str:
        return 'trade_completion'
    
    def handle(self, message: Dict[str, Any], bridge_context: 'NinjaTradeBridge') -> None:
        """Handle trade completion message."""
        try:
            trade_completion = bridge_context._parse_trade_completion(message)
            
            if bridge_context.on_trade_completion:
                try:
                    bridge_context.on_trade_completion(trade_completion)
                except Exception as e:
                    self.logger.error(f"Error in trade completion callback: {e}")
            
            self.logger.info(f"Trade: PnL ${trade_completion.pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error handling trade completion: {e}")


class UnknownMessageHandler(MessageHandler):
    """Handler for unknown message types (fallback)."""
    
    def get_message_type(self) -> str:
        return 'unknown'
    
    def handle(self, message: Dict[str, Any], bridge_context: 'NinjaTradeBridge') -> None:
        """Handle unknown message types."""
        msg_type = message.get('type', 'undefined')
        self.logger.warning(f"Unknown message type: {msg_type}")
        
        # Optionally log the full message for debugging
        self.logger.debug(f"Unknown message content: {message}")


class MessageHandlerFactory:
    """Factory for creating and managing message handlers."""
    
    def __init__(self, logger: TradingLogger, data_lock: Lock):
        self.logger = logger if isinstance(logger, TradingLogger) else get_logger(__name__)
        self.logger.set_context(component="message_handler_factory")
        self.data_lock = data_lock
        self._handlers: Dict[str, MessageHandler] = {}
        self._unknown_handler = UnknownMessageHandler(self.logger, data_lock)
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register all default message handlers."""
        default_handlers = [
            HistoricalDataHandler(self.logger, self.data_lock),
            LiveDataHandler(self.logger, self.data_lock),
            RealtimeTickHandler(self.logger, self.data_lock),
            TradeCompletionHandler(self.logger, self.data_lock),
        ]
        
        for handler in default_handlers:
            self.register_handler(handler)
    
    def register_handler(self, handler: MessageHandler):
        """Register a message handler for a specific message type."""
        message_type = handler.get_message_type()
        self._handlers[message_type] = handler
        self.logger.debug(f"Registered handler for message type: {message_type}")
    
    def unregister_handler(self, message_type: str):
        """Unregister a message handler."""
        if message_type in self._handlers:
            del self._handlers[message_type]
            self.logger.debug(f"Unregistered handler for message type: {message_type}")
    
    def handle_message(self, message: Dict[str, Any], bridge_context: 'NinjaTradeBridge'):
        """Route message to appropriate handler using factory pattern."""
        if not isinstance(message, dict):
            self.logger.error(f"Invalid message format: expected dict, got {type(message)}")
            return
        
        msg_type = message.get('type', '')
        
        # Get handler for message type (fallback to unknown handler)
        handler = self._handlers.get(msg_type, self._unknown_handler)
        
        # Process message with appropriate handler
        try:
            handler.handle(message, bridge_context)
        except Exception as e:
            self.logger.error(f"Error in message handler for type '{msg_type}': {e}")
    
    def get_supported_message_types(self) -> list:
        """Get list of supported message types."""
        return list(self._handlers.keys())
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get statistics about registered handlers."""
        return {
            'total_handlers': len(self._handlers),
            'supported_types': list(self._handlers.keys()),
            'has_unknown_fallback': self._unknown_handler is not None
        }
    
    def create_custom_handler(self, message_type: str, 
                             handler_func: Callable[[Dict[str, Any], 'NinjaTradeBridge'], None]) -> MessageHandler:
        """Create a custom handler from a function."""
        
        class CustomMessageHandler(MessageHandler):
            def __init__(self, msg_type: str, func: Callable, logger: TradingLogger, data_lock: Lock):
                super().__init__(logger, data_lock)
                self.msg_type = msg_type
                self.handler_func = func
            
            def get_message_type(self) -> str:
                return self.msg_type
            
            def handle(self, message: Dict[str, Any], bridge_context: 'NinjaTradeBridge') -> None:
                try:
                    self.handler_func(message, bridge_context)
                except Exception as e:
                    self.logger.error(f"Error in custom handler for {self.msg_type}: {e}")
        
        return CustomMessageHandler(message_type, handler_func, self.logger, self.data_lock)