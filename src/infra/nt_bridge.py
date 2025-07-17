"""
NinjaTrader TCP Bridge
Handles communication with NinjaScript strategy over TCP sockets.
"""
import socket
import threading
import json
import struct
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging


@dataclass
class MarketData:
    """Market data structure for multi-timeframe bars."""
    price_1m: list[float]
    price_5m: list[float]
    price_15m: list[float]
    price_1h: list[float]
    volume_1m: list[float]
    volume_5m: list[float]
    volume_15m: list[float]
    volume_1h: list[float]
    account_balance: float
    buying_power: float
    daily_pnl: float
    unrealized_pnl: float
    open_positions: int
    current_price: float
    volatility: float
    timestamp: int


@dataclass
class TradeSignal:
    """Trade signal structure for sending to NinjaScript."""
    action: int  # 1=buy, 2=sell, 0=close_all
    position_size: int
    confidence: float
    use_stop: bool = False
    stop_price: float = 0.0
    use_target: bool = False
    target_price: float = 0.0


@dataclass
class TradeCompletion:
    """Trade completion data from NinjaScript."""
    pnl: float
    entry_price: float
    exit_price: float
    size: int
    exit_reason: str
    entry_time: int
    exit_time: int
    trade_duration_minutes: float


class NinjaTradeBridge:
    """TCP bridge for communication with NinjaScript strategy."""
    
    def __init__(self, data_port: int = 5556, signal_port: int = 5557):
        self.data_port = data_port
        self.signal_port = signal_port
        self.data_socket: Optional[socket.socket] = None
        self.signal_socket: Optional[socket.socket] = None
        self.is_running = False
        self.historical_data: Dict[str, Any] = {}
        self.latest_market_data: Optional[MarketData] = None
        
        # Callback functions
        self.on_historical_data: Optional[Callable] = None
        self.on_market_data: Optional[Callable] = None
        self.on_trade_completion: Optional[Callable] = None
        
        # Threading
        self.data_thread: Optional[threading.Thread] = None
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the TCP bridge listeners."""
        self.logger.info("Starting NinjaTrader TCP Bridge...")
        self.is_running = True
        
        # Start data listener
        self.data_thread = threading.Thread(target=self._listen_data, daemon=True)
        self.data_thread.start()
        
        # Start signal listener  
        self.signal_thread = threading.Thread(target=self._listen_signals, daemon=True)
        self.signal_thread.start()
        
        self.logger.info(f"Bridge started - Data: {self.data_port}, Signals: {self.signal_port}")
    
    def stop(self):
        """Stop the TCP bridge."""
        self.logger.info("Stopping NinjaTrader TCP Bridge...")
        self.is_running = False
        
        if self.data_socket:
            self.data_socket.close()
        if self.signal_socket:
            self.signal_socket.close()
    
    def _listen_signals(self):
        """Listen for signal connections from NinjaScript."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', self.signal_port))
        server_socket.listen(1)
        
        self.logger.info(f"Listening for signals on port {self.signal_port}")
        
        while self.is_running:
            try:
                client_socket, addr = server_socket.accept()
                self.logger.info(f"NinjaScript signal connection from {addr}")
                self.signal_socket = client_socket
                
                # Keep the connection alive
                while self.is_running and self.signal_socket:
                    try:
                        # Just keep the connection open for sending signals
                        time.sleep(1)
                    except Exception as e:
                        self.logger.error(f"Signal connection error: {e}")
                        break
                
                client_socket.close()
                self.signal_socket = None
                
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Signal listener error: {e}")
                    time.sleep(1)
        
        server_socket.close()
    
    def _listen_data(self):
        """Listen for data from NinjaScript."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', self.data_port))
        server_socket.listen(1)
        
        self.logger.info(f"Listening for data on port {self.data_port}")
        
        while self.is_running:
            try:
                client_socket, addr = server_socket.accept()
                self.logger.info(f"NinjaScript connected from {addr}")
                self.data_socket = client_socket
                
                while self.is_running:
                    try:
                        # Read message length header (4 bytes)
                        header_data = self._recv_all(client_socket, 4)
                        if not header_data:
                            break
                        
                        message_length = struct.unpack('<I', header_data)[0]
                        
                        # Read message data
                        message_data = self._recv_all(client_socket, message_length)
                        if not message_data:
                            break
                        
                        # Parse JSON message
                        message = json.loads(message_data.decode('utf-8'))
                        self._handle_message(message)
                        
                    except Exception as e:
                        self.logger.error(f"Error receiving data: {e}")
                        break
                
                client_socket.close()
                self.data_socket = None
                
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Data listener error: {e}")
                    time.sleep(1)
        
        server_socket.close()
    
    def _recv_all(self, sock: socket.socket, length: int) -> bytes:
        """Receive exactly length bytes from socket."""
        data = b''
        while len(data) < length:
            packet = sock.recv(length - len(data))
            if not packet:
                return b''
            data += packet
        return data
    
    def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming message from NinjaScript."""
        msg_type = message.get('type', '')
        
        if msg_type == 'historical_data':
            self.historical_data = message
            if self.on_historical_data:
                self.on_historical_data(message)
            self.logger.info("Historical data received")
        
        elif msg_type == 'live_data':
            self.latest_market_data = self._parse_market_data(message)
            if self.on_market_data:
                self.on_market_data(self.latest_market_data)
        
        elif msg_type == 'trade_completion':
            trade_completion = self._parse_trade_completion(message)
            if self.on_trade_completion:
                self.on_trade_completion(trade_completion)
            self.logger.info(f"Trade completed: PnL ${trade_completion.pnl:.2f}")
        
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")
    
    def _parse_market_data(self, message: Dict[str, Any]) -> MarketData:
        """Parse market data message into MarketData object."""
        return MarketData(
            price_1m=message.get('price_1m', []),
            price_5m=message.get('price_5m', []),
            price_15m=message.get('price_15m', []),
            price_1h=message.get('price_1h', []),
            volume_1m=message.get('volume_1m', []),
            volume_5m=message.get('volume_5m', []),
            volume_15m=message.get('volume_15m', []),
            volume_1h=message.get('volume_1h', []),
            account_balance=message.get('account_balance', 0.0),
            buying_power=message.get('buying_power', 0.0),
            daily_pnl=message.get('daily_pnl', 0.0),
            unrealized_pnl=message.get('unrealized_pnl', 0.0),
            open_positions=message.get('open_positions', 0),
            current_price=message.get('current_price', 0.0),
            volatility=message.get('volatility', 0.0),
            timestamp=message.get('timestamp', 0)
        )
    
    def _parse_trade_completion(self, message: Dict[str, Any]) -> TradeCompletion:
        """Parse trade completion message into TradeCompletion object."""
        return TradeCompletion(
            pnl=message.get('pnl', 0.0),
            entry_price=message.get('entry_price', 0.0),
            exit_price=message.get('exit_price', 0.0),
            size=message.get('size', 0),
            exit_reason=message.get('exit_reason', ''),
            entry_time=message.get('entry_time', 0),
            exit_time=message.get('exit_time', 0),
            trade_duration_minutes=message.get('trade_duration_minutes', 0.0)
        )
    
    def send_signal(self, signal: TradeSignal) -> bool:
        """Send trade signal to NinjaScript."""
        if not self.signal_socket:
            self.logger.error("Signal socket not connected - NinjaScript must connect first")
            return False
        
        try:
            # Create signal message
            message = {
                'action': signal.action,
                'position_size': signal.position_size,
                'confidence': signal.confidence,
                'use_stop': signal.use_stop,
                'stop_price': signal.stop_price,
                'use_target': signal.use_target,
                'target_price': signal.target_price
            }
            
            # Serialize to JSON
            json_data = json.dumps(message).encode('utf-8')
            
            # Send length header + data
            header = struct.pack('<I', len(json_data))
            self.signal_socket.send(header + json_data)
            
            action_str = "BUY" if signal.action == 1 else "SELL" if signal.action == 2 else "CLOSE_ALL"
            self.logger.info(f"Signal sent: {action_str} {signal.position_size} @ {signal.confidence:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send signal: {e}")
            return False
    
    def get_latest_data(self) -> Optional[MarketData]:
        """Get the most recent market data."""
        return self.latest_market_data
    
    def get_historical_data(self) -> Dict[str, Any]:
        """Get historical data received from NinjaScript."""
        return self.historical_data
    
    def is_connected(self) -> bool:
        """Check if bridge is connected to NinjaScript."""
        return self.data_socket is not None
    
    def is_fully_connected(self) -> bool:
        """Check if both data and signal sockets are connected."""
        return self.data_socket is not None and self.signal_socket is not None