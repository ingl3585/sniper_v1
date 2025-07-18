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
from queue import Queue, Empty
import numpy as np


@dataclass
class MarketData:
    """Market data structure for multi-timeframe bars."""
    price_1m: list[float]
    price_5m: list[float]
    price_15m: list[float]
    price_30m: list[float]
    price_1h: list[float]
    volume_1m: list[float]
    volume_5m: list[float]
    volume_15m: list[float]
    volume_30m: list[float]
    volume_1h: list[float]
    account_balance: float
    buying_power: float
    daily_pnl: float
    unrealized_pnl: float
    open_positions: int
    current_price: float
    timestamp: int
    
    # Volatility fields
    volatility_1m: float = 0.0
    volatility_5m: float = 0.0
    volatility_15m: float = 0.0
    volatility_30m: float = 0.0
    volatility_1h: float = 0.0
    volatility_regime: str = 'medium'  # 'low', 'medium', 'high'
    volatility_percentile: float = 0.5
    volatility_breakout: Optional[dict] = None  # Breakout detection results
    
    @property
    def data_age_seconds(self) -> float:
        """Calculate age of market data in seconds."""
        current_time = int(time.time() * 1000)  # Current time in milliseconds
        return (current_time - self.timestamp) / 1000.0
    
    def get_data_freshness_warning(self) -> str:
        """Get warning message if data is stale."""
        age = self.data_age_seconds
        if age > 300:  # 5 minutes
            return f"⚠️  STALE DATA: {age:.0f}s old"
        elif age > 120:  # 2 minutes
            return f"⚠️  Aging data: {age:.0f}s old"
        elif age > 60:  # 1 minute
            return f"Data age: {age:.0f}s"
        else:
            return f"Fresh data: {age:.0f}s"
    
    @property
    def volatility(self) -> float:
        """Calculate volatility from 1m price data (computed on-demand)."""
        if len(self.price_1m) < 20:
            return 0.02  # Default volatility
        
        # Calculate log returns
        returns = []
        for i in range(1, len(self.price_1m)):
            if self.price_1m[i-1] > 0:
                returns.append(np.log(self.price_1m[i] / self.price_1m[i-1]))
        
        if len(returns) < 2:
            return 0.02
        
        # Use the last 20 returns for rolling volatility
        recent_returns = returns[-20:]
        return np.std(recent_returns) * np.sqrt(1440)  # Annualized


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
    
    def __init__(self, config=None):
        from src.config import SystemConfig
        
        if config is None:
            config = SystemConfig.default()
        
        # Network configuration
        self.network_config = config.network
        self.data_port = self.network_config.data_port
        self.signal_port = self.network_config.signal_port
        
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
        self.signal_thread: Optional[threading.Thread] = None
        
        # Thread safety
        self._data_lock = threading.RLock()
        self._signal_lock = threading.RLock()
        
        # Error recovery from config
        self.max_reconnect_attempts = self.network_config.max_reconnect_attempts
        self.reconnect_delay = self.network_config.reconnect_delay
        self.connection_timeout = self.network_config.connection_timeout
        
        # Signal queue for thread-safe signal sending
        self.signal_queue = Queue()
        
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
        
        # Start signal queue processor
        self.signal_processor_thread = threading.Thread(target=self._process_signal_queue, daemon=True)
        self.signal_processor_thread.start()
        
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
        """Listen for data from NinjaScript with reconnection logic."""
        reconnect_attempts = 0
        
        while self.is_running and reconnect_attempts < self.max_reconnect_attempts:
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.settimeout(self.connection_timeout)
                server_socket.bind(('localhost', self.data_port))
                server_socket.listen(1)
                
                self.logger.info(f"Listening for data on port {self.data_port}")
                reconnect_attempts = 0  # Reset on successful bind
                
                while self.is_running:
                    try:
                        client_socket, addr = server_socket.accept()
                        self.logger.info(f"NinjaScript connected from {addr}")
                        
                        with self._data_lock:
                            self.data_socket = client_socket
                        
                        # Handle client connection
                        self._handle_data_connection(client_socket)
                        
                    except socket.timeout:
                        continue  # Keep listening for connections
                    except Exception as e:
                        if self.is_running:
                            self.logger.error(f"Error accepting data connection: {e}")
                            time.sleep(1)
                        break
                
                server_socket.close()
                
            except Exception as e:
                if self.is_running:
                    reconnect_attempts += 1
                    self.logger.error(f"Data listener error (attempt {reconnect_attempts}): {e}")
                    if reconnect_attempts < self.max_reconnect_attempts:
                        self.logger.info(f"Retrying in {self.reconnect_delay} seconds...")
                        time.sleep(self.reconnect_delay)
                    else:
                        self.logger.error("Max reconnection attempts reached for data listener")
                        break
    
    def _handle_data_connection(self, client_socket: socket.socket):
        """Handle individual data connection."""
        try:
            while self.is_running:
                try:
                    # Read message length header (4 bytes)
                    header_data = self._recv_all(client_socket, 4)
                    if not header_data:
                        break
                    
                    message_length = struct.unpack('<I', header_data)[0]
                    
                    # Validate message length
                    if message_length > 10 * 1024 * 1024:  # 10MB limit
                        self.logger.error(f"Message too large: {message_length} bytes")
                        break
                    
                    # Read message data
                    message_data = self._recv_all(client_socket, message_length)
                    if not message_data:
                        break
                    
                    # Parse JSON message
                    try:
                        message = json.loads(message_data.decode('utf-8'))
                        self._handle_message(message)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON received: {e}")
                        continue
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.error(f"Error receiving data: {e}")
                    break
                    
        finally:
            client_socket.close()
            with self._data_lock:
                self.data_socket = None
            self.logger.info("Data connection closed")
    
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
        """Handle incoming message from NinjaScript (thread-safe)."""
        try:
            msg_type = message.get('type', '')
            
            if msg_type == 'historical_data':
                with self._data_lock:
                    self.historical_data = message
                if self.on_historical_data:
                    try:
                        self.on_historical_data(message)
                    except Exception as e:
                        self.logger.error(f"Error in historical data callback: {e}")
                self.logger.info("Historical data received")
            
            elif msg_type == 'live_data':
                market_data = self._parse_market_data(message)
                with self._data_lock:
                    self.latest_market_data = market_data
                if self.on_market_data:
                    try:
                        self.on_market_data(market_data)
                    except Exception as e:
                        self.logger.error(f"Error in market data callback: {e}")
            
            elif msg_type == 'trade_completion':
                trade_completion = self._parse_trade_completion(message)
                if self.on_trade_completion:
                    try:
                        self.on_trade_completion(trade_completion)
                    except Exception as e:
                        self.logger.error(f"Error in trade completion callback: {e}")
                self.logger.info(f"Trade completed: PnL ${trade_completion.pnl:.2f}")
            
            else:
                self.logger.warning(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    def _parse_market_data(self, message: Dict[str, Any]) -> MarketData:
        """Parse market data message into MarketData object."""
        # Handle timestamp with proper fallback and logging
        raw_timestamp = message.get('timestamp')
        current_time_ms = int(time.time() * 1000)
        
        if raw_timestamp is not None:
            # Debug: Log what timestamp format we're receiving
            self.logger.info(f"Received timestamp from NT: {raw_timestamp} (type: {type(raw_timestamp)})")
            
            # Handle different timestamp formats that NinjaTrader might send
            if isinstance(raw_timestamp, str):
                try:
                    # Try parsing as ISO format or other string formats
                    from datetime import datetime
                    dt = datetime.fromisoformat(raw_timestamp.replace('Z', '+00:00'))
                    timestamp = int(dt.timestamp() * 1000)
                    self.logger.info(f"Parsed string timestamp: {raw_timestamp} -> {timestamp}")
                except:
                    # If parsing fails, use current time
                    timestamp = current_time_ms
                    self.logger.warning(f"Failed to parse timestamp string: {raw_timestamp}, using current time")
            elif isinstance(raw_timestamp, (int, float)):
                # Check if timestamp is in seconds (need to convert to milliseconds)
                if raw_timestamp < 1e10:  # Likely seconds (before year 2286)
                    timestamp = int(raw_timestamp * 1000)
                    self.logger.info(f"Converted seconds timestamp: {raw_timestamp} -> {timestamp}")
                else:
                    timestamp = int(raw_timestamp)
                    self.logger.info(f"Using milliseconds timestamp: {raw_timestamp}")
            else:
                timestamp = current_time_ms
                self.logger.warning(f"Unknown timestamp type: {type(raw_timestamp)}, using current time")
        else:
            timestamp = current_time_ms
            self.logger.debug(f"No timestamp from NinjaTrader, using current time: {timestamp}")
        
        # Validate timestamp is reasonable (within last 24 hours)
        time_diff = abs(current_time_ms - timestamp)
        if time_diff > 86400000:  # 24 hours in milliseconds
            self.logger.warning(f"Timestamp seems invalid (diff: {time_diff/1000:.0f}s), using current time")
            timestamp = current_time_ms
        
        return MarketData(
            price_1m=message.get('price_1m', []),
            price_5m=message.get('price_5m', []),
            price_15m=message.get('price_15m', []),
            price_30m=message.get('price_30m', []),
            price_1h=message.get('price_1h', []),
            volume_1m=message.get('volume_1m', []),
            volume_5m=message.get('volume_5m', []),
            volume_15m=message.get('volume_15m', []),
            volume_30m=message.get('volume_30m', []),
            volume_1h=message.get('volume_1h', []),
            account_balance=message.get('account_balance', 0.0),
            buying_power=message.get('buying_power', 0.0),
            daily_pnl=message.get('daily_pnl', 0.0),
            unrealized_pnl=message.get('unrealized_pnl', 0.0),
            open_positions=message.get('open_positions', 0),
            current_price=message.get('current_price', 0.0),
            timestamp=timestamp
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
        """Send trade signal to NinjaScript (thread-safe)."""
        try:
            # Add signal to queue for processing by signal thread
            self.signal_queue.put(signal, timeout=self.network_config.signal_timeout)
            return True
        except Exception as e:
            self.logger.error(f"Failed to queue signal: {e}")
            return False
    
    def _process_signal_queue(self):
        """Process queued signals (runs in signal thread)."""
        while self.is_running:
            try:
                # Get signal from queue with timeout
                signal = self.signal_queue.get(timeout=self.network_config.signal_timeout)
                
                # Send the signal
                success = self._send_signal_direct(signal)
                
                if not success:
                    # Requeue signal for retry (with limit)
                    if not hasattr(signal, '_retry_count'):
                        signal._retry_count = 0
                    
                    signal._retry_count += 1
                    if signal._retry_count < 3:
                        self.logger.warning(f"Retrying signal send (attempt {signal._retry_count})")
                        self.signal_queue.put(signal)
                    else:
                        self.logger.error("Max signal retry attempts reached, dropping signal")
                
                self.signal_queue.task_done()
                
            except Empty:
                continue  # Timeout waiting for signal
            except Exception as e:
                self.logger.error(f"Error processing signal queue: {e}")
    
    def _send_signal_direct(self, signal: TradeSignal) -> bool:
        """Send signal directly to NinjaScript."""
        with self._signal_lock:
            if not self.signal_socket:
                self.logger.error("Signal socket not connected")
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
        """Get the most recent market data (thread-safe)."""
        with self._data_lock:
            return self.latest_market_data
    
    def get_historical_data(self) -> Dict[str, Any]:
        """Get historical data received from NinjaScript (thread-safe)."""
        with self._data_lock:
            return self.historical_data.copy() if self.historical_data else {}
    
    def is_connected(self) -> bool:
        """Check if bridge is connected to NinjaScript (thread-safe)."""
        with self._data_lock:
            return self.data_socket is not None
    
    def is_fully_connected(self) -> bool:
        """Check if both data and signal sockets are connected (thread-safe)."""
        with self._data_lock, self._signal_lock:
            return self.data_socket is not None and self.signal_socket is not None