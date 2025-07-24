"""
Network Manager for TCP Communication with NinjaTrader
Handles socket connections, message parsing, and data exchange.
"""
import socket
import threading
import json
import struct
import time
from typing import Dict, Any, Optional, Callable
from logging_config import get_logger


class NetworkManager:
    """Manages TCP connections with NinjaTrader strategy."""
    
    def __init__(self, data_port: int, signal_port: int, config):
        self.data_port = data_port
        self.signal_port = signal_port
        self.config = config
        self.logger = get_logger(__name__)
        self.logger.set_context(component='network_manager')
        
        # Server sockets
        self.data_server = None
        self.signal_server = None
        
        # Client connections
        self.data_client = None
        self.signal_client = None
        
        # Threading
        self.is_running = False
        self.data_thread = None
        self.signal_thread = None
        
        # Connection status
        self.data_connected = False
        self.signal_connected = False
        
        # Message callback
        self.on_message_received: Optional[Callable] = None
        
    def start(self):
        """Start the TCP servers."""
        self.logger.info(f"Starting TCP servers on ports {self.data_port}, {self.signal_port}")
        
        try:
            # Create server sockets
            self.data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.signal_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Set socket options
            self.data_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.signal_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to localhost
            self.data_server.bind(('localhost', self.data_port))
            self.signal_server.bind(('localhost', self.signal_port))
            
            # Start listening
            self.data_server.listen(1)
            self.signal_server.listen(1)
            
            self.is_running = True
            
            # Start threads to accept connections
            self.data_thread = threading.Thread(target=self._data_server_loop, daemon=True)
            self.signal_thread = threading.Thread(target=self._signal_server_loop, daemon=True)
            
            self.data_thread.start()
            self.signal_thread.start()
            
            self.logger.info("TCP servers started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start TCP servers: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the TCP servers."""
        self.logger.info("Stopping TCP servers...")
        self.is_running = False
        
        # Close client connections
        if self.data_client:
            try:
                self.data_client.close()
            except:
                pass
        if self.signal_client:
            try:
                self.signal_client.close()
            except:
                pass
        
        # Close server sockets
        if self.data_server:
            try:
                self.data_server.close()
            except:
                pass
        if self.signal_server:
            try:
                self.signal_server.close()
            except:
                pass
        
        # Reset connection status
        self.data_connected = False
        self.signal_connected = False
        
        self.logger.info("TCP servers stopped")
    
    def _data_server_loop(self):
        """Accept and handle data connections."""
        self.logger.info(f"Data server listening on port {self.data_port}")
        
        while self.is_running:
            try:
                # Accept connection
                client, addr = self.data_server.accept()
                self.data_client = client
                self.data_connected = True
                
                self.logger.info(f"Data client connected from {addr}")
                
                # Handle messages from this client
                self._handle_data_client(client)
                
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Data server error: {e}")
                    time.sleep(1)
    
    def _signal_server_loop(self):
        """Accept and handle signal connections."""
        self.logger.info(f"Signal server listening on port {self.signal_port}")
        
        while self.is_running:
            try:
                # Accept connection
                client, addr = self.signal_server.accept()
                self.signal_client = client
                self.signal_connected = True
                
                self.logger.info(f"Signal client connected from {addr}")
                
                # Keep connection alive (signals are sent TO client, not FROM client)
                while self.is_running and self.signal_connected:
                    time.sleep(1)
                    
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Signal server error: {e}")
                    time.sleep(1)
    
    def _handle_data_client(self, client):
        """Handle incoming data messages."""
        while self.is_running and self.data_connected:
            try:
                # Read message header (4 bytes = message length)
                header = self._recv_exact(client, 4)
                if not header:
                    break
                
                # Unpack message length
                message_length = struct.unpack('<I', header)[0]
                
                # Read message body
                message_data = self._recv_exact(client, message_length)
                if not message_data:
                    break
                
                # Parse JSON message
                message_json = message_data.decode('utf-8')
                message = json.loads(message_json)
                
                # Process message
                if self.on_message_received:
                    self.on_message_received(message)
                
            except Exception as e:
                self.logger.error(f"Error handling data client: {e}")
                break
        
        # Client disconnected
        self.data_connected = False
        try:
            client.close()
        except:
            pass
        self.logger.info("Data client disconnected")
    
    def _recv_exact(self, client, length):
        """Receive exactly 'length' bytes from client."""
        data = b''
        while len(data) < length:
            chunk = client.recv(length - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def send_signal(self, signal_data: Dict[str, Any]):
        """Send trading signal to NinjaTrader."""
        if not self.signal_connected or not self.signal_client:
            self.logger.warning("Cannot send signal - no signal client connected")
            return False
        
        try:
            # Serialize message
            message_json = json.dumps(signal_data)
            message_data = message_json.encode('utf-8')
            
            # Send message with length header
            header = struct.pack('<I', len(message_data))
            self.signal_client.send(header + message_data)
            
            self.logger.info(f"Signal sent: {signal_data.get('action', 'unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send signal: {e}")
            self.signal_connected = False
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status."""
        return {
            'is_running': self.is_running,
            'data_connected': self.data_connected,
            'signal_connected': self.signal_connected,
            'data_port': self.data_port,
            'signal_port': self.signal_port
        }