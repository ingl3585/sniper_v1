"""
Trading System Core
Main application logic for the MNQ trading system with clean separation of concerns.
"""
import sys
import time
import signal
import threading
from typing import Optional

from src.infra.nt_bridge import NinjaTradeBridge, MarketData, TradeCompletion
from src.strategies.strategy_factory import StrategyFactory
from src.engine.meta_allocator import MetaAllocator
from storage import DataManager, PriceHistoryManager
from config import SystemConfig
from logging_config import setup_logging, get_logger

# Import orchestration components
from src.engine.signal_processor import SignalProcessor
from src.engine.risk_manager import RiskManager
from src.infra.connection_manager import ConnectionManager


class TradingSystem:
    """Main trading system orchestrator with separated concerns."""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig.default()
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Setup advanced logging system
        self.logging_config = setup_logging({
            'log_level': self.config.trading.log_level,
            'structured_logging': getattr(self.config, 'structured_logging', False),
            'console_logging': True
        })
        self.logger = get_logger(__name__)
        self.logger.set_context(component='trading_system')
        
        # Initialize core components
        self._initialize_components()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_components(self):
        """Initialize all system components."""
        # Core infrastructure
        self.bridge = NinjaTradeBridge(self.config)
        self.data_manager = DataManager(self.config.trading.data_dir)
        self.price_history_manager = PriceHistoryManager(self.config)
        
        # Trading strategies (using factory pattern)
        self.strategy_factory = StrategyFactory(self.config, self.price_history_manager)
        strategies = self.strategy_factory.create_enabled_strategies()
        
        # Extract individual strategies for compatibility
        self.mean_reversion = strategies.get('mean_reversion')
        self.momentum = strategies.get('momentum')
        self.vol_carry = strategies.get('volatility_carry')
        self.vol_breakout = strategies.get('volatility_breakout')
        
        # ML components
        self.meta_allocator = MetaAllocator(
            self.config.meta_allocator.model_path, 
            self.config.meta_allocator
        ) if self.config.trading.enable_ml_allocator else None
        
        # No execution agent for now
        self.execution_agent = None
        
        # Orchestration components
        self.signal_processor = SignalProcessor(
            self.mean_reversion, 
            self.momentum, 
            self.vol_carry, 
            self.vol_breakout,
            self.meta_allocator, 
            self.execution_agent,
            self.config
        )
        self.risk_manager = RiskManager(self.config)
        self.connection_manager = ConnectionManager(
            self.bridge, 
            self.data_manager, 
            self.price_history_manager
        )
        
        # Connect risk manager to connection manager for historical data status
        self.risk_manager.connection_manager = self.connection_manager
        
        # Setup bridge callbacks
        self.bridge.on_historical_data = self._on_historical_data
        self.bridge.on_market_data = self._on_market_data
        self.bridge.on_realtime_tick = self._on_realtime_tick
        self.bridge.on_trade_completion = self._on_trade_completion
        
        # Real-time signal tracking
        self.last_realtime_signal = None
        self.last_realtime_signal_time = 0
        
        # Current market data state
        self.current_market_data = None
    
    def run(self):
        """Start the trading system."""
        self.logger.info("Starting MNQ Trading System...")
        
        try:
            # Start bridge and wait for connection
            self.connection_manager.start_bridge()
            self.connection_manager.wait_for_connection(self.shutdown_event)
            
            if self.shutdown_event.is_set():
                self.logger.info("Shutdown requested during startup")
                return
            
            self.is_running = True
            self.logger.info("Trading system started successfully")
            
            # Main trading loop
            self._run_trading_loop()
            
        except Exception as e:
            self.logger.error(f"Fatal error in trading system: {e}")
            raise
        finally:
            self.shutdown()
    
    def _run_trading_loop(self):
        """Main trading loop."""
        last_status_log = 0
        
        while not self.shutdown_event.is_set():
            try:
                # Log status periodically
                current_time = time.time()
                if current_time - last_status_log > 300:  # Every 5 minutes
                    self._log_system_status()
                    last_status_log = current_time
                
                # Check risk limits
                if self.current_market_data and not self.risk_manager.check_daily_limits(self.current_market_data):
                    self.logger.warning("Daily risk limits exceeded - stopping trading")
                    break
                
                # Brief sleep to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                time.sleep(1)
    
    def _on_historical_data(self, data):
        """Handle historical data from NinjaTrader."""
        try:
            self.connection_manager.process_historical_data(data)
        except Exception as e:
            self.logger.error(f"Error processing historical data: {e}")
    
    def _on_market_data(self, market_data: MarketData):
        """Handle real-time market data."""
        try:
            # Store current market data for risk management
            self.current_market_data = market_data
            
            # Don't generate signals until historical data is loaded
            if not self.connection_manager.is_ready_for_trading():
                return
            
            # Process market data through signal processor
            self.logger.debug("Processing market data (regular)")
            trade_signal = self.signal_processor.process_market_data(market_data)
            
            if trade_signal:
                self.logger.debug(f"Trade signal received for validation: {trade_signal.action}, size={trade_signal.position_size}")
                # Apply risk management
                if self.risk_manager.validate_trade_signal(trade_signal, market_data):
                    self.logger.info(f"Sending trade signal to NinjaTrader: Action={trade_signal.action}, Size={trade_signal.position_size}")
                    self.bridge.send_signal(trade_signal)
                else:
                    self.logger.info("Trade signal blocked by risk manager")
            else:
                self.logger.debug("No trade signal received from signal processor")
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
    
    def _on_realtime_tick(self, tick_data: dict):
        """Handle real-time tick data for immediate execution."""
        try:
            # Create market data from tick for real-time processing
            market_data = self.connection_manager.create_market_data_from_tick(tick_data)
            if market_data:
                self._process_realtime_tick_signals(market_data, tick_data)
        except Exception as e:
            self.logger.error(f"Error processing realtime tick: {e}")
    
    def _process_realtime_tick_signals(self, market_data: MarketData, tick_data: dict):
        """Process signals with real-time tick data for immediate execution."""
        try:
            # Don't generate signals until historical data is loaded
            if not self.connection_manager.is_ready_for_trading():
                return
                
            # Generate signals with real-time flag
            self.logger.debug("Processing market data (realtime tick)")
            trade_signal = self.signal_processor.process_market_data(market_data, is_realtime_tick=True)
            
            if trade_signal and self._should_execute_realtime_signal(trade_signal, tick_data):
                self.logger.debug(f"Realtime trade signal received: {trade_signal.action}, size={trade_signal.position_size}")
                # Apply expedited risk management for real-time signals
                if self.risk_manager.validate_realtime_signal(trade_signal, market_data, tick_data):
                    self.logger.info(f"Sending realtime trade signal to NinjaTrader: Action={trade_signal.action}, Size={trade_signal.position_size}")
                    self.bridge.send_signal(trade_signal)
                else:
                    self.logger.info("Realtime trade signal blocked by risk manager")
            elif trade_signal:
                self.logger.debug("Realtime trade signal created but should not execute immediately")
                
                # Update tracking
                self.last_realtime_signal = trade_signal
                self.last_realtime_signal_time = time.time()
        except Exception as e:
            self.logger.error(f"Error processing real-time tick signals: {e}")
    
    def _should_execute_realtime_signal(self, signal, tick_data: dict) -> bool:
        """Determine if real-time signal should be executed immediately."""
        # Minimum time between real-time signals (prevent spam)
        if self.last_realtime_signal_time and time.time() - self.last_realtime_signal_time < 5:
            return False
        
        # Only execute high-confidence real-time signals
        return signal.confidence > 0.7
    
    def _on_trade_completion(self, completion: TradeCompletion):
        """Handle trade completion notifications."""
        try:
            self.logger.info(f"Trade completed: PnL=${completion.pnl:.2f}")
            
            # Update risk tracking
            self.risk_manager.update_trade_completion(completion)
            
            # Update ML models if enabled
            if self.meta_allocator:
                self.meta_allocator.update_performance_tracking(completion)
            
        except Exception as e:
            self.logger.error(f"Error processing trade completion: {e}")
    
    def _log_system_status(self):
        """Log periodic system status."""
        try:
            status = {
                'strategies_active': len([s for s in [self.mean_reversion, self.momentum, self.vol_carry, self.vol_breakout] if s]),
                'ml_allocator': 'enabled' if self.meta_allocator else 'disabled',
                'rl_execution': 'enabled' if self.execution_agent else 'disabled',
                'bridge_connected': self.bridge.is_running if hasattr(self.bridge, 'is_running') else 'unknown'
            }
            
            self.logger.debug(f"System status check: {status}")
            
        except Exception as e:
            self.logger.error(f"Error logging system status: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
    
    def shutdown(self):
        """Graceful shutdown of the trading system."""
        if not self.is_running:
            return
        
        self.logger.info("Shutting down trading system...")
        self.is_running = False
        
        try:
            # Stop bridge
            if hasattr(self.bridge, 'stop'):
                self.bridge.stop()
            
            # Stop components
            if hasattr(self.connection_manager, 'cleanup'):
                self.connection_manager.cleanup()
            
            self.logger.info("Trading system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")