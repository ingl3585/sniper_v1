"""
Main Trading System Orchestrator - Refactored
Coordinates all components of the MNQ trading system with clean separation of concerns.
"""
import sys
import time
import signal
import threading
import logging
from typing import Optional

from src.infra.nt_bridge import NinjaTradeBridge, MarketData, TradeCompletion
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.models.meta_allocator import MetaAllocator
from src.models.ppo_execution import PPOExecutionAgent
from src.utils.data_manager import DataManager
from src.utils.price_history_manager import PriceHistoryManager
from src.config import SystemConfig

# Import new orchestration components
from src.orchestration.signal_processor import SignalProcessor
from src.orchestration.risk_manager import RiskManager
from src.orchestration.connection_manager import ConnectionManager


class TradingSystem:
    """Lightweight trading system orchestrator with separated concerns."""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig.default()
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.trading.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self._initialize_components()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_components(self):
        """Initialize all system components."""
        # Core infrastructure
        self.bridge = NinjaTradeBridge(self.config.trading.data_port, self.config.trading.signal_port)
        self.data_manager = DataManager(self.config.trading.data_dir)
        self.price_history_manager = PriceHistoryManager()
        
        # Trading strategies
        self.mean_reversion = MeanReversionStrategy(self.config.mean_reversion, self.price_history_manager)
        self.momentum = MomentumStrategy(self.config.momentum, self.price_history_manager)
        
        # ML components
        self.meta_allocator = MetaAllocator(self.config.meta_allocator.model_path, self.config.meta_allocator) if self.config.trading.enable_ml_allocator else None
        self.execution_agent = PPOExecutionAgent(self.config.ppo_execution.model_path) if self.config.trading.enable_rl_execution else None
        
        # Orchestration components
        self.signal_processor = SignalProcessor(self.mean_reversion, self.momentum, self.meta_allocator, self.execution_agent)
        self.risk_manager = RiskManager(self.config)
        self.connection_manager = ConnectionManager(self.bridge, self.data_manager, self.price_history_manager)
        
        # Setup bridge callbacks
        self.bridge.on_historical_data = self._on_historical_data
        self.bridge.on_market_data = self._on_market_data
        self.bridge.on_trade_completion = self._on_trade_completion
    
    def start(self):
        """Start the trading system."""
        self.logger.info("Starting MNQ Trading System...")
        
        try:
            # Start bridge and wait for connection
            self.connection_manager.start_bridge()
            self.connection_manager.wait_for_connection(self.shutdown_event)
            
            self.is_running = True
            self.logger.info("Trading system started successfully")
            
            # Main event loop
            self._run_main_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting trading system: {e}")
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the trading system."""
        self.logger.info("Shutting down trading system...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop bridge
        self.connection_manager.stop_bridge()
        
        self.logger.info("Trading system shutdown complete")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    def _run_main_loop(self):
        """Main event loop - simplified."""
        self.logger.info("Starting main event loop...")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Simple health check
                if not self.bridge.is_connected():
                    self.logger.warning("Bridge disconnected")
                    time.sleep(5)
                    continue
                
                # Brief sleep to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(1)
    
    def _on_historical_data(self, historical_data):
        """Handle historical data from NinjaTrader."""
        self.connection_manager.initialize_strategies(historical_data)
    
    def _on_market_data(self, market_data: MarketData):
        """Handle live market data from NinjaTrader."""
        self.logger.info(f"Market Data: Price=${market_data.current_price:.2f}, "
                        f"Balance=${market_data.account_balance:.2f}, "
                        f"Positions={market_data.open_positions}, "
                        f"Daily PnL=${market_data.daily_pnl:.2f}, "
                        f"Volatility={market_data.volatility:.4f}")
        
        # Check if we should trade
        if not self.risk_manager.should_trade(market_data):
            self.logger.debug("Risk manager rejected trading")
            return
        
        # Check for emergency close
        if self.risk_manager.needs_emergency_close(market_data):
            emergency_signal = self.risk_manager.create_emergency_signal()
            self.bridge.send_signal(emergency_signal)
            self.logger.critical("EMERGENCY: CLOSE_ALL signal sent")
            return
        
        # Process signals
        self._process_trading_signals(market_data)
    
    def _on_trade_completion(self, trade_completion: TradeCompletion):
        """Handle trade completion from NinjaTrader."""
        self.logger.info(f"Trade completed: PnL ${trade_completion.pnl:.2f}")
        
        # Store trade data (basic data management)
        self.data_manager.store_trade_completion(trade_completion)
    
    def _process_trading_signals(self, market_data: MarketData):
        """Process trading signals - delegated to signal processor."""
        try:
            trade_signal = self.signal_processor.process_market_data(market_data)
            
            if trade_signal:
                # Get execution decision
                execution_decision = self.signal_processor.get_execution_decision(trade_signal, market_data)
                
                if execution_decision:
                    self.logger.info(f"Execution Decision: Order Type={execution_decision.order_type}, "
                                   f"Urgency={execution_decision.urgency:.2f}")
                else:
                    self.logger.info("Execution Decision: Using default market order")
                
                # Send signal
                success = self.bridge.send_signal(trade_signal)
                if success:
                    action_str = "BUY" if trade_signal.action == 1 else "SELL" if trade_signal.action == 2 else "CLOSE_ALL"
                    self.logger.info(f"SIGNAL SENT: {action_str} {trade_signal.position_size} contracts @ confidence {trade_signal.confidence:.3f}")
                    if trade_signal.use_stop:
                        self.logger.info(f"Stop Loss: ${trade_signal.stop_price:.2f}")
                    if trade_signal.use_target:
                        self.logger.info(f"Target Price: ${trade_signal.target_price:.2f}")
                else:
                    self.logger.error("FAILED to send signal to NinjaTrader")
            else:
                self.logger.debug("No trading signal generated")
                    
        except Exception as e:
            self.logger.error(f"Error processing signals: {e}")
    


def main():
    """Main entry point."""
    config = SystemConfig.from_env()
    
    # Override config from command line args
    if len(sys.argv) > 1:
        if '--no-ml' in sys.argv:
            config.trading.enable_ml_allocator = False
        if '--no-rl' in sys.argv:
            config.trading.enable_rl_execution = False
    
    system = TradingSystem(config)
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("Received keyboard interrupt, shutting down...")
        system.shutdown()
    except Exception as e:
        print(f"Fatal error: {e}")
        system.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()