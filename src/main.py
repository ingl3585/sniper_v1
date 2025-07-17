"""
Main Trading System Orchestrator
Coordinates all components of the MNQ trading system.
"""
import sys
import time
import signal
import threading
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

from src.infra.nt_bridge import NinjaTradeBridge, MarketData, TradeSignal, TradeCompletion
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.models.meta_allocator import MetaAllocator, AllocationDecision
from src.models.ppo_execution import PPOExecutionAgent, ExecutionDecision
from src.utils.data_manager import DataManager
from src.config import SystemConfig


class TradingSystem:
    """Main trading system orchestrator."""
    
    def __init__(self, config: SystemConfig = None):
        if config is None:
            config = SystemConfig.default()
        
        self.config = config
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.trading.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.bridge = NinjaTradeBridge(config.trading.data_port, config.trading.signal_port)
        self.data_manager = DataManager(config.trading.data_dir)
        
        # Initialize strategies
        self.mean_reversion = MeanReversionStrategy(config.mean_reversion)
        self.momentum = MomentumStrategy(config.momentum)
        
        # Initialize ML components
        self.meta_allocator = MetaAllocator(config.meta_allocator.model_path, config.meta_allocator) if config.trading.enable_ml_allocator else None
        self.execution_agent = PPOExecutionAgent(config.ppo_execution.model_path) if config.trading.enable_rl_execution else None
        
        # Performance tracking
        self.strategy_performance = {
            'mean_reversion': 0.0,
            'momentum': 0.0
        }
        self.total_trades = 0
        self.total_pnl = 0.0
        self.last_market_data: Optional[MarketData] = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup bridge callbacks
        self.bridge.on_historical_data = self._on_historical_data
        self.bridge.on_market_data = self._on_market_data
        self.bridge.on_trade_completion = self._on_trade_completion
    
    def start(self):
        """Start the trading system."""
        self.logger.info("Starting MNQ Trading System...")
        self.logger.info("Note: Make sure NinjaTrader is running with the ResearchStrategy enabled")
        
        try:
            # Start TCP bridge
            self.bridge.start()
            
            # Wait for initial connection
            self._wait_for_connection()
            
            self.is_running = True
            self.logger.info("Trading system started successfully - ready to receive data from NinjaTrader")
            
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
        self.bridge.stop()
        
        # Save final state
        self._save_system_state()
        
        self.logger.info("Trading system shutdown complete")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    def _wait_for_connection(self, timeout: int = None):
        """Wait for NinjaTrader connection indefinitely."""
        self.logger.info("Waiting for NinjaTrader connection (will wait indefinitely)...")
        
        while not self.bridge.is_connected() and not self.shutdown_event.is_set():
            if self.shutdown_event.wait(1):  # Check for shutdown every second
                raise ConnectionError("Shutdown requested while waiting for connection")
            
            # Check if we have any data connection (even without signal connection)
            if self.bridge.data_socket is not None:
                self.logger.info("Data connection established, continuing...")
                break
        
        if self.shutdown_event.is_set():
            raise ConnectionError("Shutdown requested")
            
        self.logger.info("NinjaTrader connection established")
    
    def _run_main_loop(self):
        """Main event loop."""
        self.logger.info("Starting main event loop...")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Check system health
                if not self._check_system_health():
                    self.logger.warning("System health check failed, pausing trading")
                    time.sleep(10)
                    continue
                
                # Process any pending actions
                self._process_pending_actions()
                
                # Sleep briefly to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(1)
    
    def _check_system_health(self) -> bool:
        """Check if system is healthy and ready to trade."""
        # Check bridge connection
        if not self.bridge.is_connected():
            return False
        
        # Check account balance
        if self.last_market_data:
            if self.last_market_data.account_balance < self.config.trading.min_account_balance:
                return False
            
            # Check daily loss limit and send emergency close if needed
            daily_loss_limit = self.last_market_data.account_balance * self.config.trading.max_daily_loss_pct
            if self.last_market_data.daily_pnl < -daily_loss_limit:
                self.logger.warning(f"Daily loss limit reached: {self.last_market_data.daily_pnl:.2f}, sending CLOSE_ALL")
                self._send_emergency_close()
                return False
        
        return True
    
    def _process_pending_actions(self):
        """Process any pending system actions."""
        # This could include model retraining, data cleanup, etc.
        pass
    
    def _on_historical_data(self, historical_data: Dict[str, Any]):
        """Handle historical data from NinjaTrader."""
        self.logger.info("Historical data received, initializing strategies...")
        
        # Store historical data
        self.data_manager.store_historical_data(historical_data)
        
        # Initialize strategies with historical data
        self._initialize_strategies(historical_data)
        
        self.logger.info("Strategies initialized with historical data")
    
    def _on_market_data(self, market_data: MarketData):
        """Handle live market data from NinjaTrader."""
        self.logger.info(f"Live market data received: Price=${market_data.current_price:.2f}, Balance=${market_data.account_balance:.2f}")
        self.last_market_data = market_data
        
        # Store market data
        self.data_manager.store_market_data(market_data)
        
        # Generate and execute trading signals
        self._process_market_data(market_data)
    
    def _on_trade_completion(self, trade_completion: TradeCompletion):
        """Handle trade completion from NinjaTrader."""
        self.logger.info(f"Trade completed: PnL ${trade_completion.pnl:.2f}")
        
        # Update performance tracking
        self.total_trades += 1
        self.total_pnl += trade_completion.pnl
        
        # Store trade data
        self.data_manager.store_trade_completion(trade_completion)
        
        # Update strategy performance (simplified)
        self._update_strategy_performance(trade_completion)
    
    def _initialize_strategies(self, historical_data: Dict[str, Any]):
        """Initialize strategies with historical data."""
        # This could involve warming up indicators, etc.
        pass
    
    def _process_market_data(self, market_data: MarketData):
        """Process market data and generate trading signals."""
        if not self._should_trade(market_data):
            return
        
        # Generate signals from strategies
        mean_reversion_signal = self.mean_reversion.generate_signal(market_data)
        momentum_signal = self.momentum.generate_signal(market_data)
        
        # Get allocation decision from meta-allocator
        allocation = self._get_allocation_decision(market_data)
        
        # Select and execute best signal
        final_signal = self._select_final_signal(
            mean_reversion_signal, momentum_signal, allocation
        )
        
        if final_signal:
            self._execute_signal(final_signal, market_data)
    
    def _should_trade(self, market_data: MarketData) -> bool:
        """Check if conditions are suitable for trading."""
        # Check position limits
        if abs(market_data.open_positions) >= self.config.trading.max_position_size:
            return False
        
        # Check account conditions
        if market_data.account_balance < self.config.trading.min_account_balance:
            return False
        
        # Check daily loss limit and send emergency close if needed
        daily_loss_limit = market_data.account_balance * self.config.trading.max_daily_loss_pct
        if market_data.daily_pnl < -daily_loss_limit:
            self.logger.warning(f"Daily loss limit reached: {market_data.daily_pnl:.2f}, sending CLOSE_ALL")
            self._send_emergency_close()
            return False
        
        return True
    
    def _get_allocation_decision(self, market_data: MarketData) -> Optional[AllocationDecision]:
        """Get allocation decision from meta-allocator."""
        if not self.meta_allocator:
            return None
        
        try:
            mean_reversion_perf = self.strategy_performance['mean_reversion']
            momentum_perf = self.strategy_performance['momentum']
            
            return self.meta_allocator.get_allocation(
                market_data, mean_reversion_perf, momentum_perf
            )
        except Exception as e:
            self.logger.error(f"Error getting allocation decision: {e}")
            return None
    
    def _select_final_signal(self, mean_reversion_signal, momentum_signal, allocation):
        """Select final signal based on allocation and signal strength."""
        if not mean_reversion_signal and not momentum_signal:
            return None
        
        # Default equal weighting if no allocation
        if not allocation:
            if mean_reversion_signal and momentum_signal:
                # Choose signal with higher confidence
                if mean_reversion_signal.confidence > momentum_signal.confidence:
                    return mean_reversion_signal
                else:
                    return momentum_signal
            else:
                return mean_reversion_signal or momentum_signal
        
        # Use allocation weights to select signal
        signals = []
        if mean_reversion_signal:
            signals.append(('mean_reversion', mean_reversion_signal, allocation.mean_reversion_weight))
        if momentum_signal:
            signals.append(('momentum', momentum_signal, allocation.momentum_weight))
        
        if not signals:
            return None
        
        # Select signal with highest weighted score
        best_signal = max(signals, key=lambda x: x[1].confidence * x[2])
        
        self.logger.info(f"Selected {best_signal[0]} signal with confidence {best_signal[1].confidence:.2f}")
        
        return best_signal[1]
    
    def _execute_signal(self, signal, market_data: MarketData):
        """Execute trading signal."""
        try:
            # Convert strategy signal to trade signal
            if hasattr(signal, 'action'):
                # It's already a strategy signal
                trade_signal = self.mean_reversion.create_trade_signal(signal, market_data)
            else:
                # It's a raw signal
                trade_signal = signal
            
            # Get execution decision from RL agent
            execution_decision = self._get_execution_decision(trade_signal, market_data)
            
            # Apply execution decision (this would modify the order in a full implementation)
            # For now, we'll just log it
            if execution_decision:
                self.logger.info(f"Execution decision: {execution_decision.order_type} order")
            
            # Send signal to NinjaTrader
            success = self.bridge.send_signal(trade_signal)
            
            if success:
                action_str = "BUY" if trade_signal.action == 1 else "SELL"
                self.logger.info(f"Signal sent: {action_str} {trade_signal.position_size} contracts")
            else:
                self.logger.error("Failed to send signal to NinjaTrader")
                
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
    
    def _get_execution_decision(self, trade_signal: TradeSignal, 
                              market_data: MarketData) -> Optional[ExecutionDecision]:
        """Get execution decision from RL agent."""
        if not self.execution_agent:
            return None
        
        try:
            # Calculate urgency based on signal confidence and market conditions
            urgency = min(1.0, trade_signal.confidence + market_data.volatility)
            
            return self.execution_agent.get_execution_decision(
                market_data, trade_signal.position_size, urgency
            )
        except Exception as e:
            self.logger.error(f"Error getting execution decision: {e}")
            return None
    
    def _update_strategy_performance(self, trade_completion: TradeCompletion):
        """Update strategy performance tracking."""
        # Simplified performance update
        # In a full implementation, this would track which strategy generated the trade
        performance_update = trade_completion.pnl / 1000.0  # Normalize
        
        # Update both strategies (simplified)
        self.strategy_performance['mean_reversion'] = (
            self.strategy_performance['mean_reversion'] * 0.9 + performance_update * 0.1
        )
        self.strategy_performance['momentum'] = (
            self.strategy_performance['momentum'] * 0.9 + performance_update * 0.1
        )
    
    def _send_emergency_close(self):
        """Send emergency CLOSE_ALL signal to NinjaScript."""
        try:
            from src.infra.nt_bridge import TradeSignal
            emergency_signal = TradeSignal(
                action=0,  # CLOSE_ALL
                position_size=0,
                confidence=1.0
            )
            self.bridge.send_signal(emergency_signal)
            self.logger.critical("EMERGENCY: CLOSE_ALL signal sent to NinjaScript")
        except Exception as e:
            self.logger.error(f"Failed to send emergency close signal: {e}")
    
    def _save_system_state(self):
        """Save system state for recovery."""
        state = {
            'total_trades': self.total_trades,
            'total_pnl': self.total_pnl,
            'strategy_performance': self.strategy_performance,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            self.data_manager.save_system_state(state)
            self.logger.info("System state saved")
        except Exception as e:
            self.logger.error(f"Error saving system state: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'is_running': self.is_running,
            'bridge_connected': self.bridge.is_connected(),
            'total_trades': self.total_trades,
            'total_pnl': self.total_pnl,
            'strategy_performance': self.strategy_performance,
            'last_market_data_time': self.last_market_data.timestamp if self.last_market_data else None,
            'meta_allocator_info': self.meta_allocator.get_model_info() if self.meta_allocator else None,
            'execution_agent_info': self.execution_agent.get_agent_info() if self.execution_agent else None
        }


def main():
    """Main entry point."""
    config = SystemConfig.from_env()
    
    # Override config from command line args if needed
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