"""
Signal Processing Engine
Handles strategy signal generation, allocation, and execution decisions.
"""
from typing import Optional, Tuple
import logging
from src.infra.nt_bridge import MarketData, TradeSignal
from src.strategies.base_strategy import Signal
from src.models.meta_allocator import MetaAllocator, AllocationDecision
from src.models.ppo_execution import PPOExecutionAgent, ExecutionDecision


class SignalProcessor:
    """Processes signals from strategies and converts them to executable trades."""
    
    def __init__(self, mean_reversion, momentum, meta_allocator=None, execution_agent=None):
        self.mean_reversion = mean_reversion
        self.momentum = momentum
        self.meta_allocator = meta_allocator
        self.execution_agent = execution_agent
        self.logger = logging.getLogger(__name__)
    
    def process_market_data(self, market_data: MarketData) -> Optional[TradeSignal]:
        """Process market data and generate trading signals."""
        self.logger.info("=== Signal Processing Start ===")
        
        # Generate signals from strategies
        self.logger.info("Generating mean reversion signal...")
        mean_reversion_signal = self.mean_reversion.generate_signal(market_data)
        if mean_reversion_signal:
            stop_str = f"${mean_reversion_signal.stop_price:.2f}" if mean_reversion_signal.stop_price else 'None'
            target_str = f"${mean_reversion_signal.target_price:.2f}" if mean_reversion_signal.target_price else 'None'
            self.logger.info(f"Mean Reversion Signal: Action={mean_reversion_signal.action}, "
                           f"Confidence={mean_reversion_signal.confidence:.3f}, "
                           f"Entry=${mean_reversion_signal.entry_price:.2f}, "
                           f"Stop={stop_str}, Target={target_str}")
        else:
            self.logger.info("Mean Reversion Signal: None")
        
        self.logger.info("Generating momentum signal...")
        momentum_signal = self.momentum.generate_signal(market_data)
        if momentum_signal:
            stop_str = f"${momentum_signal.stop_price:.2f}" if momentum_signal.stop_price else 'None'
            target_str = f"${momentum_signal.target_price:.2f}" if momentum_signal.target_price else 'None'
            self.logger.info(f"Momentum Signal: Action={momentum_signal.action}, "
                           f"Confidence={momentum_signal.confidence:.3f}, "
                           f"Entry=${momentum_signal.entry_price:.2f}, "
                           f"Stop={stop_str}, Target={target_str}")
        else:
            self.logger.info("Momentum Signal: None")
        
        # Get allocation decision
        allocation = self._get_allocation_decision(market_data)
        if allocation:
            self.logger.info(f"Meta Allocator: MR Weight={allocation.mean_reversion_weight:.2f}, "
                           f"Momentum Weight={allocation.momentum_weight:.2f}")
        else:
            self.logger.info("Meta Allocator: Using equal weights")
        
        # Select best signal
        final_signal = self._select_final_signal(mean_reversion_signal, momentum_signal, allocation)
        
        if final_signal:
            trade_signal = self._convert_to_trade_signal(final_signal, market_data)
            self.logger.info(f"Final Trade Signal: Action={trade_signal.action}, "
                           f"Size={trade_signal.position_size}, "
                           f"Confidence={trade_signal.confidence:.3f}")
            self.logger.info("=== Signal Processing End ===")
            return trade_signal
        else:
            self.logger.info("Final Signal: None - No trading action")
            self.logger.info("=== Signal Processing End ===")
        
        return None
    
    def _get_allocation_decision(self, market_data: MarketData) -> Optional[AllocationDecision]:
        """Get allocation decision from meta-allocator."""
        if not self.meta_allocator:
            return None
        
        try:
            # Use default performance metrics for now
            return self.meta_allocator.get_allocation(market_data, 0.0, 0.0)
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
                return mean_reversion_signal if mean_reversion_signal.confidence > momentum_signal.confidence else momentum_signal
            return mean_reversion_signal or momentum_signal
        
        # Use allocation weights
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
    
    def _convert_to_trade_signal(self, signal: Signal, market_data: MarketData) -> TradeSignal:
        """Convert strategy signal to trade signal."""
        if hasattr(signal, 'confidence') and hasattr(signal, 'action'):
            return self.mean_reversion.create_trade_signal(signal, market_data)
        return signal
    
    def get_execution_decision(self, trade_signal: TradeSignal, market_data: MarketData) -> Optional[ExecutionDecision]:
        """Get execution decision from RL agent."""
        if not self.execution_agent:
            return None
        
        try:
            urgency = min(1.0, trade_signal.confidence + market_data.volatility)
            return self.execution_agent.get_execution_decision(
                market_data, trade_signal.position_size, urgency
            )
        except Exception as e:
            self.logger.error(f"Error getting execution decision: {e}")
            return None
    
