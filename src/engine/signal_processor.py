"""
Signal Processing Engine
Handles strategy signal generation, allocation, and execution decisions.
"""
from typing import Optional, Tuple
import time
from src.infra.nt_bridge import MarketData, TradeSignal
from src.strategies.base_strategy import Signal
from src.engine.meta_allocator import MetaAllocator, AllocationDecision
from src.engine.live_gateway import ExecutionEngine, ExecutionDecision
from logging_config import get_logger


class SignalProcessor:
    """Processes signals from strategies and converts them to executable trades."""
    
    def __init__(self, mean_reversion, momentum, vol_carry=None, vol_breakout=None, meta_allocator=None, execution_agent=None, config=None):
        self.mean_reversion = mean_reversion
        self.momentum = momentum
        self.vol_carry = vol_carry
        self.vol_breakout = vol_breakout
        self.meta_allocator = meta_allocator
        self.config = config
        self.logger = get_logger(__name__)
        self.logger.set_context(component='signal_processor')
        self.last_no_signal_log = 0  # Rate limit "no signal" messages
        
        # Initialize execution engine
        self.execution_engine = ExecutionEngine(config, execution_agent)
    
    def process_market_data(self, market_data: MarketData, is_realtime_tick: bool = False) -> Optional[TradeSignal]:
        """Process market data and generate trading signals."""
        # Processing signals (real-time mode suppresses verbose logging)
        if not is_realtime_tick:
            self.logger.debug("Processing market data signals")
        
        # Generate signals from all strategies
        signals = {}
        
        # Mean Reversion Strategy
        # Mean reversion
        mean_reversion_signal = self.mean_reversion.generate_signal(market_data)
        if mean_reversion_signal:
            signals['mean_reversion'] = mean_reversion_signal
        
        # Momentum
        momentum_signal = self.momentum.generate_signal(market_data)
        if momentum_signal:
            signals['momentum'] = momentum_signal
        
        # Volatility Carry Strategy (if available)
        # Vol carry
        if self.vol_carry:
            vol_carry_signal = self.vol_carry.generate_signal(market_data)
            if vol_carry_signal:
                signals['vol_carry'] = vol_carry_signal
        
        # Volatility Breakout Strategy (if available)
        # Vol breakout  
        if self.vol_breakout:
            vol_breakout_signal = self.vol_breakout.generate_signal(market_data)
            if vol_breakout_signal:
                signals['vol_breakout'] = vol_breakout_signal
        
        # Get allocation decision
        allocation = self._get_allocation_decision(market_data, signals)
        # Meta allocation computed
        if not is_realtime_tick:
            self.logger.debug("Meta allocation completed")
        
        # Select best signal
        final_signal = self._select_final_signal(signals, allocation, is_realtime_tick)
        
        if final_signal:
            trade_signal = self.execution_engine.execute_signal(final_signal, market_data)
            if trade_signal:
                # Final trade signal generated
                self.logger.info("=== Signal Processing End ===")
                return trade_signal
        else:
            # Rate limit "no signal" messages to avoid spam
            current_time = time.time()
            if current_time - self.last_no_signal_log > 300:  # Log every 5 minutes
                self.logger.info("Final Signal: None - No trading action")
                self.last_no_signal_log = current_time
            # Always log processing end for debugging
            if not is_realtime_tick:
                self.logger.info("=== Signal Processing End ===")
        
        return None
    
    def _get_allocation_decision(self, market_data: MarketData, signals: dict) -> Optional[AllocationDecision]:
        """Get allocation decision from meta-allocator."""
        if not self.meta_allocator:
            return None
        
        try:
            # Use default performance metrics for now
            # TODO: Update to use actual performance metrics
            return self.meta_allocator.get_allocation(market_data, 0.0, 0.0)
        except Exception as e:
            self.logger.error(f"Error getting allocation decision: {e}")
            return None
    
    def _select_final_signal(self, signals: dict, allocation, is_realtime_tick: bool = False):
        """Select final signal based on allocation and signal strength."""
        if not signals:
            self.logger.debug("No signals generated")
            return None
        
        # Default equal weighting if no allocation
        if not allocation:
            # Select signal with highest confidence
            best_signal = max(signals.items(), key=lambda x: x[1].confidence)
            if not is_realtime_tick:
                self.logger.debug(f"Selected signal: {best_signal[0]} (confidence: {best_signal[1].confidence:.3f})")
            return best_signal[1]
        
        # Use allocation weights
        weighted_signals = []
        
        # Add mean reversion signal
        if 'mean_reversion' in signals:
            weighted_signals.append(('mean_reversion', signals['mean_reversion'], allocation.mean_reversion_weight))
        
        # Add momentum signal
        if 'momentum' in signals:
            weighted_signals.append(('momentum', signals['momentum'], allocation.momentum_weight))
        
        # Add volatility carry signal (if available and allocation supports it)
        if 'vol_carry' in signals and hasattr(allocation, 'vol_carry_weight'):
            weighted_signals.append(('vol_carry', signals['vol_carry'], allocation.vol_carry_weight))
        
        # Add volatility breakout signal (if available and allocation supports it)
        if 'vol_breakout' in signals and hasattr(allocation, 'vol_breakout_weight'):
            weighted_signals.append(('vol_breakout', signals['vol_breakout'], allocation.vol_breakout_weight))
        
        if not weighted_signals:
            return None
        
        # Select signal with highest weighted score
        best_signal = max(weighted_signals, key=lambda x: x[1].confidence * x[2])
        if not is_realtime_tick:
            self.logger.debug(f"Selected weighted signal: {best_signal[0]} (score: {best_signal[1].confidence * best_signal[2]:.3f})")
        
        return best_signal[1]
    
    def _convert_to_trade_signal(self, signal: Signal, market_data: MarketData) -> TradeSignal:
        """Convert strategy signal to trade signal - delegated to ExecutionEngine."""
        # This method is now handled by ExecutionEngine.execute_signal()
        # Keeping for backward compatibility
        if hasattr(signal, 'confidence') and hasattr(signal, 'action'):
            return self.mean_reversion.create_trade_signal(signal, market_data)
        return signal
    
    def get_execution_decision(self, trade_signal: TradeSignal, market_data: MarketData) -> Optional[ExecutionDecision]:
        """Get execution decision - delegated to ExecutionEngine."""
        return self.execution_engine.get_execution_decision(trade_signal, market_data)
    
    def cleanup(self):
        """Cleanup signal processor resources."""
        if hasattr(self, 'execution_engine'):
            self.execution_engine.cleanup()
        # Signal processor cleanup complete
    
    def get_execution_metrics(self):
        """Get execution metrics from the execution engine."""
        if hasattr(self, 'execution_engine'):
            return self.execution_engine.get_execution_metrics()
        return {}
    
