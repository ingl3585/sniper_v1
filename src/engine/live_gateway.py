"""
Execution Engine
Centralized order management and execution logic.
"""
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from logging_config import get_logger
from src.infra.nt_bridge import MarketData, TradeSignal
from src.strategies.base_strategy import Signal
# PPOExecutionAgent functionality removed - using direct execution for now


@dataclass
class ExecutionDecision:
    """Execution decision containing order type and parameters."""
    order_type: str  # "market", "limit", "stop"
    limit_offset: float = 0.0  # Price offset for limit orders
    confidence: float = 1.0  # Execution confidence
    expected_slippage: float = 0.0  # Expected slippage
    urgency_score: float = 1.0  # Urgency for execution


class ExecutionEngine:
    """Centralized execution engine for managing order placement and execution logic."""
    
    def __init__(self, config, execution_agent = None):
        """Initialize the execution engine.
        
        Args:
            config: System configuration
            execution_agent: Optional execution agent for smart order types
        """
        self.config = config
        self.execution_agent = execution_agent
        self.logger = get_logger(__name__)
        self.logger.set_context(component='execution_engine')
        
        # Execution tracking
        self.pending_orders = {}
        self.execution_history = []
        self.order_id_counter = 1
    
    def execute_signal(self, signal: Signal, market_data: MarketData) -> Optional[TradeSignal]:
        """Convert strategy signal to executable trade signal.
        
        Args:
            signal: Strategy signal to execute
            market_data: Current market data
            
        Returns:
            Trade signal ready for execution or None if execution blocked
        """
        try:
            # Validate signal
            if not self._validate_signal(signal, market_data):
                return None
            
            # Convert to trade signal
            trade_signal = self._convert_to_trade_signal(signal, market_data)
            
            # Get execution decision (order type, timing, etc.)
            execution_decision = self.get_execution_decision(trade_signal, market_data)
            
            # Apply execution decision to trade signal
            if execution_decision:
                trade_signal = self._apply_execution_decision(trade_signal, execution_decision)
            
            # Register pending order
            order_id = self._register_pending_order(trade_signal, signal, execution_decision)
            trade_signal.order_id = order_id
            
            self.logger.info(f"ExecutionEngine: Created trade signal {order_id} - "
                           f"Action={trade_signal.action}, Size={trade_signal.position_size}, "
                           f"Type={execution_decision.order_type if execution_decision else 'market'}")
            
            return trade_signal
            
        except Exception as e:
            self.logger.error(f"ExecutionEngine: Error executing signal: {e}")
            return None
    
    def get_execution_decision(self, trade_signal: TradeSignal, market_data: MarketData) -> Optional[ExecutionDecision]:
        """Get execution decision from RL agent or apply default logic.
        
        Args:
            trade_signal: Trade signal to execute
            market_data: Current market data
            
        Returns:
            Execution decision with order type and parameters
        """
        try:
            # Check if market orders are forced
            if self.config.trading.force_market_orders:
                self.logger.debug("FORCE_MARKET_ORDERS enabled - using market orders only")
                return ExecutionDecision(
                    order_type="market",
                    limit_offset=0.0,
                    confidence=1.0,
                    expected_slippage=self._estimate_market_slippage(market_data),
                    urgency_score=1.0
                )
            
            # Use RL agent if available
            if self.execution_agent:
                return self._get_rl_execution_decision(trade_signal, market_data)
            
            # Fallback to intelligent default execution
            return self._get_default_execution_decision(trade_signal, market_data)
            
        except Exception as e:
            self.logger.error(f"ExecutionEngine: Error getting execution decision: {e}")
            return self._get_emergency_execution_decision()
    
    def update_order_status(self, order_id: str, status: str, fill_info: Optional[Dict] = None):
        """Update the status of a pending order.
        
        Args:
            order_id: Order ID to update
            status: New order status ('filled', 'cancelled', 'rejected', etc.)
            fill_info: Optional fill information
        """
        if order_id in self.pending_orders:
            self.pending_orders[order_id]['status'] = status
            self.pending_orders[order_id]['updated_at'] = datetime.now()
            
            if fill_info:
                self.pending_orders[order_id]['fill_info'] = fill_info
            
            # Move to history if completed
            if status in ['filled', 'cancelled', 'rejected']:
                self.execution_history.append(self.pending_orders[order_id])
                del self.pending_orders[order_id]
            
            self.logger.info(f"ExecutionEngine: Order {order_id} status updated to {status}")
    
    def get_pending_orders(self) -> Dict[str, Dict]:
        """Get all pending orders.
        
        Returns:
            Dictionary of pending orders
        """
        return self.pending_orders.copy()
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics.
        
        Returns:
            Dictionary of execution metrics
        """
        total_orders = len(self.execution_history)
        if total_orders == 0:
            return {
                'total_orders': 0,
                'fill_rate': 0.0,
                'avg_slippage': 0.0,
                'pending_orders': len(self.pending_orders)
            }
        
        filled_orders = sum(1 for order in self.execution_history if order['status'] == 'filled')
        fill_rate = filled_orders / total_orders
        
        # Calculate average slippage for filled orders
        slippage_values = []
        for order in self.execution_history:
            if order['status'] == 'filled' and 'fill_info' in order:
                fill_info = order['fill_info']
                expected_price = order['trade_signal'].entry_price
                actual_price = fill_info.get('fill_price', expected_price)
                slippage = abs(actual_price - expected_price) / expected_price
                slippage_values.append(slippage)
        
        avg_slippage = sum(slippage_values) / len(slippage_values) if slippage_values else 0.0
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'fill_rate': fill_rate,
            'avg_slippage': avg_slippage,
            'pending_orders': len(self.pending_orders)
        }
    
    def cleanup(self):
        """Cleanup execution engine resources."""
        self.logger.info("ExecutionEngine: Cleaning up resources...")
        
        # Cancel all pending orders
        for order_id in list(self.pending_orders.keys()):
            self.update_order_status(order_id, 'cancelled')
        
        self.pending_orders.clear()
        self.logger.info("ExecutionEngine: Cleanup complete")
    
    def _validate_signal(self, signal: Signal, market_data: MarketData) -> bool:
        """Validate signal before execution.
        
        Args:
            signal: Signal to validate
            market_data: Current market data
            
        Returns:
            True if signal is valid for execution
        """
        # Check signal validity
        if not signal or signal.action == 0:
            return False
        
        # Check confidence threshold
        min_confidence = self.config.risk_management.min_confidence
        if signal.confidence < min_confidence:
            self.logger.debug(f"Signal confidence {signal.confidence:.3f} below minimum {min_confidence}")
            return False
        
        # Check market conditions
        if not self._check_market_conditions(market_data):
            return False
        
        return True
    
    def _convert_to_trade_signal(self, signal: Signal, market_data: MarketData) -> TradeSignal:
        """Convert strategy signal to trade signal.
        
        Args:
            signal: Strategy signal
            market_data: Current market data
            
        Returns:
            Trade signal ready for execution
        """
        # Calculate position size if not specified
        position_size = signal.size
        
        # Debug logging for position sizing
        self.logger.info(f"Position sizing debug: signal.size={signal.size}, "
                        f"account_balance={market_data.account_balance}, "
                        f"buying_power={market_data.buying_power}, "
                        f"entry_price={signal.entry_price}, stop_price={signal.stop_price}")
        
        if position_size == 1 and signal.stop_price:
            # Calculate risk-based position size
            risk_per_share = abs(signal.entry_price - signal.stop_price)
            
            # For futures like MNQ, use a default account size if balance is 0 or unavailable
            account_balance = market_data.account_balance
            if account_balance <= 0:
                account_balance = 50000.0  # Default $50K account for MNQ
                self.logger.info(f"Using default account balance: ${account_balance}")
            
            max_risk = account_balance * self.config.risk_management.risk_per_trade
            calculated_size = int(max_risk / risk_per_share) if risk_per_share > 0 else 1
            
            self.logger.info(f"Risk calculation: max_risk=${max_risk:.2f}, "
                            f"risk_per_share=${risk_per_share:.2f}, calculated_size={calculated_size}")
            
            # Apply position limits - for MNQ, max 5 contracts  
            max_size = self.config.risk_management.max_position_size
            position_size = max(1, min(calculated_size, max_size))
            
            self.logger.info(f"Final position size: {position_size} (max allowed: {max_size})")
        
        return TradeSignal(
            action=signal.action,
            position_size=position_size,
            confidence=signal.confidence,
            use_stop=signal.stop_price is not None,
            stop_price=signal.stop_price or 0.0,
            use_target=signal.target_price is not None,
            target_price=signal.target_price or 0.0,
            entry_price=signal.entry_price,
            reason=signal.reason,
            timestamp=signal.timestamp or datetime.now()
        )
    
    def _get_rl_execution_decision(self, trade_signal: TradeSignal, market_data: MarketData) -> Optional[ExecutionDecision]:
        """Get execution decision from RL agent.
        
        Args:
            trade_signal: Trade signal
            market_data: Market data
            
        Returns:
            RL execution decision
        """
        try:
            urgency = min(1.0, trade_signal.confidence + market_data.volatility)
            decision = self.execution_agent.get_execution_decision(
                market_data, trade_signal.position_size, urgency
            )
            self.logger.debug(f"RL execution decision: {decision.order_type}")
            return decision
        except Exception as e:
            self.logger.error(f"Error from RL execution agent: {e}")
            return self._get_default_execution_decision(trade_signal, market_data)
    
    def _get_default_execution_decision(self, trade_signal: TradeSignal, market_data: MarketData) -> ExecutionDecision:
        """Get default execution decision based on market conditions.
        
        Args:
            trade_signal: Trade signal
            market_data: Market data
            
        Returns:
            Default execution decision
        """
        # Use market orders for high confidence/urgency
        if trade_signal.confidence > 0.8 or market_data.volatility > 0.05:
            return ExecutionDecision(
                order_type="market",
                limit_offset=0.0,
                confidence=0.9,
                expected_slippage=self._estimate_market_slippage(market_data),
                urgency_score=0.9
            )
        
        # Use limit orders with small offset for medium confidence
        elif trade_signal.confidence > 0.6:
            spread = self._estimate_bid_ask_spread(market_data)
            offset = spread * 0.3  # 30% of spread
            
            return ExecutionDecision(
                order_type="limit",
                limit_offset=offset if trade_signal.action == 1 else -offset,
                confidence=0.7,
                expected_slippage=offset / market_data.current_price,
                urgency_score=0.7
            )
        
        # Conservative limit orders for low confidence
        else:
            spread = self._estimate_bid_ask_spread(market_data)
            offset = spread * 0.5  # 50% of spread
            
            return ExecutionDecision(
                order_type="limit",
                limit_offset=offset if trade_signal.action == 1 else -offset,
                confidence=0.6,
                expected_slippage=0.0001,
                urgency_score=0.5
            )
    
    def _get_emergency_execution_decision(self) -> ExecutionDecision:
        """Get emergency execution decision for error cases.
        
        Returns:
            Emergency execution decision
        """
        return ExecutionDecision(
            order_type="market",
            limit_offset=0.0,
            confidence=0.5,
            expected_slippage=0.002,  # Conservative slippage estimate
            urgency_score=0.5
        )
    
    def _apply_execution_decision(self, trade_signal: TradeSignal, execution_decision: ExecutionDecision) -> TradeSignal:
        """Apply execution decision parameters to trade signal.
        
        Args:
            trade_signal: Trade signal to modify
            execution_decision: Execution decision to apply
            
        Returns:
            Modified trade signal
        """
        # Set order type specific parameters
        if execution_decision.order_type == "limit":
            if trade_signal.action == 1:  # Buy
                trade_signal.entry_price += execution_decision.limit_offset
            else:  # Sell
                trade_signal.entry_price -= execution_decision.limit_offset
        
        # Add execution metadata
        trade_signal.order_type = execution_decision.order_type
        trade_signal.expected_slippage = execution_decision.expected_slippage
        trade_signal.urgency_score = execution_decision.urgency_score
        
        return trade_signal
    
    def _register_pending_order(self, trade_signal: TradeSignal, original_signal: Signal, 
                              execution_decision: Optional[ExecutionDecision]) -> str:
        """Register a pending order for tracking.
        
        Args:
            trade_signal: Trade signal
            original_signal: Original strategy signal
            execution_decision: Execution decision used
            
        Returns:
            Order ID
        """
        order_id = f"ORD_{self.order_id_counter:06d}"
        self.order_id_counter += 1
        
        self.pending_orders[order_id] = {
            'order_id': order_id,
            'trade_signal': trade_signal,
            'original_signal': original_signal,
            'execution_decision': execution_decision,
            'status': 'pending',
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        return order_id
    
    def _check_market_conditions(self, market_data: MarketData) -> bool:
        """Check if market conditions are suitable for execution.
        
        Args:
            market_data: Market data to check
            
        Returns:
            True if conditions are suitable
        """
        # Check volatility
        if market_data.volatility > 0.1:  # 10% volatility threshold
            self.logger.debug("Market volatility too high for execution")
            return False
        
        # Check account balance
        if market_data.account_balance <= 0:
            self.logger.debug("Account balance insufficient")
            return False
        
        return True
    
    def _estimate_market_slippage(self, market_data: MarketData) -> float:
        """Estimate expected slippage for market orders.
        
        Args:
            market_data: Market data
            
        Returns:
            Estimated slippage as fraction
        """
        # Base slippage estimate
        base_slippage = 0.0005  # 0.05% base slippage
        
        # Adjust for volatility
        vol_adjustment = market_data.volatility * 2.0
        
        # Adjust for time of day (if available)
        # TODO: Add time-based slippage adjustments
        
        return min(base_slippage + vol_adjustment, 0.002)  # Cap at 0.2%
    
    def _estimate_bid_ask_spread(self, market_data: MarketData) -> float:
        """Estimate bid-ask spread.
        
        Args:
            market_data: Market data
            
        Returns:
            Estimated spread in price units
        """
        # For MNQ, typical spread is $0.25-$1.00
        base_spread = 0.25
        
        # Adjust for volatility
        vol_spread = market_data.volatility * market_data.current_price * 0.01
        
        return base_spread + vol_spread