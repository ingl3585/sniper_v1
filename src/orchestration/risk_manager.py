"""
Risk Management System
Handles position limits, daily loss limits, and emergency actions.
"""
import logging
import time
from src.infra.nt_bridge import MarketData, TradeSignal


class RiskManager:
    """Manages trading risks and position limits."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def should_trade(self, market_data: MarketData) -> bool:
        """Check if conditions are suitable for trading."""
        # Risk check
        
        # Check account balance
        if market_data.account_balance < self.config.trading.min_account_balance:
            self.logger.warning(f"RISK BLOCK: Account balance too low: ${market_data.account_balance:.2f} < ${self.config.trading.min_account_balance:.2f}")
            return False
        
        # Check position limits (reduce spam logging)
        if abs(market_data.open_positions) >= self.config.risk_management.max_position_size:
            # Only log position limit occasionally to avoid spam
            if not hasattr(self, '_last_position_warning') or (time.time() - getattr(self, '_last_position_warning', 0)) > 10:
                self.logger.warning(f"RISK BLOCK: Position limit reached: {market_data.open_positions} >= {self.config.risk_management.max_position_size}")
                self._last_position_warning = time.time()
            return False
        
        # Check daily loss limit
        daily_loss_limit = market_data.account_balance * self.config.trading.max_daily_loss_pct
        if market_data.daily_pnl < -daily_loss_limit:
            self.logger.warning(f"RISK BLOCK: Daily loss limit exceeded: ${market_data.daily_pnl:.2f} < -${daily_loss_limit:.2f}")
            return False
        
        # Check volatility (avoid trading in extreme conditions)
        if market_data.volatility > 0.1:  # 10% volatility threshold
            self.logger.warning(f"RISK BLOCK: Volatility too high: {market_data.volatility:.4f} > 0.1000")
            return False
        
        # Risk check passed
        return True
    
    def needs_emergency_close(self, market_data: MarketData) -> bool:
        """Check if emergency close is needed."""
        daily_loss_limit = market_data.account_balance * self.config.trading.max_daily_loss_pct
        return market_data.daily_pnl < -daily_loss_limit
    
    def create_emergency_signal(self) -> TradeSignal:
        """Create emergency CLOSE_ALL signal."""
        return TradeSignal(
            action=0,  # CLOSE_ALL
            position_size=0,
            confidence=1.0
        )
    
