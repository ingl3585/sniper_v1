"""
Risk Management System
Handles position limits, daily loss limits, and emergency actions.
"""
import time
from logging_config import get_logger
from src.infra.nt_bridge import MarketData, TradeSignal


class RiskManager:
    """Manages trading risks and position limits."""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)
        self.logger.set_context(component='risk_manager')
        self.connection_manager = None  # Will be set by TradingSystem
    
    def should_trade(self, market_data: MarketData) -> bool:
        """Check if conditions are suitable for trading."""
        # Check if historical data is ready before any trading
        if self.connection_manager and not self.connection_manager.is_ready_for_trading():
            # Don't spam log this - only log occasionally
            if not hasattr(self, '_last_historical_warning') or (time.time() - getattr(self, '_last_historical_warning', 0)) > 30:
                # Waiting for historical data
                self._last_historical_warning = time.time()
            return False
        
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
    
    def check_daily_limits(self, market_data: MarketData) -> bool:
        """Check if daily risk limits are satisfied."""
        try:
            # Check daily loss limit
            daily_loss_limit = market_data.account_balance * self.config.trading.max_daily_loss_pct
            if market_data.daily_pnl < -daily_loss_limit:
                self.logger.warning(f"Daily loss limit exceeded: ${market_data.daily_pnl:.2f} < -${daily_loss_limit:.2f}")
                return False
            
            # Check account balance
            if market_data.account_balance < self.config.trading.min_account_balance:
                self.logger.warning(f"Account balance too low: ${market_data.account_balance:.2f} < ${self.config.trading.min_account_balance:.2f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking daily limits: {e}")
            # Return False on error to be safe
            return False
    
    def validate_trade_signal(self, trade_signal, market_data: MarketData) -> bool:
        """Validate a trade signal against risk parameters."""
        try:
            # First check if we should trade at all
            if not self.should_trade(market_data):
                return False
            
            # Additional signal-specific validations could go here
            # For now, if general trading conditions are met, approve the signal
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating trade signal: {e}")
            return False
    
    def validate_realtime_signal(self, trade_signal, market_data: MarketData, tick_data: dict) -> bool:
        """Validate a real-time trade signal with additional tick-level checks."""
        try:
            # First do standard validation
            if not self.validate_trade_signal(trade_signal, market_data):
                return False
            
            # Additional real-time specific checks
            tick_age = tick_data.get('tick_age_seconds', 0)
            if tick_age > 5.0:  # Don't trade on stale tick data
                self.logger.warning(f"Real-time signal blocked: tick data too old ({tick_age:.1f}s)")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating realtime signal: {e}")
            return False
    
