"""
Position Sizing Engine
Centralized position sizing logic for risk management.
"""
from typing import Union, Optional
import numpy as np

from logging_config import get_logger
from constants import RISK

log = get_logger(__name__)


class PositionSizer:
    """Centralized position sizing calculations for risk management."""
    
    def __init__(self, account_balance: float = 100000.0):
        """Initialize position sizer.
        
        Args:
            account_balance: Total account balance for risk calculations
        """
        self.account_balance = account_balance
    
    def calculate_position_size(
        self, 
        entry_price: float, 
        stop_price: float, 
        risk_amount: Optional[float] = None,
        max_position_value: Optional[float] = None
    ) -> int:
        """Calculate position size based on risk management rules.
        
        Args:
            entry_price: Intended entry price
            stop_price: Stop loss price
            risk_amount: Dollar amount to risk (defaults to RISK.PER_TRADE * account_balance)
            max_position_value: Maximum position value allowed
            
        Returns:
            Position size in contracts/shares
        """
        if entry_price <= 0 or stop_price <= 0:
            log.warning(f"Invalid prices: entry={entry_price}, stop={stop_price}")
            return 0
        
        # Calculate risk per contract
        risk_per_contract = abs(entry_price - stop_price)
        if risk_per_contract <= 0:
            log.warning(f"Invalid risk per contract: {risk_per_contract}")
            return 0
        
        # Default risk amount
        if risk_amount is None:
            risk_amount = self.account_balance * RISK.PER_TRADE
        
        # Calculate base position size
        position_size = int(risk_amount / risk_per_contract)
        
        # Apply maximum position value constraint
        if max_position_value is not None:
            max_size_by_value = int(max_position_value / entry_price)
            position_size = min(position_size, max_size_by_value)
        
        # Apply maximum account percentage constraint
        position_value = position_size * entry_price
        max_position_by_account = self.account_balance * RISK.MAX_POSITION_SIZE
        if position_value > max_position_by_account:
            position_size = int(max_position_by_account / entry_price)
        
        log.info(f"Position size calculated: {position_size} contracts, "
                f"risk_per_contract=${risk_per_contract:.2f}, "
                f"total_risk=${position_size * risk_per_contract:.2f}")
        
        return max(0, position_size)
    
    def calculate_kelly_size(
        self, 
        win_rate: float, 
        avg_win: float, 
        avg_loss: float,
        max_kelly_fraction: float = 0.25
    ) -> float:
        """Calculate Kelly criterion position size.
        
        Args:
            win_rate: Historical win rate (0.0 to 1.0)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive value)
            max_kelly_fraction: Maximum Kelly fraction to use
            
        Returns:
            Kelly fraction of account to risk
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Cap at maximum allowed fraction
        kelly_fraction = min(kelly_fraction, max_kelly_fraction)
        
        # Don't allow negative Kelly (means strategy has negative expectancy)
        kelly_fraction = max(0.0, kelly_fraction)
        
        log.info(f"Kelly fraction calculated: {kelly_fraction:.4f} "
                f"(win_rate={win_rate:.2f}, b={b:.2f})")
        
        return kelly_fraction
    
    def update_account_balance(self, new_balance: float) -> None:
        """Update account balance for position sizing calculations.
        
        Args:
            new_balance: New account balance
        """
        log.info(f"Account balance updated: ${self.account_balance:.2f} -> ${new_balance:.2f}")
        self.account_balance = new_balance