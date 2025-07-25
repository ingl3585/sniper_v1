"""
Strategy Factory
Centralized factory for creating trading strategies with proper dependency injection.
"""
from typing import Dict, List, Optional, Type
from logging_config import get_logger
from src.strategies.base_strategy import BaseStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.volatility_carry import VolatilityCarryStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
from storage import PriceHistoryManager
from config import SystemConfig


class StrategyFactory:
    """Factory for creating trading strategies with dependency injection."""
    
    # Strategy registry mapping strategy names to classes
    STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
        'mean_reversion': MeanReversionStrategy,
        'momentum': MomentumStrategy,
        'volatility_carry': VolatilityCarryStrategy,
        'volatility_breakout': VolatilityBreakoutStrategy,
    }
    
    def __init__(self, system_config: SystemConfig, price_history_manager: PriceHistoryManager):
        """Initialize the strategy factory.
        
        Args:
            system_config: System configuration containing all strategy configs
            price_history_manager: Shared price history manager instance
        """
        self.system_config = system_config
        self.price_history_manager = price_history_manager
        self.logger = get_logger(__name__)
        self.logger.set_context(component='strategy_factory')
        self._created_strategies: Dict[str, BaseStrategy] = {}
    
    def create_strategy(self, strategy_name: str) -> Optional[BaseStrategy]:
        """Create a single strategy instance.
        
        Args:
            strategy_name: Name of strategy to create (e.g., 'mean_reversion')
            
        Returns:
            Strategy instance or None if strategy name is invalid
        """
        if strategy_name not in self.STRATEGY_REGISTRY:
            self.logger.error(f"Unknown strategy: {strategy_name}. Available: {list(self.STRATEGY_REGISTRY.keys())}")
            return None
        
        if strategy_name in self._created_strategies:
            # Returning existing instance
            return self._created_strategies[strategy_name]
        
        try:
            strategy_class = self.STRATEGY_REGISTRY[strategy_name]
            strategy_config = self._get_strategy_config(strategy_name)
            
            if strategy_config is None:
                self.logger.error(f"No configuration found for strategy: {strategy_name}")
                return None
            
            # Create strategy instance with dependency injection
            strategy = strategy_class(
                config=strategy_config,
                system_config=self.system_config,
                price_history_manager=self.price_history_manager
            )
            
            self._created_strategies[strategy_name] = strategy
            self.logger.debug(f"Created strategy: {strategy_name}")
            return strategy
            
        except Exception as e:
            self.logger.error(f"Failed to create strategy {strategy_name}: {e}")
            return None
    
    def create_all_strategies(self) -> Dict[str, BaseStrategy]:
        """Create all available strategies.
        
        Returns:
            Dictionary mapping strategy names to instances
        """
        strategies = {}
        
        for strategy_name in self.STRATEGY_REGISTRY.keys():
            strategy = self.create_strategy(strategy_name)
            if strategy:
                strategies[strategy_name] = strategy
            else:
                self.logger.warning(f"Failed to create strategy: {strategy_name}")
        
        # Strategies created
        return strategies
    
    def create_enabled_strategies(self) -> Dict[str, BaseStrategy]:
        """Create only enabled strategies based on system configuration.
        
        Returns:
            Dictionary mapping strategy names to instances
        """
        enabled_strategies = {}
        
        # Check which strategies are enabled in config
        strategy_enablement = {
            'mean_reversion': getattr(self.system_config.trading, 'enable_mean_reversion', True),
            'momentum': getattr(self.system_config.trading, 'enable_momentum', True),
            'volatility_carry': getattr(self.system_config.trading, 'enable_vol_carry', True),
            'volatility_breakout': getattr(self.system_config.trading, 'enable_vol_breakout', True),
        }
        
        for strategy_name, is_enabled in strategy_enablement.items():
            if is_enabled:
                strategy = self.create_strategy(strategy_name)
                if strategy:
                    enabled_strategies[strategy_name] = strategy
                else:
                    self.logger.warning(f"Failed to create enabled strategy: {strategy_name}")
            else:
                # Strategy disabled in config
                pass
        
        # Strategies initialized
        return enabled_strategies
    
    def get_strategy_list(self) -> List[BaseStrategy]:
        """Get list of all created strategies.
        
        Returns:
            List of strategy instances
        """
        return list(self._created_strategies.values())
    
    def get_strategy_names(self) -> List[str]:
        """Get list of available strategy names.
        
        Returns:
            List of strategy names
        """
        return list(self.STRATEGY_REGISTRY.keys())
    
    def _get_strategy_config(self, strategy_name: str):
        """Get configuration for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy configuration object or None
        """
        config_mapping = {
            'mean_reversion': self.system_config.mean_reversion,
            'momentum': self.system_config.momentum,
            'volatility_carry': self.system_config.vol_carry,
            'volatility_breakout': self.system_config.vol_breakout,
        }
        
        return config_mapping.get(strategy_name)
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseStrategy]):
        """Register a new strategy class.
        
        Args:
            name: Strategy name for registration
            strategy_class: Strategy class to register
        """
        cls.STRATEGY_REGISTRY[name] = strategy_class
        get_logger(__name__).info(f"Registered new strategy: {name}")
    
    def get_strategy_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all created strategies.
        
        Returns:
            Dictionary mapping strategy names to their metrics
        """
        metrics = {}
        for name, strategy in self._created_strategies.items():
            try:
                metrics[name] = strategy.get_strategy_metrics()
            except Exception as e:
                metrics[name] = {'error': str(e)}
        
        return metrics
    
    def cleanup(self):
        """Cleanup resources for all strategies."""
        for strategy_name, strategy in self._created_strategies.items():
            try:
                if hasattr(strategy, 'cleanup'):
                    strategy.cleanup()
                # Strategy cleaned up
            except Exception as e:
                self.logger.error(f"Error cleaning up strategy {strategy_name}: {e}")
        
        self._created_strategies.clear()