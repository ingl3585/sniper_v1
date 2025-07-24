"""
Volatility Breakout Strategy
Trades rapid expansions/contractions in volatility.
"""
from logging_config import get_logger
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np

from src.strategies.base_strategy import BaseStrategy, Signal
from src.infra.nt_bridge import MarketData
from config import SystemConfig
from src.strategies.technical_indicators import TechnicalIndicators


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility breakout strategy based on rapid volatility regime changes."""
    
    def __init__(self, config, system_config: SystemConfig, price_history_manager=None):
        super().__init__("VolatilityBreakout", config, system_config, price_history_manager)
        self.config = config
        self.logger = get_logger(__name__)
        
        # Breakout tracking
        self.volatility_history = {}  # Store volatility time series
        self.last_breakout_signal = None
        self.breakout_cooldown = None  # Prevent rapid-fire signals
        self.regime_transition_time = None
        
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """Generate volatility breakout signal."""
        try:
            # Update price history first
            self.update_price_history(market_data)
            
            # Check if we have sufficient data
            if not self._has_sufficient_data():
                self.logger.info("VolBreakout: Insufficient data for analysis")
                return None
            
            # Check cooldown period
            if self._is_in_cooldown():
                self.logger.info("VolBreakout: In cooldown period")
                return None
            
            # Analyze volatility breakout
            primary_signal = self._analyze_volatility_breakout(market_data)
            if not primary_signal:
                self.logger.info("VolBreakout: No breakout signal generated")
                return None
            
            # Apply regime filters
            if not self._validate_regime_transition(primary_signal, market_data):
                self.logger.info("VolBreakout: Regime transition validation failed")
                return None
            
            # Apply risk management
            if not self.should_trade(market_data):
                self.logger.info("VolBreakout: Risk management blocked trade")
                return None
            
            # Check confidence threshold
            if primary_signal.confidence < self.system_config.risk_management.min_confidence:
                self.logger.info(f"VolBreakout: Signal confidence {primary_signal.confidence:.3f} below minimum {self.system_config.risk_management.min_confidence}")
                return None
                
            # Set cooldown period
            self.breakout_cooldown = datetime.now() + timedelta(minutes=self.config.cooldown_minutes)
            
            self.logger.info(f"VolBreakout: Generated {['HOLD', 'BUY', 'SELL'][primary_signal.action]} signal with confidence {primary_signal.confidence:.3f}")
            return primary_signal
            
        except Exception as e:
            self.logger.error(f"VolBreakout: Error generating signal: {e}")
            return None
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have any data for analysis."""
        status = self.price_history_manager.get_status()
        
        # Check if we have any data at all (no hard requirements)
        timeframes_to_check = ['5m', '15m', '1h']
        
        # Only log detailed status occasionally to avoid spam
        if self.should_log_detailed_analysis():
            self.logger.info(f"VolBreakout: Data status - {status}")
        
        for timeframe in timeframes_to_check:
            available = status.get(timeframe, {}).get('data_points', 0)
            if available > 10:  # Just need some basic data points
                return True
        
        # Only log this occasionally to avoid spam
        if self.should_log_detailed_analysis():
            self.logger.info("VolBreakout: No sufficient data available yet")
        return False
    
    def _is_in_cooldown(self) -> bool:
        """Check if we're in cooldown period."""
        if self.breakout_cooldown is None:
            return False
        return datetime.now() < self.breakout_cooldown
    
    def _analyze_volatility_breakout(self, market_data: MarketData) -> Optional[Signal]:
        """Analyze volatility breakout conditions using guard clauses."""
        try:
            # Get breakout information with early return
            breakout_info = self._get_breakout_info('15m')
            if not breakout_info:
                return None
            
            # Extract breakout data
            breakout_data = self._extract_breakout_data(breakout_info)
            if not breakout_data:
                return None
            
            # Generate signal based on breakout direction
            if breakout_data['direction'] == 'up':
                return self._handle_volatility_expansion(market_data, breakout_data)
            elif breakout_data['direction'] == 'down':
                return self._handle_volatility_contraction(market_data, breakout_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"VolBreakout: Error in breakout analysis: {e}")
            return None
    
    def _get_breakout_info(self, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get volatility breakout information with validation."""
        self.logger.info(f"VolBreakout: Analyzing {timeframe} timeframe for breakouts")
        
        breakout_info = self.price_history_manager.calculate_volatility_breakout(
            timeframe, self.config.breakout_z_threshold
        )
        
        if not breakout_info:
            self.logger.info("VolBreakout: No breakout info returned from price history manager")
            return None
        
        self.logger.info(f"VolBreakout: Breakout info received: {breakout_info}")
        return breakout_info
    
    def _extract_breakout_data(self, breakout_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract and validate breakout data."""
        # Check for significant breakout
        if not breakout_info.get('is_breakout', False):
            self.logger.info(f"VolBreakout: No significant breakout detected")
            self.logger.info(f"VolBreakout: Current z-score: {breakout_info.get('z_score', 0.0):.3f}, threshold: {self.config.breakout_z_threshold}")
            return None
        
        breakout_data = {
            'strength': breakout_info.get('breakout_strength', 0.0),
            'direction': breakout_info.get('breakout_direction', 'none'),
            'current_vol': breakout_info.get('current_vol', 0.0),
            'mean_vol': breakout_info.get('mean_vol', 0.0)
        }
        
        self._log_breakout_analysis(breakout_data)
        return breakout_data
    
    def _handle_volatility_expansion(self, market_data: MarketData, breakout_data: Dict[str, Any]) -> Optional[Signal]:
        """Handle volatility expansion breakouts."""
        # Get price momentum data
        recent_prices = market_data.price_15m[-10:] if len(market_data.price_15m) >= 10 else []
        if len(recent_prices) < 5:
            return None
        
        price_momentum = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5]
        current_price = market_data.current_price
        
        # Strong upward momentum - buy signal
        if price_momentum > self.config.momentum_threshold:
            return self._create_expansion_signal(
                current_price, breakout_data, action=1, 
                reason=f"Vol breakout: Expansion + momentum (strength={breakout_data['strength']:.2f})"
            )
        
        # Strong downward momentum - sell signal  
        elif price_momentum < -self.config.momentum_threshold:
            return self._create_expansion_signal(
                current_price, breakout_data, action=2,
                reason=f"Vol breakout: Expansion + momentum down (strength={breakout_data['strength']:.2f})"
            )
        
        return None
    
    def _handle_volatility_contraction(self, market_data: MarketData, breakout_data: Dict[str, Any]) -> Optional[Signal]:
        """Handle volatility contraction (mean reversion opportunities)."""
        # Get price extension data
        recent_prices = market_data.price_15m[-20:] if len(market_data.price_15m) >= 20 else []
        if len(recent_prices) < 20:
            return None
        
        current_price = market_data.current_price
        price_mean = np.mean(recent_prices)
        price_std = np.std(recent_prices)
        
        if price_std <= 0:
            return None
        
        z_score = (current_price - price_mean) / price_std
        
        # Price extended upward - sell signal
        if z_score > self.config.price_extension_threshold:
            return self._create_contraction_signal(
                current_price, price_mean, recent_prices, breakout_data, 
                action=2, reason=f"Vol contraction: Mean reversion (z-score={z_score:.2f})"
            )
        
        # Price extended downward - buy signal
        elif z_score < -self.config.price_extension_threshold:
            return self._create_contraction_signal(
                current_price, price_mean, recent_prices, breakout_data,
                action=1, reason=f"Vol contraction: Mean reversion (z-score={z_score:.2f})"
            )
        
        return None
    
    def _create_expansion_signal(self, current_price: float, breakout_data: Dict[str, Any], 
                               action: int, reason: str) -> Signal:
        """Create signal for volatility expansion."""
        confidence = min(0.6 + breakout_data['strength'] * 0.1, 0.9)
        atr = current_price * 0.01  # Fallback ATR
        
        if action == 1:  # BUY
            stop_price = current_price - (atr * self.system_config.risk_management.stop_loss_atr_multiplier)
            target_price = current_price + (atr * self.config.target_atr_multiplier)
        else:  # SELL
            stop_price = current_price + (atr * self.system_config.risk_management.stop_loss_atr_multiplier)
            target_price = current_price - (atr * self.config.target_atr_multiplier)
        
        return Signal(
            action=action,
            confidence=confidence,
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            reason=reason
        )
    
    def _create_contraction_signal(self, current_price: float, target_price: float, 
                                 recent_prices: list, breakout_data: Dict[str, Any],
                                 action: int, reason: str) -> Signal:
        """Create signal for volatility contraction."""
        confidence = min(0.6 + breakout_data['strength'] * 0.1, 0.9)
        atr = self.calculate_atr_simple(recent_prices)
        
        if action == 1:  # BUY
            stop_price = current_price - (atr * self.system_config.risk_management.stop_loss_atr_multiplier)
        else:  # SELL
            stop_price = current_price + (atr * self.system_config.risk_management.stop_loss_atr_multiplier)
        
        return Signal(
            action=action,
            confidence=confidence,
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,  # Mean reversion target
            reason=reason
        )
    
    def _log_breakout_analysis(self, breakout_data: Dict[str, Any]):
        """Log breakout analysis details."""
        self.logger.info(f"VolBreakout Analysis:")
        self.logger.info(f"  Breakout direction: {breakout_data['direction']}")
        self.logger.info(f"  Breakout strength: {breakout_data['strength']:.2f}")
        self.logger.info(f"  Current vol: {breakout_data['current_vol']:.4f}")
        self.logger.info(f"  Mean vol: {breakout_data['mean_vol']:.4f}")
    
    def _validate_regime_transition(self, signal: Signal, market_data: MarketData) -> bool:
        """Validate that this represents a genuine regime transition."""
        try:
            # Check multiple timeframes for regime consistency
            timeframes = ['5m', '15m', '1h']
            regime_scores = []
            
            for timeframe in timeframes:
                breakout_info = self.price_history_manager.calculate_volatility_breakout(timeframe)
                if breakout_info:
                    strength = breakout_info.get('breakout_strength', 0.0)
                    if breakout_info.get('is_breakout', False):
                        regime_scores.append(strength)
            
            # Need at least 2 timeframes showing breakout
            if len(regime_scores) < 2:
                return False
            
            # Check average breakout strength
            avg_strength = np.mean(regime_scores)
            if avg_strength < self.config.min_regime_strength:
                return False
            
            # Check volume confirmation if available
            if len(market_data.volume_15m) >= 20:
                recent_volume = np.mean(market_data.volume_15m[-5:])
                avg_volume = np.mean(market_data.volume_15m[-20:])
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                
                # Breakouts should be accompanied by above-average volume
                if volume_ratio < self.config.min_volume_confirmation:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"VolBreakout: Error validating regime transition: {e}")
            return False
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get detailed strategy status."""
        try:
            vol_metrics = self.calculate_volatility_metrics(None)
            
            return {
                'name': self.name,
                'last_signal': self.last_breakout_signal,
                'cooldown_until': self.breakout_cooldown,
                'regime_transition_time': self.regime_transition_time,
                'volatility_metrics': vol_metrics,
                'is_in_cooldown': self._is_in_cooldown()
            }
        except Exception as e:
            self.logger.error(f"VolBreakout: Error getting strategy status: {e}")
            return {'name': self.name, 'status': 'error'}