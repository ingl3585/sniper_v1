"""
Volatility Breakout Strategy
Trades rapid expansions/contractions in volatility.
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np

from src.strategies.base_strategy import BaseStrategy, Signal
from src.infra.nt_bridge import MarketData
from src.config import SystemConfig


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility breakout strategy based on rapid volatility regime changes."""
    
    def __init__(self, config, system_config: SystemConfig, price_history_manager=None):
        super().__init__("VolatilityBreakout", config, system_config, price_history_manager)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
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
        """Check if we have sufficient data for analysis."""
        status = self.price_history_manager.get_status()
        
        # Need at least 150 data points for robust breakout detection
        required_data = {
            '5m': 150,   # 12+ hours
            '15m': 150,  # 37+ hours
            '1h': 100    # 4+ days
        }
        
        self.logger.info(f"VolBreakout: Checking data sufficiency - Status: {status}")
        
        for timeframe, required in required_data.items():
            available = status.get(timeframe, {}).get('data_points', 0)
            self.logger.info(f"VolBreakout: {timeframe} - Available: {available}, Required: {required}")
            if available < required:
                self.logger.info(f"VolBreakout: Insufficient data for {timeframe} ({available} < {required})")
                return False
        
        self.logger.info("VolBreakout: All timeframes have sufficient data")
        return True
    
    def _is_in_cooldown(self) -> bool:
        """Check if we're in cooldown period."""
        if self.breakout_cooldown is None:
            return False
        return datetime.now() < self.breakout_cooldown
    
    def _analyze_volatility_breakout(self, market_data: MarketData) -> Optional[Signal]:
        """Analyze volatility breakout conditions."""
        try:
            # Focus on 15m timeframe for primary breakout detection
            primary_timeframe = '15m'
            
            self.logger.info(f"VolBreakout: Analyzing {primary_timeframe} timeframe for breakouts")
            
            # Get volatility breakout information
            breakout_info = self.price_history_manager.calculate_volatility_breakout(
                primary_timeframe, self.config.breakout_z_threshold
            )
            
            if not breakout_info:
                self.logger.info("VolBreakout: No breakout info returned from price history manager")
                return None
            
            self.logger.info(f"VolBreakout: Breakout info received: {breakout_info}")
            
            # Check for significant breakout
            if not breakout_info.get('is_breakout', False):
                self.logger.info(f"VolBreakout: No significant breakout detected (is_breakout: {breakout_info.get('is_breakout', False)})")
                self.logger.info(f"VolBreakout: Current z-score: {breakout_info.get('z_score', 0.0):.3f}, threshold: {self.config.breakout_z_threshold}")
                return None
            
            breakout_strength = breakout_info.get('breakout_strength', 0.0)
            breakout_direction = breakout_info.get('breakout_direction', 'none')
            current_vol = breakout_info.get('current_vol', 0.0)
            mean_vol = breakout_info.get('mean_vol', 0.0)
            
            # Determine trading direction based on breakout type and market context
            signal = None
            current_price = market_data.current_price
            
            if breakout_direction == 'up':
                # Volatility expansion - typically precedes large price moves
                # Strategy: Buy in direction of anticipated move
                
                # Check price momentum to determine direction
                recent_prices = market_data.price_15m[-10:] if len(market_data.price_15m) >= 10 else []
                if len(recent_prices) >= 5:
                    price_momentum = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5]
                    
                    # Strong upward price momentum with vol expansion = buy
                    if price_momentum > self.config.momentum_threshold:
                        confidence = min(0.6 + breakout_strength * 0.1, 0.9)
                        atr = self.calculate_atr_simple(market_data.price_15m[-20:]) if len(market_data.price_15m) >= 20 else current_price * 0.01
                        
                        stop_price = current_price - (atr * self.system_config.risk_management.stop_loss_atr_multiplier)
                        target_price = current_price + (atr * self.config.target_atr_multiplier)
                        
                        signal = Signal(
                            action=1,  # BUY
                            confidence=confidence,
                            entry_price=current_price,
                            stop_price=stop_price,
                            target_price=target_price,
                            reason=f"Vol breakout: Expansion + momentum (strength={breakout_strength:.2f})"
                        )
                    
                    # Strong downward price momentum with vol expansion = sell
                    elif price_momentum < -self.config.momentum_threshold:
                        confidence = min(0.6 + breakout_strength * 0.1, 0.9)
                        atr = self.calculate_atr_simple(market_data.price_15m[-20:]) if len(market_data.price_15m) >= 20 else current_price * 0.01
                        
                        stop_price = current_price + (atr * self.system_config.risk_management.stop_loss_atr_multiplier)
                        target_price = current_price - (atr * self.config.target_atr_multiplier)
                        
                        signal = Signal(
                            action=2,  # SELL
                            confidence=confidence,
                            entry_price=current_price,
                            stop_price=stop_price,
                            target_price=target_price,
                            reason=f"Vol breakout: Expansion + momentum down (strength={breakout_strength:.2f})"
                        )
            
            elif breakout_direction == 'down':
                # Volatility contraction - typically mean reversion opportunity
                # Strategy: Fade extreme moves, expect consolidation
                
                # Check if price is extended from mean
                recent_prices = market_data.price_15m[-20:] if len(market_data.price_15m) >= 20 else []
                if len(recent_prices) >= 20:
                    price_mean = np.mean(recent_prices)
                    price_std = np.std(recent_prices)
                    z_score = (current_price - price_mean) / price_std if price_std > 0 else 0
                    
                    # Price extended upward with vol contraction = sell
                    if z_score > self.config.price_extension_threshold:
                        confidence = min(0.6 + breakout_strength * 0.1, 0.9)
                        atr = self.calculate_atr_simple(recent_prices) if len(recent_prices) >= 20 else current_price * 0.01
                        
                        stop_price = current_price + (atr * self.system_config.risk_management.stop_loss_atr_multiplier)
                        target_price = price_mean  # Target mean reversion
                        
                        signal = Signal(
                            action=2,  # SELL
                            confidence=confidence,
                            entry_price=current_price,
                            stop_price=stop_price,
                            target_price=target_price,
                            reason=f"Vol contraction: Mean reversion (z-score={z_score:.2f})"
                        )
                    
                    # Price extended downward with vol contraction = buy
                    elif z_score < -self.config.price_extension_threshold:
                        confidence = min(0.6 + breakout_strength * 0.1, 0.9)
                        atr = self.calculate_atr_simple(recent_prices) if len(recent_prices) >= 20 else current_price * 0.01
                        
                        stop_price = current_price - (atr * self.system_config.risk_management.stop_loss_atr_multiplier)
                        target_price = price_mean  # Target mean reversion
                        
                        signal = Signal(
                            action=1,  # BUY
                            confidence=confidence,
                            entry_price=current_price,
                            stop_price=stop_price,
                            target_price=target_price,
                            reason=f"Vol contraction: Mean reversion (z-score={z_score:.2f})"
                        )
            
            # Log breakout analysis
            self.logger.info(f"VolBreakout Analysis:")
            self.logger.info(f"  Breakout direction: {breakout_direction}")
            self.logger.info(f"  Breakout strength: {breakout_strength:.2f}")
            self.logger.info(f"  Current vol: {current_vol:.4f}")
            self.logger.info(f"  Mean vol: {mean_vol:.4f}")
            self.logger.info(f"  Signal: {signal.action if signal else 'None'}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"VolBreakout: Error in breakout analysis: {e}")
            return None
    
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