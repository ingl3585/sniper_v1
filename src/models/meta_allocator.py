"""
ML Meta-Allocator
LightGBM classifier to allocate capital between mean reversion and momentum strategies.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pickle
import os
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from src.infra.nt_bridge import MarketData


@dataclass
class AllocationDecision:
    """Capital allocation decision from ML model."""
    mean_reversion_weight: float
    momentum_weight: float
    vol_carry_weight: float
    vol_breakout_weight: float
    confidence: float
    regime: str
    features: Dict[str, float]


class MetaAllocator:
    """ML-based meta-allocator for strategy selection."""
    
    def __init__(self, model_path: str = "data/meta_allocator_model.pkl", config=None):
        from src.config import MetaAllocatorConfig
        if config is None:
            config = MetaAllocatorConfig()
            
        self.config = config
        self.model_path = model_path
        self.model: Optional[lgb.LGBMClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        self.feature_history = []
        self.performance_history = []
        
        # Model parameters from config
        self.lookback_period = config.lookback_period
        self.retrain_interval = config.retrain_interval
        self.prediction_count = 0
        
        # Load existing model if available
        self._load_model()
    
    def get_allocation(self, market_data: MarketData, 
                      mean_reversion_performance: float,
                      momentum_performance: float) -> AllocationDecision:
        """Get capital allocation decision based on market conditions."""
        # Extract features from market data
        features = self._extract_features(market_data)
        
        if not self.is_trained:
            # Default allocation if model not trained - equal weights across all strategies
            return AllocationDecision(
                mean_reversion_weight=0.25,
                momentum_weight=0.25,
                vol_carry_weight=0.25,
                vol_breakout_weight=0.25,
                confidence=0.5,
                regime="unknown",
                features=features
            )
        
        # Make prediction
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        if self.scaler:
            feature_array = self.scaler.transform(feature_array)
        
        # Get probability predictions
        probabilities = self.model.predict_proba(feature_array)[0]
        prediction = self.model.predict(feature_array)[0]
        
        # Convert prediction to allocation weights
        allocation = self._prediction_to_allocation(
            prediction, probabilities, features
        )
        
        # Store for retraining
        self._store_prediction_data(features, allocation, 
                                   mean_reversion_performance, momentum_performance)
        
        self.prediction_count += 1
        
        # Retrain if needed
        if self.prediction_count % self.retrain_interval == 0:
            self._retrain_model()
        
        return allocation
    
    def _extract_features(self, market_data: MarketData) -> Dict[str, float]:
        """Extract features for ML model."""
        features = {}
        
        # Initialize volatility features with defaults
        features['volatility_realized'] = market_data.volatility if market_data.volatility is not None else 0.0
        features['volatility_1h'] = 0.0
        
        # Volatility regime features for strategy selection
        features['volatility_regime_low'] = 1.0 if market_data.volatility_regime == 'low' else 0.0
        features['volatility_regime_medium'] = 1.0 if market_data.volatility_regime == 'medium' else 0.0
        features['volatility_regime_high'] = 1.0 if market_data.volatility_regime == 'high' else 0.0
        features['volatility_percentile'] = market_data.volatility_percentile
        
        # Volatility breakout features
        features['volatility_breakout_detected'] = 0.0
        features['volatility_breakout_strength'] = 0.0
        features['volatility_breakout_direction'] = 0.0  # -1=down, 0=none, 1=up
        
        if market_data.volatility_breakout:
            features['volatility_breakout_detected'] = 1.0 if market_data.volatility_breakout.get('is_breakout', False) else 0.0
            features['volatility_breakout_strength'] = market_data.volatility_breakout.get('breakout_strength', 0.0)
            breakout_dir = market_data.volatility_breakout.get('breakout_direction', 'none')
            features['volatility_breakout_direction'] = 1.0 if breakout_dir == 'up' else -1.0 if breakout_dir == 'down' else 0.0
        
        # Multi-timeframe volatility features
        features['volatility_1m'] = market_data.volatility_1m
        features['volatility_5m'] = market_data.volatility_5m
        features['volatility_15m'] = market_data.volatility_15m
        features['volatility_30m'] = market_data.volatility_30m
        
        # Volatility term structure (carry opportunities)
        features['vol_term_structure_slope'] = 0.0
        features['vol_carry_opportunity'] = 0.0
        
        # Calculate volatility term structure slope
        if market_data.volatility_5m > 0 and market_data.volatility_1h > 0:
            features['vol_term_structure_slope'] = (market_data.volatility_1h - market_data.volatility_5m) / market_data.volatility_5m
            # Carry opportunity strength
            features['vol_carry_opportunity'] = min(1.0, abs(features['vol_term_structure_slope']) * 5.0)
        
        # Price-based features
        if market_data.price_1h and len(market_data.price_1h) >= 20:
            prices_1h = market_data.price_1h[-20:]
            
            # Volatility features
            returns = [prices_1h[i] / prices_1h[i-1] - 1 for i in range(1, len(prices_1h))]
            features['volatility_1h'] = np.std(returns) if returns else 0.0
            
            # Trend features
            features['price_momentum_1h'] = (prices_1h[-1] - prices_1h[0]) / prices_1h[0]
            features['price_momentum_5h'] = (prices_1h[-1] - prices_1h[-5]) / prices_1h[-5] if len(prices_1h) >= 5 else 0.0
            
            # Mean reversion features
            mean_price = np.mean(prices_1h)
            features['deviation_from_mean'] = (prices_1h[-1] - mean_price) / mean_price
            
            # Regime features
            features['regime_trend_strength'] = self._calculate_trend_strength(prices_1h)
            features['regime_mean_reversion'] = self._calculate_mean_reversion_strength(prices_1h)
        
        # Volume features
        if market_data.volume_1h and len(market_data.volume_1h) >= 10:
            volumes_1h = market_data.volume_1h[-10:]
            features['volume_trend'] = (volumes_1h[-1] - volumes_1h[0]) / volumes_1h[0]
            features['volume_volatility'] = np.std(volumes_1h) / np.mean(volumes_1h) if np.mean(volumes_1h) > 0 else 0.0
        
        # Account and risk features
        features['account_balance'] = market_data.account_balance
        features['daily_pnl_pct'] = market_data.daily_pnl / market_data.account_balance if market_data.account_balance > 0 else 0.0
        features['position_size'] = abs(market_data.open_positions)
        features['unrealized_pnl_pct'] = market_data.unrealized_pnl / market_data.account_balance if market_data.account_balance > 0 else 0.0
        
        # Time-based features
        now = datetime.now()
        features['hour_of_day'] = now.hour
        features['day_of_week'] = now.weekday()
        features['is_market_open'] = 1.0 if not (16 <= now.hour < 17) else 0.0
        
        # Market microstructure features
        volatility_value = market_data.volatility if market_data.volatility is not None else 0.0
        features['bid_ask_spread_proxy'] = volatility_value * 0.1  # Approximate
        features['market_impact_proxy'] = features['position_size'] * features['volatility_realized']
        
        return features
    
    def _calculate_trend_strength(self, prices: list) -> float:
        """Calculate trend strength indicator."""
        if len(prices) < 10:
            return 0.0
        
        # Calculate moving averages
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-10:])
        
        # Trend strength based on MA separation
        trend_strength = abs(short_ma - long_ma) / long_ma if long_ma > 0 else 0.0
        
        return min(1.0, trend_strength * 10)  # Normalize
    
    def _calculate_mean_reversion_strength(self, prices: list) -> float:
        """Calculate mean reversion strength indicator."""
        if len(prices) < 10:
            return 0.0
        
        # Calculate deviations from mean
        mean_price = np.mean(prices)
        deviations = [(p - mean_price) / mean_price for p in prices]
        
        # Mean reversion strength based on oscillation
        oscillation = np.std(deviations)
        
        return min(1.0, oscillation * 20)  # Normalize
    
    def _prediction_to_allocation(self, prediction: int, probabilities: np.ndarray, 
                                features: Dict[str, float]) -> AllocationDecision:
        """Convert ML prediction to allocation decision."""
        # Classes: 0=mean_reversion, 1=momentum, 2=vol_carry, 3=vol_breakout, 4=balanced
        confidence = max(probabilities)
        
        # Base allocations for each strategy type
        if prediction == 0:  # Mean reversion favored
            mean_reversion_weight = 0.5
            momentum_weight = 0.2
            vol_carry_weight = 0.2
            vol_breakout_weight = 0.1
            regime = "mean_reverting"
        elif prediction == 1:  # Momentum favored
            mean_reversion_weight = 0.2
            momentum_weight = 0.5
            vol_carry_weight = 0.1
            vol_breakout_weight = 0.2
            regime = "trending"
        elif prediction == 2:  # Vol carry favored
            mean_reversion_weight = 0.2
            momentum_weight = 0.1
            vol_carry_weight = 0.5
            vol_breakout_weight = 0.2
            regime = "vol_carry"
        elif prediction == 3:  # Vol breakout favored
            mean_reversion_weight = 0.1
            momentum_weight = 0.2
            vol_carry_weight = 0.2
            vol_breakout_weight = 0.5
            regime = "vol_breakout"
        else:  # Balanced (prediction == 4)
            mean_reversion_weight = 0.25
            momentum_weight = 0.25
            vol_carry_weight = 0.25
            vol_breakout_weight = 0.25
            regime = "balanced"
        
        # Adjust weights based on market conditions and features
        self._adjust_weights_based_on_features(features, locals())
        
        # Adjust weights based on confidence
        if confidence < 0.6:
            # Low confidence -> more balanced allocation
            target_balanced = 0.25
            mean_reversion_weight = target_balanced + (mean_reversion_weight - target_balanced) * 0.3
            momentum_weight = target_balanced + (momentum_weight - target_balanced) * 0.3
            vol_carry_weight = target_balanced + (vol_carry_weight - target_balanced) * 0.3
            vol_breakout_weight = target_balanced + (vol_breakout_weight - target_balanced) * 0.3
        
        # Ensure weights sum to 1
        total_weight = mean_reversion_weight + momentum_weight + vol_carry_weight + vol_breakout_weight
        mean_reversion_weight /= total_weight
        momentum_weight /= total_weight
        vol_carry_weight /= total_weight
        vol_breakout_weight /= total_weight
        
        return AllocationDecision(
            mean_reversion_weight=mean_reversion_weight,
            momentum_weight=momentum_weight,
            vol_carry_weight=vol_carry_weight,
            vol_breakout_weight=vol_breakout_weight,
            confidence=confidence,
            regime=regime,
            features=features
        )
    
    def _adjust_weights_based_on_features(self, features: Dict[str, float], weights: Dict[str, float]):
        """Adjust strategy weights based on current market features."""
        # Increase vol carry weight when strong term structure slope is detected
        if features.get('vol_carry_opportunity', 0) > 0.5:
            weights['vol_carry_weight'] *= 1.3
        
        # Increase vol breakout weight when volatility breakout is detected
        if features.get('volatility_breakout_detected', 0) > 0:
            weights['vol_breakout_weight'] *= 1.5
        
        # Reduce vol strategies in extreme volatility regimes
        if features.get('volatility_regime_high', 0) > 0:
            weights['vol_carry_weight'] *= 0.8  # Reduce carry in high vol
        
        if features.get('volatility_regime_low', 0) > 0:
            weights['vol_breakout_weight'] *= 0.7  # Reduce breakout in low vol
        
        # Increase momentum weight when strong trend detected
        if features.get('regime_trend_strength', 0) > 0.7:
            weights['momentum_weight'] *= 1.2
        
        # Increase mean reversion weight when high mean reversion strength
        if features.get('regime_mean_reversion', 0) > 0.7:
            weights['mean_reversion_weight'] *= 1.2
    
    def _store_prediction_data(self, features: Dict[str, float], 
                             allocation: AllocationDecision,
                             mean_reversion_perf: float, 
                             momentum_perf: float,
                             vol_carry_perf: float = 0.0,
                             vol_breakout_perf: float = 0.0):
        """Store prediction data for model retraining."""
        # Determine actual best strategy (target)
        performances = {
            0: mean_reversion_perf,    # Mean reversion
            1: momentum_perf,          # Momentum
            2: vol_carry_perf,         # Vol carry
            3: vol_breakout_perf,      # Vol breakout
        }
        
        # Find best performing strategy
        best_strategy = max(performances, key=performances.get)
        best_perf = performances[best_strategy]
        
        # Check if performance is significantly better than others
        other_perfs = [p for i, p in performances.items() if i != best_strategy]
        avg_other_perf = sum(other_perfs) / len(other_perfs) if other_perfs else 0
        
        if best_perf > avg_other_perf * 1.1:
            target = best_strategy
        else:
            target = 4  # Balanced
        
        # Store feature vector and target
        self.feature_history.append(features)
        self.performance_history.append(target)
        
        # Keep only recent history
        if len(self.feature_history) > self.config.feature_history_size:
            keep_size = int(self.config.feature_history_size * 0.6)  # Keep 60% when trimming
            self.feature_history = self.feature_history[-keep_size:]
            self.performance_history = self.performance_history[-keep_size:]
    
    def train_model(self, feature_data: list = None, targets: list = None):
        """Train the ML model."""
        # Use provided data or stored history
        if feature_data is None:
            feature_data = self.feature_history
        if targets is None:
            targets = self.performance_history
        
        if len(feature_data) < 100:
            print("Insufficient data for training")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_data)
        y = np.array(targets)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train LightGBM model for 5 classes (0-4)
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            num_class=5,  # 5 classes: mean_reversion, momentum, vol_carry, vol_breakout, balanced
            random_state=42,
            verbose=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Model trained - Train score: {train_score:.3f}, Test score: {test_score:.3f}")
        
        self.is_trained = True
        
        # Save model
        self._save_model()
    
    def _retrain_model(self):
        """Retrain model with recent data."""
        if len(self.feature_history) >= 200:
            print("Retraining meta-allocator model...")
            self.train_model()
    
    def _save_model(self):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_history': self.feature_history[-1000:],  # Keep recent history
            'performance_history': self.performance_history[-1000:]
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {self.model_path}")
    
    def _load_model(self):
        """Load trained model from disk."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.is_trained = model_data['is_trained']
                self.feature_history = model_data.get('feature_history', [])
                self.performance_history = model_data.get('performance_history', [])
                
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.is_trained = False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        return {
            'is_trained': self.is_trained,
            'prediction_count': self.prediction_count,
            'feature_history_length': len(self.feature_history),
            'performance_history_length': len(self.performance_history),
            'model_path': self.model_path,
            'lookback_period': self.lookback_period,
            'retrain_interval': self.retrain_interval
        }