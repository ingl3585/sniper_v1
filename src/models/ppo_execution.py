"""
PPO RL Agent for Order Execution
Reinforcement learning agent for order type selection (limit vs market).
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces
import pickle
import os
from datetime import datetime
from src.infra.nt_bridge import MarketData, TradeSignal


@dataclass
class ExecutionDecision:
    """Execution decision from RL agent."""
    order_type: str  # 'market' or 'limit'
    limit_offset: float  # Offset from current price for limit orders
    confidence: float
    expected_slippage: float
    urgency_score: float


class OrderExecutionEnv(gym.Env):
    """Gym environment for order execution RL training."""
    
    def __init__(self):
        super(OrderExecutionEnv, self).__init__()
        
        # Action space: [order_type, limit_offset]
        # order_type: 0=market, 1=limit
        # limit_offset: -0.01 to 0.01 (as percentage of price)
        self.action_space = spaces.Box(
            low=np.array([0.0, -0.01], dtype=np.float32), 
            high=np.array([1.0, 0.01], dtype=np.float32), 
            dtype=np.float32
        )
        
        # Observation space: market conditions and order context
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(15,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset environment for new episode."""
        self.current_price = 100.0
        self.volatility = 0.02
        self.volume = 1000.0
        self.spread = 0.01
        self.position_size = 1
        self.urgency = 0.5
        self.market_impact = 0.001
        self.trend_strength = 0.0
        self.order_book_depth = 10
        self.recent_volatility = 0.02
        self.time_decay = 0.0
        self.market_stress = 0.0
        self.liquidity_score = 0.8
        self.price_momentum = 0.0
        self.volume_profile = 0.5
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return new state, reward, done, info."""
        order_type_raw = action[0]
        limit_offset = action[1]
        
        # Convert continuous action to discrete order type
        order_type = 0 if order_type_raw < 0.5 else 1  # 0=market, 1=limit
        
        # Calculate execution cost and slippage
        execution_cost, fill_probability = self._calculate_execution_metrics(
            order_type, limit_offset
        )
        
        # Calculate reward
        reward = self._calculate_reward(execution_cost, fill_probability)
        
        # Update environment state (simplified)
        self._update_market_state()
        
        # Episode ends after one trade
        done = True
        
        info = {
            'execution_cost': execution_cost,
            'fill_probability': fill_probability,
            'order_type': 'market' if order_type == 0 else 'limit',
            'limit_offset': limit_offset
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get current observation vector."""
        return np.array([
            self.current_price / 100.0,  # Normalized price
            self.volatility,
            self.volume / 1000.0,
            self.spread,
            self.position_size / 10.0,
            self.urgency,
            self.market_impact,
            self.trend_strength,
            self.order_book_depth / 50.0,
            self.recent_volatility,
            self.time_decay,
            self.market_stress,
            self.liquidity_score,
            self.price_momentum,
            self.volume_profile
        ], dtype=np.float32)
    
    def _calculate_execution_metrics(self, order_type: int, limit_offset: float) -> Tuple[float, float]:
        """Calculate execution cost and fill probability."""
        if order_type == 0:  # Market order
            # Market orders have guaranteed fill but higher cost
            execution_cost = self.spread / 2 + self.market_impact * self.position_size
            fill_probability = 1.0
        else:  # Limit order
            # Limit orders have lower cost but uncertain fill
            execution_cost = max(0, abs(limit_offset) - self.spread / 2)
            
            # Fill probability depends on offset and market conditions
            if limit_offset == 0:
                fill_probability = 0.5
            else:
                # Better price = lower fill probability
                distance_factor = abs(limit_offset) / self.volatility
                fill_probability = max(0.1, 1.0 - distance_factor)
                
                # Adjust for market conditions
                fill_probability *= (1 - self.market_stress)
                fill_probability *= self.liquidity_score
        
        return execution_cost, fill_probability
    
    def _calculate_reward(self, execution_cost: float, fill_probability: float) -> float:
        """Calculate reward for the action."""
        # Reward function balances cost and fill probability
        cost_penalty = -execution_cost * 1000  # Scale up for learning
        fill_reward = fill_probability * 100
        
        # Urgency bonus for fast execution
        urgency_bonus = self.urgency * 20 if fill_probability > 0.8 else 0
        
        # Volatility penalty for risky periods
        volatility_penalty = -self.volatility * 50 if execution_cost > 0.01 else 0
        
        total_reward = cost_penalty + fill_reward + urgency_bonus + volatility_penalty
        
        return total_reward
    
    def _update_market_state(self):
        """Update market state for next observation."""
        # Simple market evolution (in real implementation, this would be driven by actual market data)
        self.volatility += np.random.normal(0, 0.001)
        self.volatility = np.clip(self.volatility, 0.005, 0.1)
        
        self.volume += np.random.normal(0, 100)
        self.volume = max(100, self.volume)
        
        self.spread += np.random.normal(0, 0.001)
        self.spread = np.clip(self.spread, 0.001, 0.05)
    
    def set_market_conditions(self, market_data: MarketData, position_size: int, urgency: float):
        """Set market conditions from real market data."""
        self.current_price = market_data.current_price
        self.volatility = market_data.volatility
        self.volume = market_data.volume_1m[-1] if market_data.volume_1m else 1000.0
        self.position_size = position_size
        self.urgency = urgency
        
        # Estimate market microstructure parameters
        self.spread = self.volatility * 0.1  # Approximate spread
        self.market_impact = self.volatility * 0.05 * position_size
        self.liquidity_score = min(1.0, self.volume / 5000.0)  # Normalized liquidity
        
        # Calculate recent volatility
        if len(market_data.price_1m) >= 10:
            prices = market_data.price_1m[-10:]
            returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
            self.recent_volatility = np.std(returns)
        
        # Calculate momentum
        if len(market_data.price_1m) >= 5:
            prices = market_data.price_1m[-5:]
            self.price_momentum = (prices[-1] - prices[0]) / prices[0]


class PPOExecutionAgent:
    """PPO RL agent for order execution optimization."""
    
    def __init__(self, model_path: str = "data/ppo_execution_model.zip"):
        self.model_path = model_path
        self.model: Optional[PPO] = None
        self.env = OrderExecutionEnv()
        self.is_trained = False
        self.training_episodes = 0
        
        # Load existing model if available
        self._load_model()
    
    def get_execution_decision(self, market_data: MarketData, 
                            position_size: int, urgency: float = 0.5) -> ExecutionDecision:
        """Get execution decision from RL agent."""
        # Set market conditions in environment
        self.env.set_market_conditions(market_data, position_size, urgency)
        
        if not self.is_trained:
            # Default decision if model not trained
            return ExecutionDecision(
                order_type='market',
                limit_offset=0.0,
                confidence=0.5,
                expected_slippage=market_data.volatility * 0.5,
                urgency_score=urgency
            )
        
        # Get observation
        obs = self.env._get_observation()
        
        # Predict action
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Convert action to execution decision
        order_type_raw = action[0]
        limit_offset = action[1]
        
        order_type = 'market' if order_type_raw < 0.5 else 'limit'
        
        # Calculate confidence based on action probability
        confidence = 0.7  # Simplified confidence
        
        # Estimate expected slippage
        if order_type == 'market':
            expected_slippage = market_data.volatility * 0.5
        else:
            expected_slippage = max(0, abs(limit_offset) - market_data.volatility * 0.1)
        
        return ExecutionDecision(
            order_type=order_type,
            limit_offset=limit_offset,
            confidence=confidence,
            expected_slippage=expected_slippage,
            urgency_score=urgency
        )
    
    def train_agent(self, total_timesteps: int = 10000):
        """Train the PPO agent."""
        print(f"Training PPO execution agent for {total_timesteps} timesteps...")
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
        
        # Train model
        self.model.learn(total_timesteps=total_timesteps)
        
        self.is_trained = True
        self.training_episodes += total_timesteps
        
        # Save model
        self._save_model()
        
        print(f"PPO agent training completed. Total episodes: {self.training_episodes}")
    
    def update_from_execution(self, execution_decision: ExecutionDecision, 
                            actual_cost: float, fill_success: bool):
        """Update agent from actual execution results."""
        # In a full implementation, this would store experience for replay
        # For now, we'll just track performance
        pass
    
    def _save_model(self):
        """Save trained model to disk."""
        if self.model is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            
            # Save metadata
            metadata_path = self.model_path.replace('.zip', '_metadata.pkl')
            metadata = {
                'is_trained': self.is_trained,
                'training_episodes': self.training_episodes,
                'model_path': self.model_path
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"PPO model saved to {self.model_path}")
    
    def _load_model(self):
        """Load trained model from disk."""
        if os.path.exists(self.model_path):
            try:
                self.model = PPO.load(self.model_path)
                self.is_trained = True
                
                # Load metadata
                metadata_path = self.model_path.replace('.zip', '_metadata.pkl')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    self.training_episodes = metadata.get('training_episodes', 0)
                
                print(f"PPO model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading PPO model: {e}")
                self.is_trained = False
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information and statistics."""
        return {
            'is_trained': self.is_trained,
            'training_episodes': self.training_episodes,
            'model_path': self.model_path,
            'model_loaded': self.model is not None
        }
    
    def evaluate_agent(self, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate agent performance."""
        if not self.is_trained:
            return {'error': 'Agent not trained'}
        
        total_reward = 0
        successful_fills = 0
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            
            total_reward += reward
            if info['fill_probability'] > 0.8:
                successful_fills += 1
        
        avg_reward = total_reward / num_episodes
        fill_rate = successful_fills / num_episodes
        
        return {
            'average_reward': avg_reward,
            'fill_rate': fill_rate,
            'episodes_evaluated': num_episodes
        }