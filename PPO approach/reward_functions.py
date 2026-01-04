"""
Reward Functions
================
Advanced reward shaping for the PPO trading agent.

Reward components:
1. Profit Reward: Normalized profit from trades
2. Transaction Cost Penalty: -0.25% per trade (Bitvavo fee)
3. Drawdown Penalty: Penalize large drawdowns
4. Sharpe Component: Reward consistent returns
5. Hold Penalty (Optional): Configurable via config
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    # Profit scaling
    profit_scale: float = 100.0
    
    # Transaction cost handling
    cost_scale: float = 1.0
    transaction_cost: float = 0.0025  # 0.25%
    
    # Risk penalties
    drawdown_penalty: float = 0.1
    max_drawdown_threshold: float = 0.1  # 10%
    
    # Sharpe bonus
    sharpe_bonus: float = 0.5
    min_periods_for_sharpe: int = 10
    
    # Hold penalty (optional)
    enable_hold_penalty: bool = False
    hold_penalty: float = 0.01
    max_hold_periods: int = 100
    
    # Position penalties
    invalid_action_penalty: float = -1.0
    
    # Reward clipping
    clip_reward: bool = True
    reward_clip_value: float = 10.0
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'RewardConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if hasattr(cls, k)})


class RewardCalculator:
    """Calculate rewards for the trading environment."""
    
    def __init__(self, config: RewardConfig = None):
        """
        Initialize reward calculator.
        
        Args:
            config: Reward configuration
        """
        self.config = config or RewardConfig()
        self.returns_history: List[float] = []
        self.equity_history: List[float] = []
        self.peak_equity: float = 0.0
        self.last_reward_components: Dict[str, float] = {}
        self.reward_history: List[float] = []
        self.component_history: List[Dict[str, float]] = []
    
    def reset(self, initial_equity: float = 10000.0):
        """Reset state for new episode."""
        self.returns_history = []
        self.equity_history = [initial_equity]
        self.peak_equity = initial_equity
        self.reward_history = []
        self.component_history = []
        self.last_reward_components = {}
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """
        Get statistics about reward distribution and components.
        
        Returns:
            Dictionary with reward statistics
        """
        if not self.reward_history:
            return {}
        
        rewards = np.array(self.reward_history)
        stats = {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'median': float(np.median(rewards)),
        }
        
        # Component statistics if available
        if self.component_history:
            components = {}
            for comp_name in ['profit', 'cost', 'drawdown', 'sharpe', 'hold']:
                if self.component_history and comp_name in self.component_history[0]:
                    comp_values = [c.get(comp_name, 0) for c in self.component_history]
                    if comp_values:
                        components[f'{comp_name}_mean'] = float(np.mean(comp_values))
                        components[f'{comp_name}_std'] = float(np.std(comp_values))
            if components:
                stats['components'] = components
        
        return stats
    
    def calculate_reward(
        self,
        profit_pct: float,
        transaction_cost: float,
        current_equity: float,
        holding_time: int,
        action_valid: bool = True,
        position_changed: bool = False,
        log_components: bool = False,
    ) -> float:
        """
        Calculate reward for current step.
        
        Args:
            profit_pct: Profit/loss percentage for this step
            transaction_cost: Transaction cost incurred
            current_equity: Current portfolio equity
            holding_time: How long current position has been held
            action_valid: Whether the action was valid
            position_changed: Whether position was opened/closed
            log_components: Whether to log reward components for diagnostics
            
        Returns:
            Calculated reward
        """
        reward = 0.0
        components = {}
        
        # Invalid action penalty
        if not action_valid:
            if log_components:
                components['invalid_action'] = self.config.invalid_action_penalty
            return self.config.invalid_action_penalty
        
        # 1. Profit reward
        profit_reward = profit_pct * self.config.profit_scale
        reward += profit_reward
        components['profit'] = profit_reward
        
        # 2. Transaction cost penalty
        if position_changed:
            cost_penalty = transaction_cost * self.config.cost_scale
            reward -= cost_penalty
            components['cost'] = -cost_penalty
            # Positive reward for taking action (encourages trading activity)
            # This helps prevent the agent from learning to only hold
            # Increased to 0.25 to offset typical transaction costs (0.25% of position)
            # For a full position (100% of capital), transaction cost is ~0.25, so bonus should match
            action_bonus = 0.25  # Bonus for executing a trade (offsets transaction costs, encourages exploration)
            reward += action_bonus
            components['action_bonus'] = action_bonus
        else:
            components['cost'] = 0.0
            components['action_bonus'] = 0.0
        
        # 3. Drawdown penalty
        drawdown_penalty = self._calculate_drawdown_penalty(current_equity)
        reward -= drawdown_penalty
        components['drawdown'] = -drawdown_penalty
        
        # 4. Sharpe component (if enough history)
        sharpe_bonus = 0.0
        if len(self.returns_history) >= self.config.min_periods_for_sharpe:
            sharpe_bonus = self._calculate_sharpe_bonus()
            reward += sharpe_bonus
        components['sharpe'] = sharpe_bonus
        
        # 5. Hold penalty (optional)
        hold_penalty = 0.0
        if self.config.enable_hold_penalty and holding_time > 0:
            hold_penalty = self._calculate_hold_penalty(holding_time)
            reward -= hold_penalty
        components['hold'] = -hold_penalty
        
        # Update history
        self._update_history(profit_pct, current_equity)
        
        # Clip reward if configured
        original_reward = reward
        if self.config.clip_reward:
            reward = np.clip(reward, -self.config.reward_clip_value, self.config.reward_clip_value)
            if reward != original_reward:
                components['clipped'] = reward - original_reward
        
        components['total'] = reward
        
        # Store components for diagnostics
        if log_components:
            self.last_reward_components = components
            self.reward_history.append(reward)
        
        return reward
    
    def _calculate_drawdown_penalty(self, current_equity: float) -> float:
        """Calculate penalty for drawdown from peak equity."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            return 0.0
        
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        if drawdown > self.config.max_drawdown_threshold:
            # Larger penalty for exceeding threshold
            penalty = drawdown * self.config.drawdown_penalty * 2
        else:
            penalty = drawdown * self.config.drawdown_penalty
        
        return penalty
    
    def _calculate_sharpe_bonus(self) -> float:
        """Calculate bonus based on Sharpe-like metric."""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history[-self.config.min_periods_for_sharpe:])
        
        if np.std(returns) == 0:
            return 0.0 if np.mean(returns) <= 0 else self.config.sharpe_bonus
        
        sharpe = np.mean(returns) / np.std(returns)
        
        # Scale and clip
        bonus = sharpe * self.config.sharpe_bonus
        return np.clip(bonus, -self.config.sharpe_bonus, self.config.sharpe_bonus * 2)
    
    def _calculate_hold_penalty(self, holding_time: int) -> float:
        """Calculate penalty for holding position too long."""
        if holding_time <= 0:
            return 0.0
        
        # Penalty increases with holding time
        ratio = min(holding_time / self.config.max_hold_periods, 1.0)
        return ratio * self.config.hold_penalty
    
    def _update_history(self, profit_pct: float, current_equity: float):
        """Update internal history."""
        self.returns_history.append(profit_pct)
        self.equity_history.append(current_equity)
        
        # Store reward components for diagnostics (if available)
        if hasattr(self, 'last_reward_components') and self.last_reward_components:
            self.component_history.append(self.last_reward_components.copy())
        
        # Keep history bounded
        max_history = 1000
        if len(self.returns_history) > max_history:
            self.returns_history = self.returns_history[-max_history:]
            self.equity_history = self.equity_history[-max_history:]


def calculate_step_reward(
    profit_pct: float,
    transaction_cost: float = 0.0,
    drawdown: float = 0.0,
    returns_history: List[float] = None,
    holding_time: int = 0,
    config: Dict = None
) -> float:
    """
    Simple function to calculate reward (matches plan specification).
    
    Args:
        profit_pct: Profit/loss as decimal (e.g., 0.01 = 1%)
        transaction_cost: Cost of transaction
        drawdown: Current drawdown from peak
        returns_history: List of recent returns
        holding_time: Steps holding current position
        config: Configuration dictionary
        
    Returns:
        Calculated reward
    """
    config = config or {}
    
    reward = profit_pct * config.get('profit_scale', 100)
    reward -= transaction_cost * config.get('cost_scale', 1.0)
    reward -= drawdown * config.get('drawdown_penalty', 0.1)
    
    if returns_history and len(returns_history) > 1:
        sharpe = np.mean(returns_history) / (np.std(returns_history) + 1e-8)
        reward += sharpe * config.get('sharpe_bonus', 0.5)
    
    # Optional hold penalty (disabled by default)
    if config.get('enable_hold_penalty', False):
        max_hold = config.get('max_hold_periods', 100)
        reward -= (holding_time / max_hold) * config.get('hold_penalty', 0.01)
    
    return reward


class DifferentialSharpeReward:
    """
    Differential Sharpe Ratio reward.
    
    Based on: "Reinforcement Learning for Trading" (Moody & Saffell)
    This provides a reward signal that directly optimizes Sharpe ratio.
    """
    
    def __init__(self, eta: float = 0.001):
        """
        Initialize.
        
        Args:
            eta: Adaptation rate for running statistics
        """
        self.eta = eta
        self.A = 0.0  # Running mean return
        self.B = 0.0  # Running mean squared return
        self.count = 0
    
    def reset(self):
        """Reset statistics."""
        self.A = 0.0
        self.B = 0.0
        self.count = 0
    
    def calculate(self, return_t: float) -> float:
        """
        Calculate differential Sharpe ratio.
        
        Args:
            return_t: Return at time t
            
        Returns:
            Differential Sharpe ratio (reward)
        """
        self.count += 1
        
        if self.count == 1:
            self.A = return_t
            self.B = return_t ** 2
            return 0.0
        
        # Calculate differential
        delta_A = return_t - self.A
        delta_B = return_t ** 2 - self.B
        
        denominator = (self.B - self.A ** 2) ** 1.5
        
        if abs(denominator) < 1e-10:
            dsr = 0.0
        else:
            numerator = self.B * delta_A - 0.5 * self.A * delta_B
            dsr = numerator / denominator
        
        # Update running statistics
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B
        
        return dsr


class ProfitAndLossReward:
    """Simple P&L based reward with optional shaping."""
    
    def __init__(
        self,
        transaction_cost: float = 0.0025,
        reward_scale: float = 100.0,
        penalize_inaction: bool = False,
        inaction_penalty: float = 0.001
    ):
        self.transaction_cost = transaction_cost
        self.reward_scale = reward_scale
        self.penalize_inaction = penalize_inaction
        self.inaction_penalty = inaction_penalty
    
    def calculate(
        self,
        pnl: float,
        traded: bool = False,
        is_holding: bool = False
    ) -> float:
        """
        Calculate reward based on P&L.
        
        Args:
            pnl: Profit/loss for this step
            traded: Whether a trade was executed
            is_holding: Whether holding a position
            
        Returns:
            Reward value
        """
        reward = pnl * self.reward_scale
        
        if traded:
            reward -= self.transaction_cost * self.reward_scale
        
        if self.penalize_inaction and not is_holding and not traded:
            reward -= self.inaction_penalty
        
        return reward


def create_reward_calculator(config: Dict = None) -> RewardCalculator:
    """
    Factory function to create reward calculator from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured RewardCalculator
    """
    if config is None:
        return RewardCalculator()
    
    reward_config = RewardConfig.from_dict(config)
    return RewardCalculator(reward_config)


if __name__ == '__main__':
    # Test reward functions
    print("Testing reward_functions.py...")
    print()
    
    # Test RewardCalculator
    config = RewardConfig(
        profit_scale=100,
        drawdown_penalty=0.1,
        sharpe_bonus=0.5,
        enable_hold_penalty=True,
        hold_penalty=0.01
    )
    
    calculator = RewardCalculator(config)
    calculator.reset(initial_equity=10000)
    
    # Simulate some steps
    scenarios = [
        {'profit_pct': 0.01, 'transaction_cost': 0.0025, 'equity': 10100, 'hold': 0, 'changed': True},
        {'profit_pct': 0.005, 'transaction_cost': 0.0, 'equity': 10150, 'hold': 1, 'changed': False},
        {'profit_pct': -0.02, 'transaction_cost': 0.0, 'equity': 9950, 'hold': 2, 'changed': False},
        {'profit_pct': 0.015, 'transaction_cost': 0.0025, 'equity': 10100, 'hold': 0, 'changed': True},
    ]
    
    print("Scenario rewards:")
    for i, s in enumerate(scenarios):
        reward = calculator.calculate_reward(
            profit_pct=s['profit_pct'],
            transaction_cost=s['transaction_cost'],
            current_equity=s['equity'],
            holding_time=s['hold'],
            position_changed=s['changed']
        )
        print(f"  Step {i+1}: profit={s['profit_pct']:.3f}, reward={reward:.4f}")
    
    print()
    
    # Test simple function
    reward = calculate_step_reward(
        profit_pct=0.02,
        transaction_cost=0.0025,
        drawdown=0.05,
        returns_history=[0.01, -0.005, 0.015, 0.02],
        holding_time=5,
        config={'profit_scale': 100, 'enable_hold_penalty': True}
    )
    print(f"Simple function reward: {reward:.4f}")
    
    print()
    
    # Test Differential Sharpe
    dsr = DifferentialSharpeReward(eta=0.01)
    returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.025]
    
    print("Differential Sharpe rewards:")
    for r in returns:
        reward = dsr.calculate(r)
        print(f"  Return={r:.3f}, DSR={reward:.6f}")


