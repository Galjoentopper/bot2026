"""
Utility Functions
=================
Helper functions for portfolio management, P&L calculation, and risk metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class Trade:
    """Represents a single trade."""
    trade_id: int
    trade_type: str  # 'long' or 'short'
    entry_time: int
    entry_price: float
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    size: float = 1.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    transaction_cost: float = 0.0


@dataclass 
class PortfolioState:
    """Represents the current portfolio state."""
    cash: float
    position: float  # Positive = long, negative = short, 0 = no position
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    total_equity: float = 0.0
    
    def update_equity(self, current_price: float):
        """Update unrealized PnL and total equity based on current price."""
        if self.position != 0 and self.entry_price > 0:
            # Calculate price change percentage
            if self.position > 0:  # Long position
                price_change_pct = (current_price - self.entry_price) / self.entry_price
            else:  # Short position
                price_change_pct = (self.entry_price - current_price) / self.entry_price
            
            # Unrealized PnL is the dollar amount: position_value * price_change_pct
            position_value = abs(self.position)
            self.unrealized_pnl = position_value * price_change_pct
        else:
            self.unrealized_pnl = 0.0
        
        # Total equity = cash + position_value + unrealized_pnl
        # Position value is already in cash (was deducted when opened), so we add unrealized PnL
        self.total_equity = self.cash + abs(self.position) + self.unrealized_pnl


class PortfolioTracker:
    """Track portfolio state, positions, and trades."""
    
    def __init__(self, initial_capital: float = 10000.0, transaction_cost: float = 0.0025):
        """
        Initialize portfolio tracker.
        
        Args:
            initial_capital: Starting cash amount
            transaction_cost: Cost per trade as decimal (0.0025 = 0.25%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        self.state = PortfolioState(
            cash=initial_capital,
            position=0.0,
            total_equity=initial_capital
        )
        
        self.trades: List[Trade] = []
        self.equity_history: List[float] = [initial_capital]
        self.returns_history: List[float] = []
        self.trade_counter = 0
        self.current_trade: Optional[Trade] = None
        self.holding_time = 0
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.state = PortfolioState(
            cash=self.initial_capital,
            position=0.0,
            total_equity=self.initial_capital
        )
        self.trades = []
        self.equity_history = [self.initial_capital]
        self.returns_history = []
        self.trade_counter = 0
        self.current_trade = None
        self.holding_time = 0
    
    def open_position(self, price: float, size: float, position_type: str, 
                      timestep: int) -> float:
        """
        Open a new position.
        
        Args:
            price: Entry price
            size: Position size (fraction of cash to use, 0-1)
            position_type: 'long' or 'short'
            timestep: Current timestep
            
        Returns:
            Transaction cost incurred
        """
        if self.state.position != 0:
            # Already have a position - close it first
            self.close_position(price, timestep)
        
        # Calculate position size in terms of capital
        position_value = self.state.cash * size
        cost = position_value * self.transaction_cost
        
        # Update state
        self.state.cash -= cost
        self.state.position = position_value if position_type == 'long' else -position_value
        self.state.entry_price = price
        self.holding_time = 0
        
        # Create trade record
        self.trade_counter += 1
        self.current_trade = Trade(
            trade_id=self.trade_counter,
            trade_type=position_type,
            entry_time=timestep,
            entry_price=price,
            size=abs(self.state.position),
            transaction_cost=cost
        )
        
        return cost
    
    def close_position(self, price: float, timestep: int) -> Tuple[float, float]:
        """
        Close current position.
        
        Args:
            price: Exit price
            timestep: Current timestep
            
        Returns:
            Tuple of (realized PnL, transaction cost)
        """
        if self.state.position == 0:
            return 0.0, 0.0
        
        position_value = abs(self.state.position)
        cost = position_value * self.transaction_cost
        
        # Calculate PnL
        if self.state.position > 0:  # Long position
            pnl_pct = (price - self.state.entry_price) / self.state.entry_price
        else:  # Short position
            pnl_pct = (self.state.entry_price - price) / self.state.entry_price
        
        pnl = position_value * pnl_pct - cost
        
        # Update cash
        self.state.cash += position_value + pnl
        
        # Record trade
        if self.current_trade is not None:
            self.current_trade.exit_time = timestep
            self.current_trade.exit_price = price
            self.current_trade.pnl = pnl
            self.current_trade.pnl_pct = pnl_pct * 100
            self.current_trade.transaction_cost += cost
            self.trades.append(self.current_trade)
            self.current_trade = None
        
        # Reset position
        self.state.position = 0.0
        self.state.entry_price = 0.0
        self.state.unrealized_pnl = 0.0
        self.holding_time = 0
        
        return pnl, cost
    
    def step(self, current_price: float):
        """
        Update portfolio state for current timestep.
        
        Args:
            current_price: Current asset price
        """
        self.state.update_equity(current_price)
        self.equity_history.append(self.state.total_equity)
        
        if len(self.equity_history) > 1:
            ret = (self.equity_history[-1] - self.equity_history[-2]) / self.equity_history[-2]
            self.returns_history.append(ret)
        
        if self.state.position != 0:
            self.holding_time += 1
    
    def get_metrics(self) -> Dict:
        """
        Calculate and return portfolio performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        equity_array = np.array(self.equity_history)
        
        total_return = (equity_array[-1] - self.initial_capital) / self.initial_capital * 100
        
        # Calculate metrics
        metrics = {
            'total_return_pct': total_return,
            'final_equity': equity_array[-1],
            'num_trades': len(self.trades),
            'sharpe_ratio': calculate_sharpe(self.returns_history),
            'max_drawdown': calculate_max_drawdown(equity_array),
            'sortino_ratio': calculate_sortino(self.returns_history),
        }
        
        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            metrics['win_rate'] = len(winning_trades) / len(self.trades) * 100
            metrics['avg_pnl_per_trade'] = np.mean([t.pnl for t in self.trades])
            metrics['avg_pnl_pct_per_trade'] = np.mean([t.pnl_pct for t in self.trades])
            metrics['total_transaction_costs'] = sum(t.transaction_cost for t in self.trades)
        else:
            metrics['win_rate'] = 0.0
            metrics['avg_pnl_per_trade'] = 0.0
            metrics['avg_pnl_pct_per_trade'] = 0.0
            metrics['total_transaction_costs'] = 0.0
        
        return metrics


def calculate_sharpe(returns: List[float], risk_free_rate: float = 0.0, 
                     annualization_factor: float = np.sqrt(365 * 24)) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate (default 0)
        annualization_factor: Factor to annualize (sqrt of periods per year)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns) * annualization_factor


def calculate_sortino(returns: List[float], risk_free_rate: float = 0.0,
                      annualization_factor: float = np.sqrt(365 * 24)) -> float:
    """
    Calculate Sortino ratio (only penalizes downside volatility).
    
    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate (default 0)
        annualization_factor: Factor to annualize
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    # Only consider negative returns for downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0 if np.mean(excess_returns) <= 0 else float('inf')
    
    downside_std = np.std(downside_returns)
    return np.mean(excess_returns) / downside_std * annualization_factor


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        equity_curve: Array of equity values over time
        
    Returns:
        Maximum drawdown as percentage
    """
    if len(equity_curve) < 2:
        return 0.0
    
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = abs(np.min(drawdown)) * 100
    
    return max_drawdown


def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).
    
    Args:
        total_return: Total return percentage
        max_drawdown: Maximum drawdown percentage
        
    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return 0.0 if total_return <= 0 else float('inf')
    
    return total_return / max_drawdown


def calculate_profit_factor(trades: List[Trade]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        trades: List of Trade objects
        
    Returns:
        Profit factor
    """
    if not trades:
        return 0.0
    
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def format_metrics(metrics: Dict, title: str = "Performance Metrics") -> str:
    """
    Format metrics dictionary as a readable string.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the output
        
    Returns:
        Formatted string
    """
    lines = [
        "=" * 50,
        title,
        "=" * 50,
    ]
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'pct' in key.lower() or 'rate' in key.lower() or 'return' in key.lower() or 'drawdown' in key.lower():
                lines.append(f"  {key}: {value:.2f}%")
            elif 'ratio' in key.lower():
                lines.append(f"  {key}: {value:.3f}")
            else:
                lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


def save_metrics(metrics: Dict, filepath: str):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save file
    """
    # Convert numpy types to Python types
    serializable = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable[key] = float(value)
        else:
            serializable[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_metrics(filepath: str) -> Dict:
    """
    Load metrics from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary of metrics
    """
    with open(filepath, 'r') as f:
        return json.load(f)


class ProgressTracker:
    """Track training progress for checkpoint resume."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.progress_file = f"{checkpoint_path}/progress.json"
        self.progress = {
            'timesteps': 0,
            'episodes': 0,
            'best_reward': float('-inf'),
            'elapsed_time': 0,
            'last_checkpoint': None,
        }
    
    def load(self) -> bool:
        """Load progress from file. Returns True if loaded successfully."""
        try:
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
            return True
        except FileNotFoundError:
            return False
    
    def save(self):
        """Save progress to file."""
        import os
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        self.progress['last_checkpoint'] = datetime.now().isoformat()
        
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def update(self, timesteps: int = None, episodes: int = None, 
               best_reward: float = None, elapsed_time: float = None):
        """Update progress values."""
        if timesteps is not None:
            self.progress['timesteps'] = timesteps
        if episodes is not None:
            self.progress['episodes'] = episodes
        if best_reward is not None and best_reward > self.progress['best_reward']:
            self.progress['best_reward'] = best_reward
        if elapsed_time is not None:
            self.progress['elapsed_time'] = elapsed_time


if __name__ == '__main__':
    # Test utilities
    print("Testing utils.py...")
    
    # Test PortfolioTracker
    tracker = PortfolioTracker(initial_capital=10000, transaction_cost=0.0025)
    
    # Simulate some trades
    prices = [100, 102, 101, 105, 103, 108, 106, 110]
    
    for i, price in enumerate(prices):
        if i == 1:
            cost = tracker.open_position(price, 0.5, 'long', i)
            print(f"Opened long at {price}, cost: {cost:.2f}")
        elif i == 5:
            pnl, cost = tracker.close_position(price, i)
            print(f"Closed at {price}, PnL: {pnl:.2f}, cost: {cost:.2f}")
        
        tracker.step(price)
    
    metrics = tracker.get_metrics()
    print("\n" + format_metrics(metrics))


