#!/usr/bin/env python3
"""
Visualization Script
====================
Plot training progress and trade analysis for PPO trading agent.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from colab_utils import get_ppo_path, get_results_path, get_logs_path, get_checkpoints_path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_training_progress(log_path: str = None, save_path: str = None):
    """
    Plot training progress from TensorBoard logs.
    
    Args:
        log_path: Path to TensorBoard logs
        save_path: Path to save figure
    """
    if log_path is None:
        log_path = get_logs_path()
    
    log_path = Path(log_path)
    
    # Try to find progress.json from checkpoints
    checkpoints = list(get_checkpoints_path().glob("*/progress.json"))
    
    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoint progress files")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for cp_file in checkpoints:
            with open(cp_file, 'r') as f:
                progress = json.load(f)
            
            name = cp_file.parent.name
            timesteps = progress.get('timesteps', 0)
            best_reward = progress.get('best_reward', 0)
            
            print(f"  {name}: {timesteps:,} timesteps, best reward: {best_reward:.2f}")
        
        # If we have tensorboard data, plot it
        try:
            from tensorboard.backend.event_processing import event_accumulator
            
            # Find event files
            event_files = list(log_path.glob("**/events.out.tfevents.*"))
            
            if event_files:
                ea = event_accumulator.EventAccumulator(str(event_files[0].parent))
                ea.Reload()
                
                # Get available scalars
                tags = ea.Tags().get('scalars', [])
                print(f"Available metrics: {tags}")
                
                # Plot reward
                if 'rollout/ep_rew_mean' in tags:
                    rewards = ea.Scalars('rollout/ep_rew_mean')
                    steps = [r.step for r in rewards]
                    values = [r.value for r in rewards]
                    
                    axes[0].plot(steps, values, label='Mean Episode Reward')
                    axes[0].set_xlabel('Timesteps')
                    axes[0].set_ylabel('Reward')
                    axes[0].set_title('Training Reward')
                    axes[0].legend()
                
                # Plot episode length
                if 'rollout/ep_len_mean' in tags:
                    lengths = ea.Scalars('rollout/ep_len_mean')
                    steps = [l.step for l in lengths]
                    values = [l.value for l in lengths]
                    
                    axes[1].plot(steps, values, label='Mean Episode Length', color='orange')
                    axes[1].set_xlabel('Timesteps')
                    axes[1].set_ylabel('Episode Length')
                    axes[1].set_title('Episode Length')
                    axes[1].legend()
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"Saved to: {save_path}")
                
                plt.show()
                return
                
        except ImportError:
            print("TensorBoard not available for parsing logs")
        except Exception as e:
            print(f"Error parsing TensorBoard logs: {e}")
    
    print("No training data found to plot")


def plot_equity_curve(equity_values: List[float], 
                     title: str = "Portfolio Equity Curve",
                     save_path: str = None):
    """
    Plot equity curve.
    
    Args:
        equity_values: List of equity values over time
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(equity_values, linewidth=2, color='blue', label='Portfolio Equity')
    
    # Add horizontal line at starting equity
    if equity_values:
        ax.axhline(y=equity_values[0], color='gray', linestyle='--', alpha=0.5, label='Starting Equity')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Equity')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add return annotation
    if len(equity_values) > 1:
        total_return = (equity_values[-1] - equity_values[0]) / equity_values[0] * 100
        ax.text(0.02, 0.98, f'Total Return: {total_return:.2f}%',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def plot_action_distribution(actions: List[int], save_path: str = None):
    """
    Plot distribution of actions taken by the agent.
    
    Args:
        actions: List of action indices
        save_path: Path to save figure
    """
    action_names = ['Hold', 'Buy Small', 'Buy Medium', 'Buy Large',
                   'Sell Small', 'Sell Medium', 'Sell Large',
                   'Close Position', 'Reverse Position']
    
    # Count actions
    action_counts = [actions.count(i) for i in range(9)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['gray', 'lightgreen', 'green', 'darkgreen',
              'lightcoral', 'red', 'darkred', 'orange', 'purple']
    
    bars = ax.bar(action_names, action_counts, color=colors)
    
    ax.set_xlabel('Action')
    ax.set_ylabel('Count')
    ax.set_title('Action Distribution')
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    # Add percentage labels
    total = sum(action_counts)
    for bar, count in zip(bars, action_counts):
        if count > 0:
            pct = count / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def plot_trades_on_price(prices: np.ndarray, 
                        trades: List[Dict],
                        title: str = "Trades on Price Chart",
                        save_path: str = None):
    """
    Plot trades overlaid on price chart.
    
    Args:
        prices: Array of prices
        trades: List of trade dictionaries with entry/exit info
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot price
    ax.plot(prices, linewidth=1, color='blue', alpha=0.7, label='Price')
    
    # Plot trades
    for trade in trades:
        entry_idx = trade.get('entry_time', 0)
        exit_idx = trade.get('exit_time', len(prices) - 1)
        entry_price = trade.get('entry_price', prices[entry_idx] if entry_idx < len(prices) else 0)
        exit_price = trade.get('exit_price', prices[exit_idx] if exit_idx < len(prices) else 0)
        trade_type = trade.get('type', 'long')
        pnl = trade.get('pnl', 0)
        
        # Entry marker
        color = 'green' if trade_type == 'long' else 'red'
        ax.scatter(entry_idx, entry_price, color=color, marker='^' if trade_type == 'long' else 'v',
                  s=100, zorder=5)
        
        # Exit marker
        exit_color = 'darkgreen' if pnl > 0 else 'darkred'
        ax.scatter(exit_idx, exit_price, color=exit_color, marker='o', s=100, zorder=5)
        
        # Connect entry and exit
        ax.plot([entry_idx, exit_idx], [entry_price, exit_price], 
               color=color, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def plot_comparison(results_df: pd.DataFrame, 
                   metric: str = 'Mean Return %',
                   save_path: str = None):
    """
    Plot comparison of different strategies.
    
    Args:
        results_df: DataFrame with comparison results
        metric: Metric to compare
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = results_df['Strategy'].tolist()
    values = results_df[metric].tolist()
    
    colors = ['green' if v > 0 else 'red' for v in values]
    
    bars = ax.bar(strategies, values, color=colors)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Strategy')
    ax.set_ylabel(metric)
    ax.set_title(f'Strategy Comparison: {metric}')
    
    # Add value labels
    for bar, val in zip(bars, values):
        va = 'bottom' if val >= 0 else 'top'
        offset = 0.5 if val >= 0 else -0.5
        ax.text(bar.get_x() + bar.get_width()/2, val + offset,
               f'{val:.2f}', ha='center', va=va, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def plot_reward_distribution(rewards: List[float], save_path: str = None):
    """
    Plot distribution of episode rewards.
    
    Args:
        rewards: List of episode rewards
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(rewards, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    axes[0].axvline(np.median(rewards), color='green', linestyle='--', label=f'Median: {np.median(rewards):.2f}')
    axes[0].set_xlabel('Reward')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Reward Distribution')
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(rewards, vert=True)
    axes[1].set_ylabel('Reward')
    axes[1].set_title('Reward Box Plot')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def create_summary_dashboard(
    equity_curve: List[float] = None,
    actions: List[int] = None,
    trades: List[Dict] = None,
    metrics: Dict = None,
    save_path: str = None
):
    """
    Create a summary dashboard with multiple plots.
    
    Args:
        equity_curve: Equity values over time
        actions: List of actions taken
        trades: List of trade dictionaries
        metrics: Performance metrics dictionary
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)
    
    # 1. Equity Curve (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    if equity_curve:
        ax1.plot(equity_curve, linewidth=2, color='blue')
        ax1.axhline(y=equity_curve[0], color='gray', linestyle='--', alpha=0.5)
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
        ax1.set_title(f'Equity Curve (Return: {total_return:.2f}%)')
    else:
        ax1.text(0.5, 0.5, 'No equity data', ha='center', va='center')
        ax1.set_title('Equity Curve')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Equity')
    
    # 2. Action Distribution (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    if actions:
        action_names = ['Hold', 'Buy S', 'Buy M', 'Buy L', 'Sell S', 'Sell M', 'Sell L', 'Close', 'Rev']
        action_counts = [actions.count(i) for i in range(9)]
        colors = ['gray', 'lightgreen', 'green', 'darkgreen', 
                 'lightcoral', 'red', 'darkred', 'orange', 'purple']
        ax2.bar(action_names, action_counts, color=colors)
        ax2.set_title('Action Distribution')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax2.text(0.5, 0.5, 'No action data', ha='center', va='center')
        ax2.set_title('Action Distribution')
    
    # 3. Trade P&L (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    if trades:
        pnls = [t.get('pnl', 0) for t in trades]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title(f'Trade P&L ({len(trades)} trades)')
        ax3.set_xlabel('Trade #')
        ax3.set_ylabel('P&L')
    else:
        ax3.text(0.5, 0.5, 'No trade data', ha='center', va='center')
        ax3.set_title('Trade P&L')
    
    # 4. Metrics Table (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    if metrics:
        metrics_text = '\n'.join([f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}'
                                 for k, v in list(metrics.items())[:10]])
        ax4.text(0.1, 0.9, 'Performance Metrics', fontsize=14, fontweight='bold',
                transform=ax4.transAxes, verticalalignment='top')
        ax4.text(0.1, 0.8, metrics_text, fontsize=11, family='monospace',
                transform=ax4.transAxes, verticalalignment='top')
    else:
        ax4.text(0.5, 0.5, 'No metrics data', ha='center', va='center')
    
    # 5. Cumulative Returns (bottom, full width)
    ax5 = fig.add_subplot(gs[2, :])
    if equity_curve:
        returns = np.diff(equity_curve) / equity_curve[:-1] * 100
        cumulative = np.cumsum(returns)
        ax5.fill_between(range(len(cumulative)), cumulative, alpha=0.3)
        ax5.plot(cumulative, linewidth=2)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.set_title('Cumulative Returns')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Cumulative Return %')
    else:
        ax5.text(0.5, 0.5, 'No return data', ha='center', va='center')
        ax5.set_title('Cumulative Returns')
    
    plt.suptitle('PPO Trading Agent - Performance Dashboard', fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved to: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize PPO training results')
    parser.add_argument('--training-progress', action='store_true',
                       help='Plot training progress from TensorBoard logs')
    parser.add_argument('--log-path', type=str, default=None,
                       help='Path to TensorBoard logs')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save figures')
    
    args = parser.parse_args()
    
    if args.training_progress:
        save_path = None
        if args.save_dir:
            save_path = Path(args.save_dir) / 'training_progress.png'
        plot_training_progress(args.log_path, save_path)
    else:
        print("Visualization module loaded.")
        print("\nAvailable functions:")
        print("  - plot_training_progress(log_path, save_path)")
        print("  - plot_equity_curve(equity_values, title, save_path)")
        print("  - plot_action_distribution(actions, save_path)")
        print("  - plot_trades_on_price(prices, trades, title, save_path)")
        print("  - plot_comparison(results_df, metric, save_path)")
        print("  - plot_reward_distribution(rewards, save_path)")
        print("  - create_summary_dashboard(equity_curve, actions, trades, metrics, save_path)")


