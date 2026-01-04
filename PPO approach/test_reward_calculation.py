#!/usr/bin/env python3
"""
Test Reward Calculation
=======================
Standalone test script to verify reward calculation works correctly
for all trading scenarios before training.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_env import TradingEnv
from colab_utils import get_datasets_path


def test_reward_scenarios():
    """Test reward calculation for different trading scenarios."""
    
    print("=" * 70)
    print("REWARD CALCULATION TEST")
    print("=" * 70)
    
    # Find a dataset
    datasets_path = get_datasets_path()
    dataset_files = list(datasets_path.glob("*.csv"))
    
    if not dataset_files:
        print("ERROR: No datasets found!")
        return False
    
    dataset_path = dataset_files[0]
    print(f"\nUsing dataset: {dataset_path.name}")
    
    # Create environment
    env = TradingEnv(
        dataset_path=dataset_path,
        prediction_models=None,  # Don't need models for reward testing
        transaction_cost=0.0025,
        initial_capital=10000.0,
        sequence_length=60,
        train_mode=True,
        max_episode_steps=100,  # Short episodes for testing
    )
    
    print(f"\nInitial capital: {env.initial_capital}")
    print(f"Transaction cost: {env.transaction_cost * 100}%")
    
    all_tests_passed = True
    
    # Test 1: Opening a position
    print("\n" + "-" * 70)
    print("TEST 1: Opening a Position (Buy Small - 25%)")
    print("-" * 70)
    obs, info = env.reset()
    initial_equity = env.portfolio.state.total_equity
    print(f"Initial equity: {initial_equity:.2f}")
    
    # Open position
    action = 1  # Buy Small
    obs, reward, terminated, truncated, info = env.step(action)
    
    current_equity = env.portfolio.state.total_equity
    position = env.portfolio.state.position
    num_closed_trades = len(env.portfolio.trades)
    has_open_trade = env.portfolio.current_trade is not None
    
    print(f"Action: {action} (Buy Small)")
    print(f"Reward: {reward:.6f}")
    print(f"Equity change: {current_equity - initial_equity:.2f}")
    print(f"Position: {position:.2f}")
    print(f"Closed trades: {num_closed_trades}")
    print(f"Open trade: {has_open_trade}")
    print(f"Profit %: {info.get('profit_pct', 0):.6f}")
    
    if reward == 0.0:
        print("❌ FAIL: Reward is 0.0 when opening position!")
        all_tests_passed = False
    else:
        print("✓ PASS: Reward is non-zero")
    
    # Check if position was opened (position > 0) or trade is being tracked
    if position == 0 and not has_open_trade:
        print("❌ FAIL: Position not opened and no trade tracked!")
        all_tests_passed = False
    else:
        print("✓ PASS: Position opened or trade tracked")
    
    # Check equity change is reasonable (should be small negative due to transaction cost)
    equity_change = current_equity - initial_equity
    if abs(equity_change) > initial_equity * 0.1:  # More than 10% change is suspicious
        print(f"⚠ WARNING: Large equity change ({equity_change:.2f}), may indicate data issue")
    else:
        print("✓ PASS: Equity change is reasonable")
    
    # Test 2: Holding with profitable position
    print("\n" + "-" * 70)
    print("TEST 2: Holding with Position (Price increases)")
    print("-" * 70)
    
    previous_equity = current_equity
    previous_reward = reward
    
    # Simulate price increase by stepping forward
    action = 0  # Hold
    obs, reward, terminated, truncated, info = env.step(action)
    
    current_equity = env.portfolio.state.total_equity
    unrealized_pnl = env.portfolio.state.unrealized_pnl
    equity_change = current_equity - previous_equity
    
    print(f"Action: {action} (Hold)")
    print(f"Reward: {reward:.6f}")
    print(f"Equity change: {equity_change:.2f}")
    print(f"Unrealized P&L: {unrealized_pnl:.2f}")
    print(f"Profit %: {info.get('profit_pct', 0):.6f}")
    
    # Check if equity change is reasonable (should be small percentage of capital)
    if abs(equity_change) > initial_equity * 0.1:  # More than 10% change is suspicious
        print(f"⚠ WARNING: Large equity change ({equity_change:.2f}), may indicate data issue")
    else:
        print("✓ PASS: Equity change is reasonable")
    
    # Reward should reflect unrealized P&L if price moved
    if abs(equity_change) > 0.01 and reward == 0.0:
        print("⚠ WARNING: Equity changed but reward is 0 (may be OK if price didn't move)")
    else:
        print("✓ PASS: Reward calculated for holding")
    
    # Test 3: Closing profitable position
    print("\n" + "-" * 70)
    print("TEST 3: Closing Profitable Position")
    print("-" * 70)
    
    previous_equity = current_equity
    previous_trades = len(env.portfolio.trades)
    
    action = 7  # Close Position
    obs, reward, terminated, truncated, info = env.step(action)
    
    current_equity = env.portfolio.state.total_equity
    num_trades = len(env.portfolio.trades)
    position = env.portfolio.state.position
    equity_change = current_equity - previous_equity
    
    print(f"Action: {action} (Close Position)")
    print(f"Reward: {reward:.6f}")
    print(f"Equity change: {equity_change:.2f}")
    print(f"Position: {position:.2f} (should be 0)")
    print(f"Trades: {num_trades} (should be {previous_trades + 1})")
    print(f"Profit %: {info.get('profit_pct', 0):.6f}")
    
    if position != 0:
        print("❌ FAIL: Position not closed!")
        all_tests_passed = False
    else:
        print("✓ PASS: Position closed")
    
    if num_trades != previous_trades + 1:
        print("❌ FAIL: Trade not recorded!")
        all_tests_passed = False
    else:
        print("✓ PASS: Trade recorded")
    
    # Check equity change is reasonable
    if abs(equity_change) > initial_equity * 0.1:  # More than 10% change is suspicious
        print(f"⚠ WARNING: Large equity change ({equity_change:.2f}), may indicate data issue")
    else:
        print("✓ PASS: Equity change is reasonable")
    
    # Test 4: Holding with no position
    print("\n" + "-" * 70)
    print("TEST 4: Holding with No Position")
    print("-" * 70)
    
    previous_equity = current_equity
    action = 0  # Hold
    obs, reward, terminated, truncated, info = env.step(action)
    
    current_equity = env.portfolio.state.total_equity
    
    print(f"Action: {action} (Hold)")
    print(f"Reward: {reward:.6f}")
    print(f"Equity change: {current_equity - previous_equity:.2f}")
    print(f"Position: {env.portfolio.state.position:.2f}")
    
    if abs(reward) > 0.001:
        print("⚠ WARNING: Reward is non-zero when holding with no position (may be OK due to drawdown/sharpe)")
    else:
        print("✓ PASS: Reward is approximately 0 when holding with no position")
    
    # Test 5: Episode tracking
    print("\n" + "-" * 70)
    print("TEST 5: Episode Tracking")
    print("-" * 70)
    
    # Run a short episode (use max_episode_steps to ensure it ends)
    test_env = TradingEnv(
        dataset_path=dataset_path,
        prediction_models=None,
        transaction_cost=0.0025,
        initial_capital=10000.0,
        sequence_length=60,
        train_mode=True,
        max_episode_steps=5,  # Very short episode for testing
    )
    
    obs, info = test_env.reset()
    episode_reward = 0.0
    episode_length = 0
    
    for _ in range(10):  # Should end after 5 steps
        action = np.random.randint(0, 9)  # Random action
        obs, reward, terminated, truncated, info = test_env.step(action)
        episode_reward += reward
        episode_length += 1
        
        if terminated or truncated:
            break
    
    # Check if episode info is in the last step's info
    if terminated or truncated:
        if 'episode' in info:
            episode_info = info['episode']
            if 'r' in episode_info and 'l' in episode_info:
                print(f"✓ PASS: Episode info found")
                print(f"  Episode reward: {episode_info['r']:.6f}")
                print(f"  Episode length: {episode_info['l']}")
                print(f"  Calculated reward: {episode_reward:.6f}")
                print(f"  Calculated length: {episode_length}")
                
                if abs(episode_info['r'] - episode_reward) > 0.01:
                    print("⚠ WARNING: Episode reward doesn't match calculated sum")
                else:
                    print("✓ PASS: Episode reward matches")
            else:
                print("❌ FAIL: Episode info missing 'r' or 'l'")
                all_tests_passed = False
        else:
            print("❌ FAIL: No episode info in info dict!")
            all_tests_passed = False
    else:
        print("⚠ WARNING: Episode didn't end, can't test episode tracking")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    if all_tests_passed:
        print("✓ ALL TESTS PASSED")
        print("\nReward calculation appears to be working correctly.")
        print("You can proceed with training.")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues before training.")
    
    test_env.close()
    env.close()
    return all_tests_passed


if __name__ == '__main__':
    success = test_reward_scenarios()
    sys.exit(0 if success else 1)

