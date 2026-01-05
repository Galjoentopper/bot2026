"""
Trading Environment
===================
Gym-compatible environment for cryptocurrency trading with PPO.

This environment uses prediction models as feature generators and
implements a discrete action space for trading decisions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import gymnasium as gym
from gymnasium import spaces

from colab_utils import get_project_path, get_datasets_path
from utils import PortfolioTracker, calculate_sharpe, calculate_max_drawdown
from reward_functions import RewardCalculator, RewardConfig


class TradingEnv(gym.Env):
    """
    Gym-compatible trading environment.
    
    State includes:
    - Prediction model outputs (class, confidence, probabilities)
    - Price and volume features
    - Technical indicators
    - Portfolio state (position, cash, unrealized P&L)
    
    Actions (Discrete, 4 options):
    - 0: Hold (do nothing)
    - 1: Buy (open/increase long position, 100% of available cash)
    - 2: Sell (reduce/close long position, 100% of position)
    - 3: Close Position (fully close any position)
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        dataset_path: Union[str, Path] = None,
        prediction_models = None,
        transaction_cost: float = 0.0025,
        initial_capital: float = 10000.0,
        sequence_length: int = 60,
        train_mode: bool = True,
        train_split: float = 0.6,
        validation_split: float = 0.2,
        validation_mode: bool = False,
        reward_config: Dict = None,
        max_episode_steps: int = None,
        render_mode: str = None,
        prediction_horizons: List[int] = None,
    ):
        """
        Initialize trading environment.
        
        Args:
            dataset_path: Path to CSV dataset file
            prediction_models: Loaded prediction model(s) for feature extraction
            transaction_cost: Cost per trade (0.0025 = 0.25%)
            initial_capital: Starting capital
            sequence_length: Lookback window for sequences
            train_mode: If True, use training data (60%); if False, use validation or test data
            train_split: Fraction of data for training (default: 0.6 for 60/20/20 split)
            validation_split: Fraction of data for validation (default: 0.2 for 60/20/20 split)
            validation_mode: If True and train_mode=False, use validation data (20%);
                           if False and train_mode=False, use test data (20%)
            reward_config: Configuration for reward calculation
            max_episode_steps: Maximum steps per episode (None = full dataset)
            render_mode: Render mode ('human' or 'ansi')
        """
        super().__init__()
        
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        self.sequence_length = sequence_length
        self.train_mode = train_mode
        self.train_split = train_split
        self.validation_split = validation_split
        self.validation_mode = validation_mode
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # Load and prepare data
        self.df = None
        self.prices = None
        self.features = None
        self.data_start_idx = 0
        self.data_end_idx = 0
        
        if dataset_path is not None:
            self._load_dataset(dataset_path)
        
        # Prediction models
        self.prediction_models = prediction_models
        self.use_predictions = prediction_models is not None
        # Multi-horizon prediction: [1, 2, 3] for short-term, [5, 10] for medium-term
        # Default: [1, 2, 3, 5, 10] (both short and medium term)
        self.prediction_horizons = prediction_horizons if prediction_horizons else [1, 2, 3, 5, 10]
        
        # Portfolio tracking
        self.portfolio = PortfolioTracker(
            initial_capital=initial_capital,
            transaction_cost=transaction_cost
        )
        
        # Reward calculator
        reward_cfg = RewardConfig.from_dict(reward_config) if reward_config else RewardConfig()
        self.reward_calculator = RewardCalculator(reward_cfg)
        
        # Episode state
        self.current_step = 0
        self.episode_start_idx = 0
        self.previous_equity = initial_capital  # Track previous equity for reward calculation
        self.done = False
        self.episode_reward = 0.0  # Track cumulative episode reward
        self.episode_length = 0  # Track episode length
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)
        
        # Observation space components:
        # - Prediction features: 
        #   * Single-step (5): class_normalized, confidence, prob_fall, prob_stationary, prob_rise
        #   * Multi-horizon: 5 + (len(horizons)-1) * 4 per additional horizon
        #     Each horizon adds: class_norm, confidence, prob_fall, prob_rise
        # - Price features (8): price_normalized, price_change, volume_normalized, rsi, macd, atr, stoch_k, stoch_d
        # - Temporal features (9): price_momentum[5,10,20], volume_momentum[5,10,20], volatility[5,10,20]
        # - Portfolio state (4): position_normalized, unrealized_pnl, cash_ratio, holding_time_normalized
        # - Action mask (4): can_hold, can_buy, can_sell, can_close
        
        # Calculate prediction feature dimension
        if self.use_predictions and len(self.prediction_horizons) > 0:
            # Base features (t+1): 5
            # Additional horizons: each adds 4 features
            pred_dim = 5 + (len(self.prediction_horizons) - 1) * 4
        else:
            pred_dim = 5  # Default features even without model
        
        # Price features: 8 (added ATR, stoch_k, stoch_d)
        # Temporal features: 9 (3 lookbacks * 3 features each)
        # Portfolio: 4
        # Action mask: 4
        self.observation_dim = pred_dim + 8 + 9 + 4 + 4  # pred + price + temporal + portfolio + action_mask
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )
        
        # Price normalization stats
        self.price_mean = 0.0
        self.price_std = 1.0
        self.volume_mean = 0.0
        self.volume_std = 1.0
    
    def _load_dataset(self, dataset_path: Union[str, Path]):
        """Load and prepare dataset."""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            # Try to find in datasets folder
            datasets_path = get_datasets_path()
            possible_paths = list(datasets_path.glob(f"*{dataset_path.stem}*"))
            if possible_paths:
                dataset_path = possible_paths[0]
            else:
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        print(f"Loading dataset: {dataset_path.name}")
        
        # Load CSV
        df = pd.read_csv(dataset_path)
        
        # Convert timestamp if present
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # Handle NaN values (using modern pandas syntax)
        df = df.ffill().bfill().fillna(0)
        
        self.df = df
        
        # Extract prices
        self.prices = df['close'].values
        
        # Calculate normalization stats
        self.price_mean = np.mean(self.prices)
        self.price_std = np.std(self.prices) + 1e-8
        self.volume_mean = np.mean(df['volume'].values)
        self.volume_std = np.std(df['volume'].values) + 1e-8
        
        # Prepare feature columns
        feature_cols = ['close', 'open', 'high', 'low', 'volume', 'rsi', 'macd', 
                       'macd_signal', 'bb_percent', 'price_change', 'volume_ratio']
        available_cols = [c for c in feature_cols if c in df.columns]
        self.features = df[available_cols].values
        
        # Calculate three-way split indices (60/20/20: train/val/test)
        total_len = len(self.prices)
        train_split_idx = int(total_len * self.train_split)  # 60%
        val_split_idx = int(total_len * (self.train_split + self.validation_split))  # 80%
        
        if self.train_mode:
            # Training data: first 60% (prediction models trained on this)
            self.data_start_idx = self.sequence_length
            self.data_end_idx = train_split_idx
            split_name = "Training"
        elif self.validation_mode:
            # Validation data: 60-80% (models never saw this)
            self.data_start_idx = train_split_idx
            self.data_end_idx = val_split_idx
            split_name = "Validation"
        else:
            # Test data: last 20% (models never saw this)
            self.data_start_idx = val_split_idx
            self.data_end_idx = total_len
            split_name = "Test"
        
        print(f"  Total samples: {total_len}")
        print(f"  {split_name} data: Using indices {self.data_start_idx} to {self.data_end_idx}")
        print(f"  Available steps: {self.data_end_idx - self.data_start_idx}")
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe."""
        result = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        result['rsi'] = (100 - (100 / (1 + rs))) / 100.0  # Normalize to 0-1
        
        # MACD
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        result['macd'] = (ema_fast - ema_slow) / df['close']  # Normalize
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma = df['close'].rolling(window=20, min_periods=1).mean()
        std = df['close'].rolling(window=20, min_periods=1).std().fillna(1)
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        result['bb_percent'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        # Price change
        result['price_change'] = df['close'].pct_change().fillna(0)
        
        # Volume ratio
        vol_ma = df['volume'].rolling(window=20, min_periods=1).mean()
        result['volume_ratio'] = df['volume'] / (vol_ma + 1e-10)
        
        # ATR (Average True Range) - 14 period
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result['atr'] = true_range.rolling(window=14, min_periods=1).mean() / df['close']  # Normalize
        
        # Stochastic Oscillator (%K, %D) - 14 period
        low_14 = df['low'].rolling(window=14, min_periods=1).min()
        high_14 = df['high'].rolling(window=14, min_periods=1).max()
        result['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
        result['stoch_d'] = result['stoch_k'].rolling(window=3, min_periods=1).mean()
        # Normalize to 0-1
        result['stoch_k'] = result['stoch_k'] / 100.0
        result['stoch_d'] = result['stoch_d'] / 100.0
        
        # Price acceleration (rate of change of price_change)
        result['price_acceleration'] = result['price_change'].diff().fillna(0)
        
        return result
    
    def _get_prediction_features(self) -> np.ndarray:
        """Get features from prediction models with multi-horizon support."""
        if not self.use_predictions or self.prediction_models is None:
            # Return default features if no model
            # If multi-horizon, pad to expected size
            if len(self.prediction_horizons) > 0:
                base_features = np.array([0.5, 0.33, 0.33, 0.34, 0.33])
                # Add zeros for additional horizons
                n_additional = len(self.prediction_horizons) - 1
                additional = np.tile([0.5, 0.33, 0.33, 0.34], n_additional)
                return np.concatenate([base_features, additional])
            else:
                return np.array([0.5, 0.33, 0.33, 0.34, 0.33])
        
        try:
            # Get sequence for prediction
            seq_start = self.episode_start_idx + self.current_step - self.sequence_length
            seq_end = self.episode_start_idx + self.current_step
            
            if seq_start < 0:
                seq_start = 0
            
            sequence = self.features[seq_start:seq_end]
            
            # Pad if needed
            if len(sequence) < self.sequence_length:
                pad_len = self.sequence_length - len(sequence)
                sequence = np.vstack([np.zeros((pad_len, sequence.shape[1])), sequence])
            
            # Get multi-horizon predictions if horizons specified
            if len(self.prediction_horizons) > 1:
                # Use get_features with horizons for multi-step prediction
                features = self.prediction_models.get_features(sequence, horizons=self.prediction_horizons)
            else:
                # Single-step prediction (backward compatible)
                pred_class, confidence, probs = self.prediction_models.predict(sequence)
                features = np.array([
                    pred_class / 2.0,  # Normalize class to 0-1
                    confidence,
                    probs[0],  # Fall probability
                    probs[1],  # Stationary probability
                    probs[2],  # Rise probability
                ])
            
            return features
        except Exception as e:
            # Return neutral features on error (with correct size)
            if len(self.prediction_horizons) > 0:
                base_features = np.array([0.5, 0.33, 0.33, 0.34, 0.33])
                n_additional = len(self.prediction_horizons) - 1
                additional = np.tile([0.5, 0.33, 0.33, 0.34], n_additional)
                return np.concatenate([base_features, additional])
            else:
                return np.array([0.5, 0.33, 0.33, 0.34, 0.33])
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector with enhanced features."""
        idx = self.episode_start_idx + self.current_step
        
        # Prediction features
        pred_features = self._get_prediction_features()
        
        # Price features (8)
        current_price = self.prices[idx]
        price_normalized = (current_price - self.price_mean) / self.price_std
        
        if idx > 0 and self.prices[idx - 1] > 0:
            price_change = (current_price - self.prices[idx - 1]) / self.prices[idx - 1]
            # Clamp to reasonable range
            price_change = np.clip(price_change, -0.9, 10.0)  # Allow up to 1000% change
        else:
            price_change = 0.0
        
        volume = self.df['volume'].iloc[idx] if 'volume' in self.df.columns else 0
        volume_normalized = (volume - self.volume_mean) / self.volume_std
        
        rsi = self.df['rsi'].iloc[idx] if 'rsi' in self.df.columns else 0.5
        macd = self.df['macd'].iloc[idx] if 'macd' in self.df.columns else 0.0
        atr = self.df['atr'].iloc[idx] if 'atr' in self.df.columns else 0.0
        stoch_k = self.df['stoch_k'].iloc[idx] if 'stoch_k' in self.df.columns else 0.5
        stoch_d = self.df['stoch_d'].iloc[idx] if 'stoch_d' in self.df.columns else 0.5
        
        price_features = np.array([
            price_normalized,
            price_change * 10,  # Scale for visibility
            np.clip(volume_normalized, -5, 5),  # Clip outliers
            rsi,
            np.clip(macd * 100, -1, 1),  # Scale and clip
            np.clip(atr * 100, 0, 1),  # ATR normalized
            stoch_k,
            stoch_d,
        ])
        
        # Temporal features (9): momentum and volatility for lookbacks [5, 10, 20]
        lookbacks = [5, 10, 20]
        temporal_features = []
        for lookback in lookbacks:
            if idx >= lookback:
                # Price momentum
                price_momentum = (current_price - self.prices[idx - lookback]) / (self.prices[idx - lookback] + 1e-10)
                price_momentum = np.clip(price_momentum, -0.9, 10.0)
                
                # Volume momentum
                vol_window = self.df['volume'].iloc[idx - lookback:idx]
                vol_mean = vol_window.mean() if len(vol_window) > 0 else volume
                volume_momentum = (volume - vol_mean) / (vol_mean + 1e-10) if vol_mean > 0 else 0.0
                volume_momentum = np.clip(volume_momentum, -5, 5)
                
                # Volatility (rolling std of returns)
                price_window = self.df['close'].iloc[idx - lookback:idx + 1]
                returns = price_window.pct_change().dropna()
                volatility = returns.std() if len(returns) > 0 else 0.0
                volatility = np.clip(volatility, 0, 1.0)
            else:
                price_momentum = 0.0
                volume_momentum = 0.0
                volatility = 0.0
            
            temporal_features.extend([price_momentum, volume_momentum, volatility])
        
        temporal_features = np.array(temporal_features)
        
        # Portfolio state (4)
        total_equity = self.portfolio.state.total_equity
        position = self.portfolio.state.position
        cash = self.portfolio.state.cash
        
        position_normalized = position / total_equity if total_equity > 0 else 0
        unrealized_pnl = self.portfolio.state.unrealized_pnl
        cash_ratio = cash / total_equity if total_equity > 0 else 1.0
        holding_time_normalized = min(self.portfolio.holding_time / 100.0, 1.0)
        
        portfolio_features = np.array([
            np.clip(position_normalized, -1, 1),
            np.clip(unrealized_pnl, -1, 1),
            cash_ratio,
            holding_time_normalized,
        ])
        
        # Action mask (4): can_hold, can_buy, can_sell, can_close
        action_mask = np.array([
            1.0,  # Hold is always valid
            1.0 if cash > 0 else 0.0,  # Can buy if have cash
            1.0 if position > 0 else 0.0,  # Can sell if have position
            1.0 if position != 0 else 0.0,  # Can close if have position
        ])
        
        # Combine all features
        observation = np.concatenate([
            pred_features, 
            price_features, 
            temporal_features, 
            portfolio_features,
            action_mask
        ])
        
        return observation.astype(np.float32)
    
    def _is_action_valid(self, action: int) -> bool:
        """
        Check if action is valid given current portfolio state.
        
        Args:
            action: Action to check (0-3)
            
        Returns:
            True if action is valid, False otherwise
        """
        position = self.portfolio.state.position
        cash = self.portfolio.state.cash
        
        if action == 0:  # Hold - always valid
            return True
        elif action == 1:  # Buy - need cash
            return cash > 0
        elif action == 2:  # Sell - need position
            return position > 0
        elif action == 3:  # Close - need position
            return position != 0
        return False
    
    def _execute_action(self, action: int) -> Tuple[float, float, bool]:
        """
        Execute trading action.
        
        Returns:
            Tuple of (profit_pct, transaction_cost, position_changed)
        """
        idx = self.episode_start_idx + self.current_step
        current_price = self.prices[idx]
        
        profit_pct = 0.0
        cost = 0.0
        position_changed = False
        
        position = self.portfolio.state.position
        cash = self.portfolio.state.cash
        
        if action == 0:  # Hold
            pass
        
        elif action == 1:  # Buy (100% of available cash)
            if cash > 0:
                cost = self.portfolio.open_position(current_price, 1.0, 'long', self.current_step)
                position_changed = True
        
        elif action == 2:  # Sell (100% of position)
            if position > 0:
                pnl, cost = self.portfolio.close_position(current_price, self.current_step)
                profit_pct = pnl / self.initial_capital if self.initial_capital > 0 else 0
                position_changed = True
        
        elif action == 3:  # Close Position
            if position != 0:
                pnl, cost = self.portfolio.close_position(current_price, self.current_step)
                profit_pct = pnl / self.initial_capital if self.initial_capital > 0 else 0
                position_changed = True
        
        # Update portfolio with current price
        self.portfolio.step(current_price)
        
        return profit_pct, cost, position_changed
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset portfolio
        self.portfolio.reset()
        
        # Reset reward calculator
        self.reward_calculator.reset(self.initial_capital)
        
        # Reset episode state
        self.current_step = 0
        self.done = False
        self.previous_equity = self.initial_capital  # Reset previous equity tracking
        self.episode_reward = 0.0  # Reset episode reward tracking
        self.episode_length = 0  # Reset episode length tracking
        
        # Determine episode start (can randomize for training)
        if self.train_mode and self.np_random is not None:
            max_start = self.data_end_idx - self.data_start_idx
            if self.max_episode_steps:
                max_start = max(0, max_start - self.max_episode_steps)
            
            if max_start > 0:
                self.episode_start_idx = self.data_start_idx + self.np_random.integers(0, max_start)
            else:
                self.episode_start_idx = self.data_start_idx
        else:
            self.episode_start_idx = self.data_start_idx
        
        observation = self._get_observation()
        
        info = {
            'episode_start_idx': self.episode_start_idx,
            'initial_capital': self.initial_capital,
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0-3)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.done:
            raise RuntimeError("Episode has ended. Call reset().")
        
        # Validate action (action masking)
        action_valid = self._is_action_valid(action)
        
        # Track position state before action execution
        has_position_before = (self.portfolio.state.position != 0)
        
        # Execute action (only if valid)
        if action_valid:
            profit_pct, transaction_cost, position_changed = self._execute_action(action)
        else:
            # Invalid action: no execution, no cost, no position change
            profit_pct = 0.0
            transaction_cost = 0.0
            position_changed = False
        
        # Calculate reward
        current_equity = self.portfolio.state.total_equity
        has_position_after = (self.portfolio.state.position != 0)
        
        # Determine if position is being opened or closed
        is_opening_position = position_changed and not has_position_before and has_position_after
        is_closing_position = position_changed and has_position_before and not has_position_after
        
        # If position changed, account for immediate equity change (transaction costs, position value)
        if position_changed:
            # Account for immediate equity change (transaction costs, position value)
            equity_change = current_equity - self.previous_equity
            if abs(equity_change) > 1e-6:  # Only if significant change
                profit_pct = equity_change / self.initial_capital if self.initial_capital > 0 else 0
        # If holding and no position change, include unrealized P&L in profit_pct
        # This ensures the agent gets rewarded for holding profitable positions
        elif not position_changed:
            if self.portfolio.state.position != 0:
                # Holding with a position: reward based on unrealized P&L change
                equity_change = current_equity - self.previous_equity
                profit_pct = equity_change / self.initial_capital if self.initial_capital > 0 else 0
            else:
                # Holding with no position: still track equity changes (should be 0, but allows for future features)
                # For now, profit_pct stays 0 when holding with no position
                profit_pct = 0.0
        
        reward = self.reward_calculator.calculate_reward(
            profit_pct=profit_pct,
            transaction_cost=transaction_cost,
            current_equity=current_equity,
            holding_time=self.portfolio.holding_time,
            action_valid=action_valid,  # Pass action validity
            position_changed=position_changed,
            has_position=has_position_after,
            is_opening_position=is_opening_position,
            action_type=action,
            log_components=True,  # Enable diagnostic logging
        )
        
        # Update previous equity for next step (after reward calculation)
        # This ensures equity tracking is correct for next step's reward calculation
        assert current_equity >= 0, f"Equity cannot be negative: {current_equity}"
        self.previous_equity = current_equity
        
        # Advance step
        self.current_step += 1
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # End if we've exhausted data
        if self.episode_start_idx + self.current_step >= self.data_end_idx - 1:
            terminated = True
        
        # End if max steps reached
        if self.max_episode_steps and self.current_step >= self.max_episode_steps:
            truncated = True
        
        # End if bankrupt (equity < 10% of initial)
        if current_equity < self.initial_capital * 0.1:
            terminated = True
            reward -= 10.0  # Large penalty for bankruptcy
        
        self.done = terminated or truncated
        
        # Get observation
        observation = self._get_observation()
        
        # Update episode tracking (after all reward modifications)
        self.episode_reward += reward
        self.episode_length += 1
        
        # Build info dict
        info = {
            'step': self.current_step,
            'equity': current_equity,
            'position': self.portfolio.state.position,
            'cash': self.portfolio.state.cash,
            'unrealized_pnl': self.portfolio.state.unrealized_pnl,
            'num_trades': len(self.portfolio.trades),
            'profit_pct': profit_pct,
            'action': action,
            'action_valid': action_valid,  # Add action validity to info
        }
        
        # Add episode info when episode ends (for stable-baselines3 EvalCallback)
        if terminated or truncated:
            # Get actual trading metrics for this episode
            episode_metrics = self.get_episode_metrics()
            total_return_pct = episode_metrics.get('total_return_pct', 0.0)
            num_trades = episode_metrics.get('num_trades', 0)
            final_equity = episode_metrics.get('final_equity', self.initial_capital)
            
            info['episode'] = {
                'r': self.episode_reward,  # Cumulative reward (for PPO learning)
                'l': self.episode_length,
                # Add actual trading performance metrics
                'total_return_pct': total_return_pct,
                'num_trades': num_trades,
                'final_equity': final_equity,
            }
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[str]:
        """Render environment state."""
        if self.render_mode == 'ansi' or self.render_mode == 'human':
            idx = self.episode_start_idx + self.current_step
            price = self.prices[idx] if idx < len(self.prices) else 0
            
            output = (
                f"Step: {self.current_step} | "
                f"Price: {price:.2f} | "
                f"Equity: {self.portfolio.state.total_equity:.2f} | "
                f"Position: {self.portfolio.state.position:.2f} | "
                f"Trades: {len(self.portfolio.trades)}"
            )
            
            if self.render_mode == 'human':
                print(output)
            
            return output
        
        return None
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_episode_metrics(self) -> Dict:
        """Get performance metrics for current episode."""
        return self.portfolio.get_metrics()
    
    def get_reward_breakdown(self) -> Dict:
        """
        Get detailed reward breakdown for debugging.
        
        Returns:
            Dictionary with reward components and statistics
        """
        if hasattr(self.reward_calculator, 'last_reward_components'):
            return self.reward_calculator.last_reward_components.copy()
        return {}
    
    def get_reward_statistics(self) -> Dict:
        """
        Get reward statistics for current episode.
        
        Returns:
            Dictionary with reward statistics
        """
        return self.reward_calculator.get_reward_statistics()


def create_trading_env(
    dataset_name: str,
    prediction_models = None,
    train_mode: bool = True,
    prediction_horizons: List[int] = None,
    **kwargs
) -> TradingEnv:
    """
    Factory function to create trading environment.
    
    Args:
        dataset_name: Name of dataset file (without path)
        prediction_models: Loaded prediction models
        train_mode: Training or evaluation mode
        prediction_horizons: List of prediction horizons (e.g., [1, 2, 3, 5, 10])
        **kwargs: Additional arguments for TradingEnv
        
    Returns:
        Configured TradingEnv
    """
    datasets_path = get_datasets_path()
    
    # Find dataset file
    matches = list(datasets_path.glob(f"*{dataset_name}*"))
    if not matches:
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")
    
    dataset_path = matches[0]
    
    # Set default horizons if not provided
    if prediction_horizons is None:
        prediction_horizons = [1, 2, 3, 5, 10]
    
    # Extract validation_mode from kwargs if present
    validation_mode = kwargs.pop('validation_mode', False)
    
    return TradingEnv(
        dataset_path=dataset_path,
        prediction_models=prediction_models,
        train_mode=train_mode,
        validation_mode=validation_mode,
        prediction_horizons=prediction_horizons,
        **kwargs
    )


if __name__ == '__main__':
    # Test the environment
    print("Testing trading_env.py...")
    print()
    
    # Create environment without prediction models (for testing)
    try:
        from colab_utils import get_datasets_path
        datasets = list(get_datasets_path().glob("*.csv"))
        
        if datasets:
            print(f"Found dataset: {datasets[0].name}")
            
            env = TradingEnv(
                dataset_path=datasets[0],
                prediction_models=None,
                train_mode=True,
                max_episode_steps=100,
                render_mode='human'
            )
            
            print(f"\nAction space: {env.action_space}")
            print(f"Observation space: {env.observation_space}")
            
            # Test reset
            obs, info = env.reset()
            print(f"\nInitial observation shape: {obs.shape}")
            print(f"Initial info: {info}")
            
            # Run a few steps
            print("\nRunning 10 random steps:")
            total_reward = 0
            for i in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                env.render()
                
                if terminated or truncated:
                    break
            
            print(f"\nTotal reward: {total_reward:.4f}")
            
            metrics = env.get_episode_metrics()
            print(f"\nEpisode metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
            
            env.close()
        else:
            print("No datasets found. Please add a CSV file to the datasets/ folder.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


