# Complete Training Pipeline Overview

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTRY POINT: runpod_main.py                  │
│  - Pre-flight validation (disk, memory, GPU, network, dataset)   │
│  - Logging setup                                                 │
│  - Calls runpod_train.py                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MAIN TRAINING: runpod_train.py                │
│  Step 1: Setup paths & environment                              │
│  Step 2: Verify dependencies                                    │
│  Step 3: Verify GPU                                             │
│  Step 4: Verify dataset                                         │
│  Step 5: Train prediction models                                │
│  Step 6: Validate models                                        │
│  Step 7: Train PPO agent                                        │
│  Step 8: Push to GitHub (optional)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Pipeline Flow

### Phase 1: Entry & Validation (`runpod_main.py`)

**Purpose**: Orchestrate the entire training process with safety checks

**Steps**:
1. **Parse Arguments**
   - `--dataset`: Dataset name (required)
   - `--verbose`: Debug logging
   - `--skip-validation`: Skip pre-flight checks
   - `--no-push`: Skip GitHub push
   - `--resume`: Resume from checkpoint

2. **Pre-flight Validation** (`preflight_validation()`)
   - ✅ Disk space check (min 10GB)
   - ✅ Memory check (min 8GB)
   - ✅ GPU detection (PyTorch CUDA)
   - ✅ Network connectivity (GitHub)
   - ✅ Dataset file exists
   - ✅ Dependencies installed (torch, stable_baselines3, pandas, numpy)
   - ✅ Write permissions (models/, checkpoints/)

3. **Execute Training**
   - Set environment variables (`DATASET_NAME`, `RESUME_TRAINING`, `RUNPOD_PUSH_TO_GITHUB`)
   - Call `runpod_train.py` as subprocess
   - Monitor exit code and log results

---

### Phase 2: Main Training Pipeline (`runpod_train.py`)

#### STEP 1: Setup Paths & Environment
- Detect project path (RunPod workspace)
- Setup Python paths for imports
- Initialize utility functions (`get_datasets_path()`, `get_ppo_path()`, etc.)
- Set `DATASET_NAME` from environment or default

#### STEP 2: Verify Dependencies
- Check `requirements.txt` exists
- Verify key packages installed:
  - `torch` (PyTorch)
  - `stable-baselines3` (PPO)
  - `pandas`, `numpy` (data processing)
  - `gymnasium` (RL environment)
- Install missing packages if needed

#### STEP 3: Verify GPU
- Check PyTorch CUDA availability
- Display GPU name and memory
- Warn if no GPU (training will be slow)

#### STEP 4: Verify Dataset
- Load dataset CSV file
- Validate required columns: `timestamp, open, high, low, close, volume`
- Display dataset info:
  - Row count
  - Date range
  - Column names

#### STEP 5: Train Prediction Models (`train_models.py`)

**Current Behavior** (before ensemble changes):
- Check if all 4 models exist: `lstm, gru, bilstm, dlstm`
- If all exist → Skip training
- If missing → Train missing models

**After Ensemble Changes**:
- Read `ensemble_models` from PPO config (e.g., `dlstm,bilstm`)
- Only check for and train ensemble models
- Skip LSTM and GRU

**Training Process** (`train_all_models()`):
1. Load dataset CSV
2. Preprocess data:
   - Convert timestamps
   - Add technical indicators (RSI, MACD, Bollinger Bands)
   - Create sequences (lookback window = 60)
   - Generate classification labels (Fall/Stationary/Rise)
3. For each model type:
   - Create model architecture (LSTM/GRU/BiLSTM/DLSTM)
   - Train with PyTorch:
     - Batch size: 1024 (auto-adjusted for GPU)
     - Epochs: 150 (with early stopping)
     - Optimizer: Adam
     - Loss: Cross-entropy (classification)
   - Save model: `{model}_{dataset}_classification.pth`
   - Save scaler: `scaler_{dataset}.pkl`
   - Generate plots: confusion matrix, training curves
   - Save metrics: accuracy, precision, recall, F1

**Output Files**:
- Models: `models/{model}_{dataset}_classification.pth`
- Scalers: `scalers/scaler_{dataset}.pkl`
- Results: `results/training_{model}_{dataset}.png`
- History: `results/history_{model}_{dataset}.json`

#### STEP 6: Validate Models Before PPO Training

**Purpose**: Ensure models can be loaded and make predictions

1. **Test Individual Models**
   - Load each model: `PredictionModel(model_name, dataset_name)`
   - Verify model loads successfully
   - Check GPU/CPU device assignment

2. **Test Ensemble Loading** (if using ensemble)
   - Load ensemble: `EnsemblePredictionModel(dataset_name, model_names)`
   - Verify all models in ensemble load
   - Check ensemble can make predictions

3. **Test Model Inference**
   - Create dummy sequence (60 timesteps, 26 features)
   - Run prediction through model(s)
   - Verify output format: (class, confidence, probabilities)

#### STEP 7: Train PPO Agent (`PPO approach/train_ppo_agent.py`)

**Configuration Loading**:
- Load `ppo_config.txt`
- Read settings:
  - `prediction_model`: `dlstm` or `ensemble`
  - `ensemble_models`: `dlstm,bilstm` (if ensemble)
  - `total_timesteps`: 150,000
  - `batch_size`: 2048
  - `n_steps`: 1024
  - `n_epochs`: 20

**Environment Setup**:
1. **Load Prediction Models**
   - If `prediction_model = ensemble`:
     - Load `EnsemblePredictionModel` with specified models
   - If `prediction_model = dlstm`:
     - Load single `PredictionModel`

2. **Create Trading Environment** (`TradingEnv`)
   - Load dataset CSV
   - Split train/test (80/20)
   - Initialize portfolio tracker
   - Setup reward calculator
   - Define observation space (22 features):
     - Prediction features (5-17): class, confidence, probabilities
     - Price features (5): price, change, volume, RSI, MACD
     - Portfolio state (4): position, P&L, cash, holding time
   - Define action space (9 discrete actions):
     - 0: Hold
     - 1-3: Buy Small/Medium/Large
     - 4-6: Sell Small/Medium/Large
     - 7: Close Position
     - 8: Reverse Position

3. **Create Vectorized Environments**
   - Try `SubprocVecEnv` (16 parallel envs) - true parallelism
   - Fallback to `DummyVecEnv` (16 workers) - sequential but batched
   - Each environment runs independently, collecting data in parallel

**PPO Training** (`ppo_trading_agent.py`):
1. **Create PPO Agent**
   - Policy: MLP (Multi-Layer Perceptron)
   - Network: [1280, 640] hidden layers
   - Device: CUDA (GPU)
   - Learning rate: 0.0003

2. **Training Loop** (`train_with_checkpoints()`):
   ```
   For each update cycle:
     a. Collect data (n_steps = 1024 across all envs)
        - Each env runs for ~64-85 steps
        - Collect: states, actions, rewards, dones
     b. Compute advantages (GAE - Generalized Advantage Estimation)
     c. Train PPO (n_epochs = 20):
        - Batch size: 2048
        - Update policy and value networks
        - Clip policy updates (PPO clipping)
     d. Evaluate on test set (every eval_freq steps)
     e. Save checkpoint (every checkpoint_freq steps)
   ```

3. **Checkpointing**:
   - Save every 25,000 timesteps
   - Location: `PPO approach/checkpoints/{model}_{dataset}/`
   - Files: `checkpoint_{timesteps}.zip`

4. **Evaluation**:
   - Every 5,000 timesteps
   - Run 5 episodes on test set
   - Log metrics: reward, episode length, Sharpe ratio

**Output Files**:
- Final model: `PPO approach/models/ppo_{model}_{dataset}.zip`
- Checkpoints: `PPO approach/checkpoints/{model}_{dataset}/*.zip`
- TensorBoard logs: `PPO approach/logs/PPO_0/`

#### STEP 8: Push Models to GitHub (Optional)

**If `RUNPOD_PUSH_TO_GITHUB=true`**:
1. Collect files to push:
   - Prediction models (`.pth` files)
   - PPO models (`.zip` files)
   - Scalers (`.pkl` files)
   - Results (`.png`, `.json` files)
   - Logs (`.log` files)
   - TensorBoard events

2. Git operations (`runpod_github.py`):
   - Configure Git for large files (increase buffer, timeout)
   - Add files to staging
   - Commit with timestamp
   - Push to GitHub (main branch)

---

## Data Flow Diagram

```
Dataset CSV
    │
    ▼
┌─────────────────────────────────────┐
│  Data Preprocessing                  │
│  - Load CSV                          │
│  - Add technical indicators          │
│  - Create sequences (60 timesteps)   │
│  - Generate labels (Fall/Stat/Rise) │
└──────────────┬──────────────────────┘
               │
               ├──────────────────────────────┐
               │                              │
               ▼                              ▼
    ┌──────────────────────┐    ┌──────────────────────┐
    │  Prediction Models   │    │  Trading Environment │
    │  (LSTM/GRU/BiLSTM/   │    │  - Uses predictions  │
    │   DLSTM/Ensemble)    │───▶│  - Portfolio state   │
    │                      │    │  - Price features    │
    │  Output:             │    │                      │
    │  - Class (0/1/2)     │    │  Observation: 22 dims│
    │  - Confidence        │    │  Action: 9 discrete   │
    │  - Probabilities     │    │  Reward: P&L based   │
    └──────────────────────┘    └──────────┬───────────┘
                                            │
                                            ▼
                                  ┌──────────────────────┐
                                  │  PPO Agent           │
                                  │  - Policy network    │
                                  │  - Value network     │
                                  │  - Learns to trade   │
                                  │                      │
                                  │  Training:           │
                                  │  - Collect episodes  │
                                  │  - Update policy     │
                                  │  - Maximize reward   │
                                  └──────────────────────┘
```

---

## Key Components

### 1. Prediction Models (`train_models.py`, `pytorch_train.py`)

**Architectures**:
- **LSTM**: Standard LSTM (3-4 layers, 256-512 units)
- **GRU**: Gated Recurrent Unit (faster, smaller)
- **BiLSTM**: Bidirectional LSTM (captures past & future)
- **DLSTM**: Deep LSTM with trend decomposition

**Training**:
- Task: 3-class classification (Fall, Stationary, Rise)
- Input: 60 timesteps × 26 features
- Output: Class probabilities [P(Fall), P(Stat), P(Rise)]
- Metrics: Accuracy, Precision, Recall, F1

### 2. Trading Environment (`PPO approach/trading_env.py`)

**Observation Space** (22 dimensions):
- Prediction features (5-17): From prediction models
- Price features (5): Normalized price, change, volume, RSI, MACD
- Portfolio state (4): Position, unrealized P&L, cash ratio, holding time

**Action Space** (9 discrete actions):
- 0: Hold (do nothing)
- 1-3: Buy (25%, 50%, 100% of cash)
- 4-6: Sell (25%, 50%, 100% of position)
- 7: Close position
- 8: Reverse position

**Reward Function** (`PPO approach/reward_functions.py`):
- Profit-based: Scaled by `profit_scale` (100x)
- Transaction cost penalty
- Drawdown penalty
- Sharpe ratio bonus
- Clipped to [-10, 10]

### 3. PPO Agent (`PPO approach/ppo_trading_agent.py`)

**Architecture**:
- Policy: MLP with [1280, 640] hidden layers
- Value: Separate value network
- Device: CUDA (GPU)

**Hyperparameters**:
- Learning rate: 0.0003
- Batch size: 2048
- n_steps: 1024 (steps per update)
- n_epochs: 20 (training epochs per update)
- Gamma: 0.99 (discount factor)
- GAE lambda: 0.95
- Clip range: 0.2

**Training Process**:
1. Collect trajectories (1024 steps across 16 envs)
2. Compute advantages (GAE)
3. Train on batches (20 epochs, batch size 2048)
4. Update policy and value networks
5. Repeat until 150,000 total timesteps

---

## Configuration Files

### `training_config.txt`
- Model architecture (units, layers, dropout)
- Training parameters (batch size, epochs, learning rate)
- Data settings (sequence length, technical indicators)
- Classification settings (smoothing_k, threshold_delta)

### `PPO approach/ppo_config.txt`
- Environment settings (transaction cost, initial capital)
- PPO hyperparameters (learning rate, batch size, n_steps)
- Reward configuration (profit scale, drawdown penalty)
- Model selection (prediction_model, ensemble_models)
- Training settings (total_timesteps, checkpoint_freq)

---

## Execution Flow

### Command Line:
```bash
python runpod_main.py --dataset ADA-EUR_1H_20240101-20251231 --verbose
```

### What Happens:
1. **runpod_main.py** validates system and calls **runpod_train.py**
2. **runpod_train.py** executes 8 steps sequentially
3. **train_models.py** trains prediction models (if needed)
4. **train_ppo_agent.py** trains PPO agent
5. Models saved to disk
6. Optional: Push to GitHub

### Time Estimates:
- Prediction models: ~30-60 minutes (4 models) or ~15-30 minutes (2 models with ensemble)
- PPO training: ~15-20 minutes (150K timesteps with optimizations)
- Total: ~45-80 minutes for full pipeline

---

## File Structure

```
bot2026/
├── runpod_main.py              # Entry point, validation
├── runpod_train.py              # Main training pipeline
├── train_models.py              # Prediction model training
├── pytorch_train.py             # PyTorch training utilities
├── training_config.txt          # Prediction model config
├── models/                      # Trained prediction models
├── scalers/                     # Feature scalers
├── results/                     # Training plots, metrics
├── datasets/                    # CSV datasets
├── logs/                        # Training logs
│
└── PPO approach/
    ├── train_ppo_agent.py       # PPO training
    ├── ppo_trading_agent.py     # PPO agent implementation
    ├── trading_env.py            # Trading environment
    ├── prediction_wrapper.py    # Model loading/prediction
    ├── reward_functions.py      # Reward calculation
    ├── ppo_config.txt           # PPO configuration
    ├── models/                  # Trained PPO models
    ├── checkpoints/             # Training checkpoints
    └── logs/                    # TensorBoard logs
```

---

## Key Features

### 1. Smart Training
- Checks if models exist before training
- Skips unnecessary training steps
- Resumes from checkpoints

### 2. GPU Optimization
- Auto-adjusts batch sizes for GPU
- Uses mixed precision (AMP) when available
- Maximizes GPU utilization (85-95%)

### 3. Parallel Processing
- 16 parallel environments for data collection
- Vectorized environment execution
- Faster training with parallel data collection

### 4. Robust Error Handling
- Pre-flight validation catches issues early
- Graceful fallbacks (DummyVecEnv if SubprocVecEnv fails)
- Comprehensive logging

### 5. Modular Design
- Each component is independent
- Easy to modify individual steps
- Configurable via text files

---

## Future Enhancements (From Plan)

1. **Ensemble Support**: Only train ensemble models (DLSTM + BiLSTM)
2. **Hyperparameter Tuning**: Automated search for optimal parameters
3. **Better Parallelization**: Fix SubprocVecEnv multiprocessing
4. **Advanced Ensembles**: Weighted voting, stacking

---

*This pipeline is designed for RunPod GPU instances but works on any system with PyTorch and CUDA support.*


