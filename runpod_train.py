#!/usr/bin/env python3
"""
RunPod Training Script - Complete Training Pipeline
Adapted from colab_training_notebook.ipynb for RunPod
"""

import os
import sys
from pathlib import Path
import subprocess

# ============================================
# STEP 1: Setup Paths
# ============================================
print("=" * 60)
print("STEP 1: Setting up paths...")
print("=" * 60)

# CONFIGURATION: Set your dataset name here
DATASET_NAME = "ADA-EUR_1H_20240101-20251231"  # CHANGE THIS TO YOUR DATASET

# Find project path (RunPod uses /workspace for volumes)
possible_paths = [
    Path('/workspace/Bot 2026'),
    Path('/workspace'),
    Path('/runpod-volume/Bot 2026'),
    Path('/data/Bot 2026'),
    Path.cwd() / 'Bot 2026',
    Path.cwd(),
]

PROJECT_PATH = None
for path in possible_paths:
    if path.exists():
        # Check if it's the project root (has 'PPO approach' folder)
        if (path / 'PPO approach').exists():
            PROJECT_PATH = path
            break
        # Or if we're already in the project root
        elif path.name == 'Bot 2026' and (path / 'PPO approach').exists():
            PROJECT_PATH = path
            break

# If not found, use current directory
if PROJECT_PATH is None:
    PROJECT_PATH = Path.cwd()
    print(f"⚠ Using current directory: {PROJECT_PATH}")
    if not (PROJECT_PATH / 'PPO approach').exists():
        print("⚠ Warning: 'PPO approach' folder not found!")
        print(f"   Current directory contents: {list(PROJECT_PATH.iterdir())}")

# Add paths to sys.path
if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))
if str(PROJECT_PATH / 'PPO approach') not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH / 'PPO approach'))

# Change to project directory
os.chdir(PROJECT_PATH)

print(f"✓ Project path: {PROJECT_PATH}")
print(f"✓ Dataset name: {DATASET_NAME}")
print(f"✓ Current directory: {os.getcwd()}")

# ============================================
# STEP 2: Verify/Install Dependencies
# ============================================
print("\n" + "=" * 60)
print("STEP 2: Verifying dependencies...")
print("=" * 60)

# Check TensorFlow
try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
except ImportError:
    print("⚠ TensorFlow not found - installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tensorflow>=2.13.0"])
    import tensorflow as tf
    print(f"✓ TensorFlow installed: {tf.__version__}")

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
except ImportError:
    print("⚠ PyTorch not found - installing...")
    # Use CUDA 12.1 (compatible with 12.4)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ])
    import torch
    print(f"✓ PyTorch installed: {torch.__version__}")

# Check stable-baselines3
try:
    import stable_baselines3
    print(f"✓ stable-baselines3: {stable_baselines3.__version__}")
except ImportError:
    print("⚠ stable-baselines3 not found - installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "stable-baselines3[extra]>=2.0.0"])
    import stable_baselines3
    print(f"✓ stable-baselines3 installed: {stable_baselines3.__version__}")

# Install other dependencies if needed
other_deps = ['pandas>=1.5.0', 'numpy>=1.23.0', 'scikit-learn>=1.2.0', 
              'matplotlib>=3.6.0', 'tqdm>=4.65.0', 'gymnasium>=0.28.0', 
              'tensorboard>=2.13.0']
print("\nChecking other dependencies...")
for dep in other_deps:
    try:
        __import__(dep.split('>=')[0].replace('-', '_'))
    except ImportError:
        print(f"  Installing {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])

# ============================================
# STEP 3: Verify GPU
# ============================================
print("\n" + "=" * 60)
print("STEP 3: Verifying GPU...")
print("=" * 60)

import torch
import tensorflow as tf

# PyTorch GPU
if torch.cuda.is_available():
    print(f"✓ PyTorch GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠ No PyTorch GPU detected!")
    print("  Training will be VERY slow on CPU")

# TensorFlow GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ TensorFlow GPU: {len(gpus)} GPU(s) available")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
else:
    print("⚠ No TensorFlow GPU detected!")

if not torch.cuda.is_available() and not gpus:
    print("\n" + "⚠" * 30)
    print("⚠ WARNING: No GPU detected!")
    print("   Training will be VERY slow on CPU.")
    print("⚠" * 30)
    response = input("\nContinue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(1)

# ============================================
# STEP 4: Verify Dataset
# ============================================
print("\n" + "=" * 60)
print("STEP 4: Verifying dataset...")
print("=" * 60)

import pandas as pd

datasets_path = PROJECT_PATH / 'datasets'
if not datasets_path.exists():
    print(f"⚠ Datasets directory not found: {datasets_path}")
    print("  Creating directory...")
    datasets_path.mkdir(parents=True, exist_ok=True)

dataset_file = datasets_path / f"{DATASET_NAME}.csv"

if not dataset_file.exists():
    # Try to find with partial match
    matches = list(datasets_path.glob(f"*{DATASET_NAME}*.csv"))
    if matches:
        dataset_file = matches[0]
        print(f"⚠ Found similar file: {dataset_file.name}")
    else:
        print(f"❌ Dataset not found: {DATASET_NAME}.csv")
        print(f"\nAvailable datasets in {datasets_path}:")
        csv_files = list(datasets_path.glob("*.csv"))
        if csv_files:
            for f in sorted(csv_files):
                print(f"  - {f.name}")
        else:
            print("  (No CSV files found)")
        print(f"\nPlease upload your dataset to: {datasets_path}")
        sys.exit(1)

print(f"✓ Dataset found: {dataset_file.name}")

# Load and display dataset info
try:
    df = pd.read_csv(dataset_file)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
    
    # Check required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n⚠ WARNING: Missing required columns: {missing_cols}")
        print("   Dataset may not work correctly for training")
    else:
        print(f"  ✓ All required columns present")
    
    # Display date range if timestamp exists
    if 'timestamp' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            print(f"\nDate Range:")
            print(f"  Start: {df['datetime'].min()}")
            print(f"  End: {df['datetime'].max()}")
            print(f"  Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")
        except:
            print("  (Could not parse timestamps)")
    
except Exception as e:
    print(f"\n❌ ERROR loading dataset: {e}")
    raise

# ============================================
# STEP 5: Train Prediction Models
# ============================================
print("\n" + "=" * 60)
print("STEP 5: Training prediction models...")
print("=" * 60)

from train_models import train_all_models, load_config

# Load configuration
config_path = PROJECT_PATH / 'training_config.txt'
if config_path.exists():
    print(f"Loading config from: {config_path}")
    config = load_config(str(config_path))
else:
    # Try config.txt, but skip if it's JSON format
    config_path_alt = PROJECT_PATH / 'config.txt'
    if config_path_alt.exists():
        # Check if it's JSON (starts with {)
        with open(config_path_alt, 'r') as f:
            first_char = f.read(1).strip()
        if first_char == '{':
            print("⚠ config.txt is JSON format (for crypto downloader), skipping")
            print("  Using default training configuration")
            config = None
        else:
            config_path = config_path_alt
            print(f"Loading config from: {config_path}")
            config = load_config(str(config_path))
    else:
        print("⚠ Config file not found, using defaults")
        config = None

# Train all models
print(f"\nTraining models on dataset: {DATASET_NAME}")
print("Models: LSTM, GRU, BiLSTM, DLSTM")
print("Task: Classification (Fall, Stationary, Rise)")
print("-" * 60)

try:
    results = train_all_models(
        datasets_dir=str(PROJECT_PATH / 'datasets'),
        config=config,
        task='classification',
        models=['lstm', 'gru', 'bilstm', 'dlstm'],
        specific_dataset=DATASET_NAME,
        use_ensemble=False
    )
    
    if results:
        print("\n" + "=" * 60)
        print("✓ All prediction models trained successfully!")
        print("=" * 60)
        
        # Print summary
        print("\nTraining Summary:")
        for result in results:
            model_name = result['model_name']
            metrics = result['metrics']
            print(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.6f}")
                else:
                    print(f"  {metric}: {value}")
    else:
        print("\n⚠ No models were trained. Check dataset name and paths.")
        
except Exception as e:
    print(f"\n❌ ERROR during training: {e}")
    import traceback
    traceback.print_exc()
    raise

# ============================================
# STEP 6: Validate Models Before PPO Training
# ============================================
print("\n" + "=" * 60)
print("STEP 6: Validating models before PPO training...")
print("=" * 60)

import numpy as np

# Import prediction wrapper
from prediction_wrapper import PredictionModel, EnsemblePredictionModel

# Test loading each individual model
print("\n1. Testing Individual Model Loading:")
print("-" * 60)
models_to_test = ['lstm', 'gru', 'bilstm', 'dlstm']
loaded_models = {}

for model_name in models_to_test:
    try:
        model = PredictionModel(model_name, DATASET_NAME)
        if model.load():
            print(f"  ✓ {model_name.upper()} loaded successfully")
            loaded_models[model_name] = model
        else:
            print(f"  ✗ {model_name.upper()} failed to load")
    except Exception as e:
        print(f"  ✗ {model_name.upper()} error: {e}")

# Test ensemble loading
print("\n2. Testing Ensemble Loading:")
print("-" * 60)
try:
    ensemble = EnsemblePredictionModel(DATASET_NAME, list(loaded_models.keys()))
    if ensemble.load():
        print(f"  ✓ Ensemble loaded with {len(ensemble.models)} model(s)")
    else:
        print(f"  ✗ Ensemble failed to load")
        raise RuntimeError("Ensemble loading failed")
except Exception as e:
    print(f"  ✗ Ensemble error: {e}")
    raise

# Verify model inference works
print("\n3. Testing Model Inference:")
print("-" * 60)
try:
    # Create dummy sequence for testing
    seq_len = ensemble.sequence_length
    # Get number of features
    if ensemble.models and hasattr(ensemble.models[0], 'feature_names') and ensemble.models[0].feature_names:
        n_features = len(ensemble.models[0].feature_names)
    elif ensemble.feature_scaler:
        if hasattr(ensemble.feature_scaler, 'n_features_in_'):
            n_features = ensemble.feature_scaler.n_features_in_
        elif hasattr(ensemble.feature_scaler, 'feature_names_in_'):
            n_features = len(ensemble.feature_scaler.feature_names_in_)
        else:
            n_features = 27  # Default
    else:
        n_features = 27  # Default fallback
    
    dummy_seq = np.random.randn(seq_len, n_features)
    print(f"  Test sequence shape: {dummy_seq.shape}")
    
    # Test single-step prediction
    pred_class, confidence, probs = ensemble.predict(dummy_seq)
    print(f"  ✓ Single-step prediction:")
    print(f"    Class: {pred_class} ({['Fall', 'Stationary', 'Rise'][pred_class]})")
    print(f"    Confidence: {confidence:.4f}")
    print(f"    Probabilities: Fall={probs[0]:.3f}, Stationary={probs[1]:.3f}, Rise={probs[2]:.3f}")
    
except Exception as e:
    print(f"  ✗ Inference test failed: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n✓ Model validation completed - ready for PPO training")

# ============================================
# STEP 7: Train PPO Agent
# ============================================
print("\n" + "=" * 60)
print("STEP 7: Training PPO agent...")
print("=" * 60)

# Change to PPO approach directory
ppo_path = PROJECT_PATH / 'PPO approach'
os.chdir(ppo_path)
sys.path.insert(0, str(ppo_path))

from train_ppo_agent import train_ppo, load_config

# Load PPO configuration
ppo_config_path = ppo_path / 'ppo_config.txt'
print(f"\nLoading PPO config from: {ppo_config_path}")

if not ppo_config_path.exists():
    print("⚠ PPO config file not found, using defaults")
    ppo_config = load_config(None)
else:
    ppo_config = load_config(str(ppo_config_path))

# Update dataset name in config if needed
if ppo_config['models']['dataset'] != DATASET_NAME:
    print(f"\n⚠ Updating dataset name in config: {ppo_config['models']['dataset']} -> {DATASET_NAME}")
    ppo_config['models']['dataset'] = DATASET_NAME

# Verify PPO config matches prediction model settings
print(f"\nConfiguration Check:")
print(f"  Dataset: {ppo_config['models']['dataset']}")
print(f"  Sequence length: {ppo_config['environment']['sequence_length']}")
print(f"  Prediction model: {ppo_config['models']['prediction_model']}")

# Configure prediction horizons
prediction_horizons = ppo_config['models'].get('prediction_horizons', [1, 2, 3, 5, 10])
print(f"\nPrediction Horizons Configuration:")
print(f"  Horizons: {prediction_horizons}")

# Train PPO agent
print(f"\n{'='*60}")
print("Starting PPO Training...")
print("="*60)

try:
    model = train_ppo(
        model_type=ppo_config['models']['prediction_model'],
        dataset=ppo_config['models']['dataset'],
        timesteps=ppo_config['training']['total_timesteps'],
        config_path=str(ppo_config_path) if ppo_config_path.exists() else None,
        resume=True,  # Resume from checkpoint if exists
    )
    
    if model is not None:
        print("\n" + "="*60)
        print("✓ PPO training completed successfully!")
        print("="*60)
        
        # Display training summary
        from colab_utils import get_ppo_models_path, get_checkpoints_path
        final_model_path = get_ppo_models_path() / f"ppo_{ppo_config['models']['prediction_model']}_{DATASET_NAME}.zip"
        if final_model_path.exists():
            size_mb = final_model_path.stat().st_size / (1024 * 1024)
            print(f"\nFinal model saved: {final_model_path.name} ({size_mb:.2f} MB)")
        
        checkpoint_path = get_checkpoints_path() / f"{ppo_config['models']['prediction_model']}_{DATASET_NAME}"
        if checkpoint_path.exists():
            checkpoints = list(checkpoint_path.glob("*.zip"))
            print(f"Checkpoints available: {len(checkpoints)}")
    else:
        print("\n⚠ PPO training returned None - check for errors above")
        
except Exception as e:
    print(f"\n❌ Error during PPO training: {e}")
    import traceback
    traceback.print_exc()
    raise

# ============================================
# DONE!
# ============================================
print("\n" + "=" * 60)
print("✓ TRAINING COMPLETE!")
print("=" * 60)
print(f"\nAll models saved to: {PROJECT_PATH}")
print(f"  Prediction models: {PROJECT_PATH / 'models'}")
print(f"  PPO models: {PROJECT_PATH / 'PPO approach' / 'ppo_models'}")
print(f"  Checkpoints: {PROJECT_PATH / 'PPO approach' / 'checkpoints'}")
print(f"  Results: {PROJECT_PATH / 'results'}")
print("\n" + "=" * 60)

