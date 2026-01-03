#!/bin/bash
# RunPod Setup Script - Run this first to verify everything is ready

set -e  # Exit on error

echo "=========================================="
echo "RunPod Setup & Verification"
echo "=========================================="

# Check current directory
echo ""
echo "1. Current directory:"
pwd

# Check workspace
echo ""
echo "2. Workspace contents:"
if [ -d "/workspace" ]; then
    cd /workspace
    ls -la | head -20
else
    echo "⚠ /workspace not found, using current directory"
fi

# Check GPU
echo ""
echo "3. GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo "✓ GPU detected"
else
    echo "⚠ nvidia-smi not found - GPU may not be available"
fi

# Check Python
echo ""
echo "4. Python version:"
python3 --version
pip3 --version

# Check if project exists
echo ""
echo "5. Checking project structure:"
if [ -f "runpod_train.py" ]; then
    echo "✓ Found runpod_train.py"
else
    echo "⚠ runpod_train.py not found in current directory"
fi

if [ -d "PPO approach" ]; then
    echo "✓ Found PPO approach directory"
else
    echo "⚠ PPO approach directory not found"
fi

if [ -d "datasets" ]; then
    echo "✓ Found datasets directory"
    DATASET_COUNT=$(find datasets -name "*.csv" 2>/dev/null | wc -l)
    echo "  CSV files: $DATASET_COUNT"
else
    echo "⚠ datasets directory not found"
    mkdir -p datasets
    echo "  Created datasets directory"
fi

# Check PyTorch
echo ""
echo "6. Checking PyTorch:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "⚠ PyTorch not installed"

# Check TensorFlow
echo ""
echo "7. Checking TensorFlow:"
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); gpus = tf.config.list_physical_devices('GPU'); print(f'GPU available: {len(gpus) > 0}'); [print(f'  {gpu.name}') for gpu in gpus]" 2>/dev/null || echo "⚠ TensorFlow not installed"

# Check stable-baselines3
echo ""
echo "8. Checking stable-baselines3:"
python3 -c "import stable_baselines3; print(f'stable-baselines3: {stable_baselines3.__version__}')" 2>/dev/null || echo "⚠ stable-baselines3 not installed"

# Check other dependencies
echo ""
echo "9. Checking other dependencies:"
python3 -c "import pandas, numpy, sklearn, gymnasium; print('✓ Core dependencies available')" 2>/dev/null || echo "⚠ Some dependencies missing"

echo ""
echo "=========================================="
echo "Setup check complete!"
echo "=========================================="
