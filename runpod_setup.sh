#!/bin/bash
# RunPod Setup Script - Run this first to verify everything is ready

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
ls -la /workspace

# Check GPU
echo ""
echo "3. GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Check Python
echo ""
echo "4. Python version:"
python3 --version

# Check if project exists
echo ""
echo "5. Checking for Bot 2026 project:"
if [ -d "/workspace/Bot 2026" ]; then
    echo "✓ Found: /workspace/Bot 2026"
    ls -la "/workspace/Bot 2026" | head -10
elif [ -d "/workspace" ]; then
    echo "⚠ Bot 2026 not found in /workspace"
    echo "  Contents of /workspace:"
    ls -la /workspace
else
    echo "⚠ /workspace not found"
fi

# Check PyTorch
echo ""
echo "6. Checking PyTorch:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "⚠ PyTorch not installed"

# Check TensorFlow
echo ""
echo "7. Checking TensorFlow:"
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')" 2>/dev/null || echo "⚠ TensorFlow not installed"

echo ""
echo "=========================================="
echo "Setup check complete!"
echo "=========================================="

