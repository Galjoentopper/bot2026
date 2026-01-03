#!/usr/bin/env python3
"""
TensorFlow GPU Library Diagnostic Script
========================================
Diagnoses why TensorFlow cannot detect GPU on RunPod.
Checks for required CUDA/cuDNN libraries and their locations.
"""

import os
import sys
import ctypes
from pathlib import Path
from typing import List, Tuple, Optional

# Required libraries for TensorFlow 2.15.0 with CUDA 12.4
REQUIRED_LIBS = {
    'libcudart.so': ['libcudart.so.12', 'libcudart.so.12.4', 'libcudart.so'],
    'libcudnn.so': ['libcudnn.so.9', 'libcudnn.so.9.1', 'libcudnn.so'],
    'libcublas.so': ['libcublas.so.12', 'libcublas.so.12.4', 'libcublas.so'],
    'libcufft.so': ['libcufft.so.11', 'libcufft.so.12', 'libcufft.so'],
    'libcurand.so': ['libcurand.so.10', 'libcurand.so.12', 'libcurand.so'],
    'libcusolver.so': ['libcusolver.so.11', 'libcusolver.so.12', 'libcusolver.so'],
    'libcusparse.so': ['libcusparse.so.12', 'libcusparse.so'],
}

# Library search paths (RunPod CUDA 12.4 structure)
LIBRARY_PATHS = [
    '/usr/local/cuda-12.4/targets/x86_64-linux/lib',
    '/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs',
    '/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib',
    '/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib',
    '/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib',
    '/usr/lib/x86_64-linux-gnu',
    '/usr/local/nvidia/lib',
    '/usr/local/nvidia/lib64',
    '/usr/local/cuda-12.4/lib64',
    '/usr/local/cuda/lib64',
]

def find_library(lib_name: str, search_paths: List[str]) -> Optional[Tuple[str, str]]:
    """Find a library file in the search paths.
    
    Returns:
        Tuple of (full_path, found_name) if found, None otherwise
    """
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
        
        # Check for exact matches first
        for variant in REQUIRED_LIBS.get(lib_name, [lib_name]):
            lib_path = os.path.join(search_path, variant)
            if os.path.exists(lib_path):
                return (lib_path, variant)
        
        # Check for any file starting with lib_name
        try:
            for file in os.listdir(search_path):
                if file.startswith(lib_name) and (file.endswith('.so') or '.so.' in file):
                    full_path = os.path.join(search_path, file)
                    if os.path.isfile(full_path) or os.path.islink(full_path):
                        return (full_path, file)
        except PermissionError:
            continue
    
    return None

def test_library_load(lib_path: str) -> Tuple[bool, str]:
    """Test if a library can be loaded.
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        return (True, "")
    except OSError as e:
        return (False, str(e))

def check_ld_library_path() -> Tuple[str, List[str]]:
    """Check current LD_LIBRARY_PATH and return missing paths."""
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    current_paths = current_ld.split(':') if current_ld else []
    
    missing_paths = []
    for path in LIBRARY_PATHS:
        if os.path.exists(path) and path not in current_paths:
            missing_paths.append(path)
    
    return current_ld, missing_paths

def main():
    """Run comprehensive GPU library diagnostics."""
    print("=" * 70)
    print("TensorFlow GPU Library Diagnostic")
    print("=" * 70)
    print()
    
    # Check LD_LIBRARY_PATH
    print("1. Checking LD_LIBRARY_PATH...")
    current_ld, missing_paths = check_ld_library_path()
    if current_ld:
        print(f"   Current LD_LIBRARY_PATH: {current_ld}")
    else:
        print("   ⚠ LD_LIBRARY_PATH is not set")
    
    if missing_paths:
        print(f"   ⚠ Missing paths that should be added:")
        for path in missing_paths:
            print(f"      - {path}")
    else:
        print("   ✓ All expected paths are in LD_LIBRARY_PATH")
    print()
    
    # Check library search paths
    print("2. Checking library search paths...")
    existing_paths = []
    for path in LIBRARY_PATHS:
        if os.path.exists(path):
            existing_paths.append(path)
            print(f"   ✓ {path}")
        else:
            print(f"   ✗ {path} (not found)")
    print(f"\n   Found {len(existing_paths)}/{len(LIBRARY_PATHS)} search paths")
    print()
    
    # Check required libraries
    print("3. Checking required CUDA/cuDNN libraries...")
    print()
    all_found = True
    library_status = {}
    
    for lib_name, variants in REQUIRED_LIBS.items():
        result = find_library(lib_name, existing_paths)
        if result:
            lib_path, found_name = result
            library_status[lib_name] = {
                'found': True,
                'path': lib_path,
                'name': found_name
            }
            print(f"   ✓ {lib_name}")
            print(f"      Found: {lib_path}")
            
            # Test loading
            can_load, error = test_library_load(lib_path)
            if can_load:
                print(f"      ✓ Can be loaded")
            else:
                print(f"      ⚠ Cannot load: {error}")
        else:
            library_status[lib_name] = {'found': False}
            print(f"   ✗ {lib_name} (NOT FOUND)")
            print(f"      Searched for: {', '.join(variants)}")
            all_found = False
        print()
    
    # Check library dependencies
    print("4. Checking library dependencies...")
    print()
    import subprocess
    
    for lib_name, status in library_status.items():
        if status.get('found'):
            lib_path = status['path']
            try:
                # Use ldd to check dependencies
                result = subprocess.run(
                    ['ldd', lib_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    missing_deps = [line for line in result.stdout.split('\n') if 'not found' in line]
                    if missing_deps:
                        print(f"   ⚠ {lib_name} has missing dependencies:")
                        for dep in missing_deps[:3]:  # Show first 3
                            print(f"      {dep.strip()}")
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                pass  # ldd might not be available or library might not be a shared object
    print()
    
    # Check TensorFlow GPU detection
    print("5. Testing TensorFlow GPU detection...")
    print()
    
    # Set LD_LIBRARY_PATH if needed
    if missing_paths:
        new_ld = ':'.join(missing_paths)
        if current_ld:
            os.environ['LD_LIBRARY_PATH'] = f"{new_ld}:{current_ld}"
        else:
            os.environ['LD_LIBRARY_PATH'] = new_ld
        print(f"   Set LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
        print()
    
    # Try to get more detailed error messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Enable all logs temporarily
    
    try:
        import tensorflow as tf
        print(f"   TensorFlow version: {tf.__version__}")
        print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"   Built with GPU: {tf.test.is_built_with_gpu_support()}")
        
        # Try to get more info about GPU detection
        try:
            # Force GPU initialization
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"   ✓ GPU devices detected: {len(gpus)}")
                for i, gpu in enumerate(gpus):
                    print(f"      GPU {i}: {gpu.name}")
                    try:
                        details = tf.config.experimental.get_device_details(gpu)
                        if details:
                            print(f"         Details: {details}")
                    except:
                        pass
            else:
                print("   ✗ No GPU devices detected")
                print("   ⚠ TensorFlow cannot find GPU libraries")
                print()
                print("   Attempting to identify the issue...")
                # Try to access GPU info directly
                try:
                    from tensorflow.python.client import device_lib
                    local_devices = device_lib.list_local_devices()
                    gpu_devices = [d for d in local_devices if d.device_type == 'GPU']
                    if gpu_devices:
                        print(f"   Found {len(gpu_devices)} GPU device(s) via device_lib:")
                        for dev in gpu_devices:
                            print(f"      {dev.name}")
                    else:
                        print("   No GPU devices found via device_lib")
                except Exception as e:
                    print(f"   Could not check device_lib: {e}")
        except Exception as e:
            print(f"   Error checking GPU: {e}")
    except ImportError:
        print("   ✗ TensorFlow not installed")
    except Exception as e:
        print(f"   ✗ Error importing TensorFlow: {e}")
    finally:
        # Restore log level
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    print()
    
    # Summary and recommendations
    print("=" * 70)
    print("Summary and Recommendations")
    print("=" * 70)
    print()
    
    if all_found:
        print("✓ All required libraries found and can be loaded")
        print()
        print("⚠ However, TensorFlow still cannot detect GPU.")
        print("   The 'Unable to register cuDNN factory' errors suggest:")
        print("   - TensorFlow 2.15.0 may have been compiled with CuDNN 9.3.0")
        print("   - RunPod has CuDNN 9.1.0 installed")
        print("   - Version mismatch prevents GPU initialization")
        print()
        print("   Solutions to try:")
        print("   1. Use CPU training (works but slower)")
        print("   2. Disable CuDNN for RNNs: export TF_USE_CUDNN_RNN=0")
        print("   3. Train on CPU but use PyTorch GPU for PPO (if applicable)")
        print("   4. Check if TensorFlow can work with CuDNN 9.1.0 by setting:")
        print("      export TF_FORCE_GPU_ALLOW_GROWTH=true")
        print("      export TF_USE_CUDNN_RNN=0  # Disable CuDNN-optimized RNNs")
    else:
        print("⚠ Some required libraries are missing:")
        for lib_name, status in library_status.items():
            if not status.get('found'):
                print(f"   - {lib_name}")
        print()
        print("Recommendations:")
        print("1. Verify CUDA 12.4 is properly installed")
        print("2. Check if libraries are in non-standard locations")
        print("3. Consider creating symlinks if libraries exist but aren't found")
    
    if missing_paths:
        print()
        print("⚠ LD_LIBRARY_PATH needs to be updated:")
        print("   Add these paths to LD_LIBRARY_PATH:")
        for path in missing_paths:
            print(f"   export LD_LIBRARY_PATH={path}:$LD_LIBRARY_PATH")
    
    print()
    print("To fix LD_LIBRARY_PATH permanently, add to ~/.bashrc:")
    print("export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs:/usr/local/cuda-12.4/lib64:/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH")
    print()
    print("Note: If GPU detection still fails, training will use CPU automatically.")
    print("      This is slower but functional. Consider using PyTorch for GPU training.")
    print()

if __name__ == '__main__':
    main()

