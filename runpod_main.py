#!/usr/bin/env python3
"""
RunPod Main Entry Point
========================
Single script to run everything: setup → training → push to GitHub

Features:
- Pre-flight validation (disk space, memory, GPU, network, dataset)
- Structured logging with file and console output
- Progress monitoring (ETA, GPU utilization, system resources)
- Cleanup utilities for old checkpoints
- Configurable via command-line arguments
"""

import os
import sys

# PyTorch handles GPU detection automatically - no manual setup needed!
# CUDA_VISIBLE_DEVICES can still be set if needed
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Ensure GPU 0 is visible

import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Setup logging before other imports
def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    """Setup structured logging with file and console output."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('runpod_main')
    logger.info(f"Logging to: {log_file}")
    return logger


def check_disk_space(path: Path, min_gb: float = 10.0) -> tuple[bool, float]:
    """Check available disk space."""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024 ** 3)
    return free_gb >= min_gb, free_gb


def check_memory(min_gb: float = 8.0) -> tuple[bool, Optional[float]]:
    """Check available memory."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        return available_gb >= min_gb, available_gb
    except ImportError:
        return True, None  # Skip check if psutil not available


def check_gpu() -> tuple[bool, Optional[str]]:
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, gpu_name
        return False, None
    except ImportError:
        return False, None


def check_network() -> bool:
    """Check network connectivity."""
    try:
        import socket
        socket.create_connection(("github.com", 443), timeout=5)
        return True
    except:
        return False


def validate_dataset(dataset_path: Path, dataset_name: str) -> tuple[bool, Optional[Path]]:
    """Validate dataset file exists and is readable."""
    dataset_file = dataset_path / f"{dataset_name}.csv"
    
    if not dataset_file.exists():
        # Try partial match
        matches = list(dataset_path.glob(f"*{dataset_name}*.csv"))
        if matches:
            return True, matches[0]
        return False, None
    
    # Check if readable
    try:
        import pandas as pd
        pd.read_csv(dataset_file, nrows=1)
        return True, dataset_file
    except:
        return False, dataset_file


def preflight_validation(
    project_path: Path,
    dataset_name: str,
    logger: logging.Logger,
    skip_network: bool = False
) -> bool:
    """Run pre-flight validation checks."""
    logger.info("=" * 60)
    logger.info("Pre-flight Validation")
    logger.info("=" * 60)
    
    all_checks_passed = True
    
    # Disk space
    logger.info("Checking disk space...")
    has_space, free_gb = check_disk_space(project_path, min_gb=10.0)
    if has_space:
        logger.info(f"  ✓ Disk space: {free_gb:.2f} GB free")
    else:
        logger.warning(f"  ✗ Disk space: Only {free_gb:.2f} GB free (need at least 10 GB)")
        all_checks_passed = False
    
    # Memory
    logger.info("Checking memory...")
    has_memory, available_gb = check_memory(min_gb=8.0)
    if has_memory:
        if available_gb:
            logger.info(f"  ✓ Memory: {available_gb:.2f} GB available")
        else:
            logger.info("  ✓ Memory: Check skipped (psutil not available)")
    else:
        logger.warning(f"  ✗ Memory: Only {available_gb:.2f} GB available (need at least 8 GB)")
        all_checks_passed = False
    
    # GPU
    logger.info("Checking GPU...")
    has_gpu, gpu_name = check_gpu()
    if has_gpu:
        logger.info(f"  ✓ GPU: {gpu_name}")
    else:
        logger.warning("  ✗ GPU: Not available (training will be slow)")
        # Don't fail on GPU check, just warn
    
    # Network (optional)
    if not skip_network:
        logger.info("Checking network connectivity...")
        if check_network():
            logger.info("  ✓ Network: Connected to GitHub")
        else:
            logger.warning("  ⚠ Network: Cannot reach GitHub (push will fail)")
    
    # Dataset
    logger.info("Checking dataset...")
    try:
        from runpod_utils import get_datasets_path
        datasets_path = get_datasets_path()
    except ImportError:
        datasets_path = project_path / 'datasets'
    
    is_valid, dataset_file = validate_dataset(datasets_path, dataset_name)
    if is_valid:
        logger.info(f"  ✓ Dataset: {dataset_file.name} found")
    else:
        logger.error(f"  ✗ Dataset: {dataset_name}.csv not found")
        all_checks_passed = False
    
    # Dependencies
    logger.info("Checking dependencies...")
    required_packages = ['tensorflow', 'torch', 'stable_baselines3', 'pandas', 'numpy']
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            logger.debug(f"  ✓ {pkg}")
        except ImportError:
            missing.append(pkg)
            logger.warning(f"  ✗ {pkg}: not installed")
    
    if missing:
        logger.warning(f"  Missing packages: {', '.join(missing)}")
        logger.warning("  Run: pip install -r requirements.txt")
        all_checks_passed = False
    
    # Permissions
    logger.info("Checking write permissions...")
    test_dirs = [
        project_path / 'models',
        project_path / 'PPO approach' / 'models',
        project_path / 'PPO approach' / 'checkpoints',
    ]
    for test_dir in test_dirs:
        test_dir.mkdir(parents=True, exist_ok=True)
        if os.access(test_dir, os.W_OK):
            logger.debug(f"  ✓ {test_dir.name}/: writable")
        else:
            logger.error(f"  ✗ {test_dir.name}/: NOT WRITABLE")
            all_checks_passed = False
    
    logger.info("=" * 60)
    if all_checks_passed:
        logger.info("✓ All pre-flight checks passed")
    else:
        logger.warning("⚠ Some pre-flight checks failed")
    
    return all_checks_passed


def cleanup_old_checkpoints(
    checkpoints_path: Path,
    keep_last: int = 3,
    logger: Optional[logging.Logger] = None
) -> int:
    """Clean up old checkpoints, keeping only the N most recent."""
    if logger is None:
        logger = logging.getLogger('runpod_main')
    
    if not checkpoints_path.exists():
        logger.info(f"Checkpoints directory not found: {checkpoints_path}")
        return 0
    
    # Find all checkpoint directories
    checkpoint_dirs = [d for d in checkpoints_path.iterdir() if d.is_dir()]
    
    if len(checkpoint_dirs) <= keep_last:
        logger.info(f"Only {len(checkpoint_dirs)} checkpoint directory(ies), nothing to clean")
        return 0
    
    # Sort by modification time
    checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Remove old ones
    to_remove = checkpoint_dirs[keep_last:]
    removed_count = 0
    
    for old_dir in to_remove:
        try:
            shutil.rmtree(old_dir)
            removed_count += 1
            logger.info(f"Removed old checkpoint: {old_dir.name}")
        except Exception as e:
            logger.warning(f"Failed to remove {old_dir.name}: {e}")
    
    logger.info(f"Cleaned up {removed_count} old checkpoint directory(ies)")
    return removed_count


def cleanup_old_logs(
    logs_path: Path,
    keep_days: int = 7,
    logger: Optional[logging.Logger] = None
) -> int:
    """Clean up old log files."""
    if logger is None:
        logger = logging.getLogger('runpod_main')
    
    if not logs_path.exists():
        return 0
    
    cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)
    removed_count = 0
    
    for log_file in logs_path.glob("*.log"):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                removed_count += 1
                logger.debug(f"Removed old log: {log_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {log_file.name}: {e}")
    
    if removed_count > 0:
        logger.info(f"Cleaned up {removed_count} old log file(s)")
    
    return removed_count


def main():
    parser = argparse.ArgumentParser(
        description='RunPod Training Pipeline - Complete training with validation and monitoring'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., ADA-EUR_1H_20240101-20251231)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run validation only, do not train'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging (DEBUG level)'
    )
    parser.add_argument(
        '--no-push',
        action='store_true',
        help='Skip pushing models to GitHub'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up old checkpoints and logs'
    )
    parser.add_argument(
        '--keep-last',
        type=int,
        default=3,
        help='Number of recent checkpoints to keep (default: 3)'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip pre-flight validation'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    try:
        # Add PPO approach to path
        current_dir = Path(__file__).parent
        ppo_path = current_dir / 'PPO approach'
        if ppo_path.exists():
            sys.path.insert(0, str(ppo_path))
        else:
            if current_dir.name == 'PPO approach':
                sys.path.insert(0, str(current_dir))
        
        from runpod_utils import get_project_path, get_logs_path, get_checkpoints_path
        project_path = get_project_path()
        logs_path = get_logs_path()
        checkpoints_path = get_checkpoints_path()
    except ImportError:
        # Fallback
        project_path = Path.cwd()
        logs_path = project_path / 'logs'
        checkpoints_path = project_path / 'PPO approach' / 'checkpoints'
        print("⚠ runpod_utils not found, using fallback paths")
    
    # Setup logging
    logger = setup_logging(logs_path, verbose=args.verbose)
    logger.info("=" * 60)
    logger.info("RunPod Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Project path: {project_path}")
    logger.info(f"Dataset: {args.dataset}")
    
    # Cleanup if requested
    if args.cleanup:
        logger.info("Running cleanup...")
        cleanup_old_checkpoints(checkpoints_path, keep_last=args.keep_last, logger=logger)
        cleanup_old_logs(logs_path, keep_days=7, logger=logger)
        if args.dry_run:
            logger.info("Cleanup complete (dry-run mode)")
            return 0
    
    # Pre-flight validation
    if not args.skip_validation:
        if not preflight_validation(project_path, args.dataset, logger, skip_network=args.no_push):
            logger.error("Pre-flight validation failed. Use --skip-validation to continue anyway.")
            if not args.dry_run:
                return 1
    else:
        logger.warning("Skipping pre-flight validation (--skip-validation)")
    
    # Dry-run mode
    if args.dry_run:
        logger.info("Dry-run mode: Validation complete, no training performed")
        return 0
    
    # Run training
    logger.info("=" * 60)
    logger.info("Starting training pipeline...")
    logger.info("=" * 60)
    
    # Set environment variable for dataset
    os.environ['DATASET_NAME'] = args.dataset
    if args.resume:
        os.environ['RESUME_TRAINING'] = 'true'
    if not args.no_push:
        os.environ['RUNPOD_PUSH_TO_GITHUB'] = 'true'
    
    # Run training script as subprocess
    try:
        import subprocess
        
        # Change to project directory
        os.chdir(project_path)
        
        # Modify runpod_train.py temporarily to use the specified dataset
        train_script = project_path / 'runpod_train.py'
        if train_script.exists():
            # Execute the script directly (it reads DATASET_NAME from environment)
            logger.info("Executing training script...")
            result = subprocess.run(
                [sys.executable, str(train_script)],
                cwd=project_path,
                check=False
            )
            
            if result.returncode == 0:
                logger.info("=" * 60)
                logger.info("Training pipeline completed successfully!")
                logger.info("=" * 60)
                return 0
            else:
                logger.error(f"Training script exited with code {result.returncode}")
                return result.returncode
        else:
            logger.error(f"Training script not found: {train_script}")
            return 1
            
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

