#!/usr/bin/env python3
"""
Script to remove old PPO models and checkpoints to start fresh training.

Usage:
    python remove_old_models.py --dataset ADA-EUR_1H_20230101-20251231
    python remove_old_models.py --all  # Remove all models
    python remove_old_models.py --checkpoints-only  # Only remove checkpoints
"""

import argparse
from pathlib import Path
import shutil

try:
    from runpod_utils import get_ppo_models_path, get_checkpoints_path
except ImportError:
    from colab_utils import get_ppo_models_path, get_checkpoints_path


def remove_model_files(dataset_name: str = None, remove_all: bool = False, checkpoints_only: bool = False):
    """
    Remove PPO model files and checkpoints.
    
    Args:
        dataset_name: Specific dataset to remove (e.g., 'ADA-EUR_1H_20230101-20251231')
        remove_all: Remove all models and checkpoints
        checkpoints_only: Only remove checkpoints, keep final models
    """
    models_path = get_ppo_models_path()
    checkpoints_path = get_checkpoints_path()
    
    removed_count = 0
    
    if remove_all:
        print("=" * 60)
        print("REMOVING ALL MODELS AND CHECKPOINTS")
        print("=" * 60)
        
        # Remove all checkpoints
        if checkpoints_path.exists():
            for checkpoint_dir in checkpoints_path.iterdir():
                if checkpoint_dir.is_dir():
                    print(f"  Removing checkpoint directory: {checkpoint_dir}")
                    shutil.rmtree(checkpoint_dir)
                    removed_count += 1
        
        # Remove all final models
        if not checkpoints_only and models_path.exists():
            for model_file in models_path.glob("*.zip"):
                print(f"  Removing model: {model_file.name}")
                model_file.unlink()
                removed_count += 1
        
        print(f"\n✓ Removed {removed_count} items")
        return
    
    # Remove specific dataset
    if dataset_name:
        print("=" * 60)
        print(f"REMOVING MODELS FOR DATASET: {dataset_name}")
        print("=" * 60)
        
        # Remove checkpoints for this dataset
        if checkpoints_path.exists():
            # Checkpoints are in subdirectories like: ensemble_ADA-EUR_1H_20230101-20251231
            for checkpoint_dir in checkpoints_path.iterdir():
                if checkpoint_dir.is_dir() and dataset_name in checkpoint_dir.name:
                    print(f"  Removing checkpoint directory: {checkpoint_dir}")
                    shutil.rmtree(checkpoint_dir)
                    removed_count += 1
        
        # Remove final models for this dataset
        if not checkpoints_only and models_path.exists():
            for model_file in models_path.glob(f"*{dataset_name}*.zip"):
                print(f"  Removing model: {model_file.name}")
                model_file.unlink()
                removed_count += 1
        
        if removed_count == 0:
            print(f"  ⚠ No models found for dataset: {dataset_name}")
        else:
            print(f"\n✓ Removed {removed_count} items for dataset: {dataset_name}")
    else:
        print("Please specify --dataset or use --all to remove everything")


def main():
    parser = argparse.ArgumentParser(
        description='Remove old PPO models and checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove models for specific dataset
  python remove_old_models.py --dataset ADA-EUR_1H_20230101-20251231
  
  # Remove all models and checkpoints
  python remove_old_models.py --all
  
  # Only remove checkpoints (keep final models)
  python remove_old_models.py --checkpoints-only --dataset ADA-EUR_1H_20230101-20251231
        """
    )
    
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name to remove (e.g., ADA-EUR_1H_20230101-20251231)')
    
    parser.add_argument('--all', action='store_true',
                       help='Remove all models and checkpoints')
    
    parser.add_argument('--checkpoints-only', action='store_true',
                       help='Only remove checkpoints, keep final models')
    
    args = parser.parse_args()
    
    if not args.dataset and not args.all:
        parser.print_help()
        print("\n❌ Error: Please specify --dataset or use --all")
        return
    
    remove_model_files(
        dataset_name=args.dataset,
        remove_all=args.all,
        checkpoints_only=args.checkpoints_only
    )


if __name__ == '__main__':
    main()

