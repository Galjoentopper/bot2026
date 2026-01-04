#!/usr/bin/env python3
"""
Hyperparameter Tuning Script
============================
Automated hyperparameter tuning using grid search or random search.
Focuses on optimizing DLSTM model (best performer from analysis).
"""

import json
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import product
import random
from datetime import datetime
from configparser import ConfigParser

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train_models import train_single_model, get_datasets_path
from model_versioning import get_version_manager


class HyperparameterTuner:
    """Hyperparameter tuning using grid search or random search."""
    
    def __init__(
        self,
        model_name: str = 'dlstm',
        dataset_name: str = None,
        search_type: str = 'grid',
        n_random_trials: int = 20,
        output_dir: str = 'tuning_results'
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            model_name: Model to tune (default: 'dlstm')
            dataset_name: Dataset to use (default: from config)
            search_type: 'grid' or 'random'
            n_random_trials: Number of random trials (for random search)
            output_dir: Directory to save results
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.search_type = search_type
        self.n_random_trials = n_random_trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define search space
        self.search_space = {
            'learning_rate': [0.0001, 0.0002, 0.0003, 0.0005],
            'units': [256, 512, 768],
            'layers': [3, 4, 5],
            'dropout': [0.1, 0.15, 0.2],
            'batch_size': [512, 1024, 1536],  # Auto-adjusted for GPU
        }
        
        # Results storage
        self.results = []
        self.best_config = None
        self.best_accuracy = 0.0
    
    def generate_configs(self) -> List[Dict]:
        """
        Generate hyperparameter configurations to test.
        
        Returns:
            List of config dictionaries
        """
        if self.search_type == 'grid':
            # Generate all combinations
            keys = list(self.search_space.keys())
            values = list(self.search_space.values())
            configs = []
            
            for combo in product(*values):
                config = dict(zip(keys, combo))
                configs.append(config)
            
            print(f"Grid search: {len(configs)} configurations to test")
            return configs
        
        elif self.search_type == 'random':
            # Randomly sample from search space
            configs = []
            for _ in range(self.n_random_trials):
                config = {}
                for key, values in self.search_space.items():
                    config[key] = random.choice(values)
                configs.append(config)
            
            print(f"Random search: {self.n_random_trials} configurations to test")
            return configs
        
        else:
            raise ValueError(f"Unknown search_type: {self.search_type}")
    
    def create_config_from_dict(self, base_config_path: Path, hyperparams: Dict) -> ConfigParser:
        """
        Create a ConfigParser from base config with hyperparameters replaced.
        
        Args:
            base_config_path: Path to base training_config.txt
            hyperparams: Dictionary of hyperparameters to set
            
        Returns:
            ConfigParser with updated values
        """
        config = ConfigParser()
        config.read(base_config_path)
        
        # Update hyperparameters
        if 'learning_rate' in hyperparams:
            config.set('MODEL', 'learning_rate', str(hyperparams['learning_rate']))
        if 'units' in hyperparams:
            config.set('MODEL', 'units', str(hyperparams['units']))
        if 'layers' in hyperparams:
            config.set('MODEL', 'layers', str(hyperparams['layers']))
        if 'dropout' in hyperparams:
            config.set('MODEL', 'dropout', str(hyperparams['dropout']))
        if 'batch_size' in hyperparams:
            config.set('TRAINING', 'batch_size', str(hyperparams['batch_size']))
        
        return config
    
    def tune(
        self,
        base_config_path: Path,
        datasets_dir: Path
    ) -> Tuple[Dict, float]:
        """
        Run hyperparameter tuning.
        
        Args:
            base_config_path: Path to base training_config.txt
            datasets_dir: Directory containing datasets
            
        Returns:
            Tuple of (best_config_dict, best_accuracy)
        """
        print("=" * 60)
        print(f"Hyperparameter Tuning: {self.model_name.upper()}")
        print("=" * 60)
        print(f"Search type: {self.search_type}")
        print(f"Dataset: {self.dataset_name}")
        print()
        
        # Generate configurations
        configs = self.generate_configs()
        
        # Load dataset
        if self.dataset_name is None:
            # Find first available dataset
            dataset_files = list(datasets_dir.glob('*.csv'))
            if not dataset_files:
                raise ValueError("No datasets found in datasets directory")
            self.dataset_name = dataset_files[0].stem
            print(f"Using dataset: {self.dataset_name}")
        
        dataset_path = datasets_dir / f"{self.dataset_name}.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded: {len(df)} rows")
        print()
        
        # Test each configuration
        for i, hyperparams in enumerate(configs, 1):
            print(f"\n{'='*60}")
            print(f"Trial {i}/{len(configs)}")
            print(f"{'='*60}")
            print("Hyperparameters:")
            for key, value in hyperparams.items():
                print(f"  {key}: {value}")
            print()
            
            try:
                # Create config for this trial
                config = self.create_config_from_dict(base_config_path, hyperparams)
                
                # Train model
                print(f"Training {self.model_name.upper()}...")
                result = train_single_model(
                    model_name=self.model_name,
                    dataset_name=self.dataset_name,
                    df=df,
                    config=config,
                    task='classification',
                    output_dir=str(project_root)
                )
                
                # Extract metrics
                metrics = result.get('metrics', {})
                history = result.get('history', {})
                
                # Get accuracy from metrics
                accuracy = metrics.get('accuracy', 0.0)
                
                # Get validation accuracy from metrics or history
                val_accuracy = metrics.get('val_accuracy', 0.0)
                if val_accuracy == 0.0 and history:
                    # Try to get from history (last epoch)
                    val_acc_history = history.get('val_accuracy', [])
                    if val_acc_history:
                        val_accuracy = val_acc_history[-1] if isinstance(val_acc_history, list) else val_accuracy
                
                # Use validation accuracy as primary metric (fallback to training accuracy)
                primary_metric = val_accuracy if val_accuracy > 0 else accuracy
                
                print(f"\n✓ Trial {i} completed")
                print(f"  Validation Accuracy: {primary_metric:.4f}")
                
                # Store result
                result_entry = {
                    'trial': i,
                    'hyperparameters': hyperparams.copy(),
                    'accuracy': accuracy,
                    'val_accuracy': val_accuracy,
                    'primary_metric': primary_metric,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }
                self.results.append(result_entry)
                
                # Update best
                if primary_metric > self.best_accuracy:
                    self.best_accuracy = primary_metric
                    self.best_config = hyperparams.copy()
                    print(f"  🎯 NEW BEST! Accuracy: {primary_metric:.4f}")
                
            except Exception as e:
                print(f"\n✗ Trial {i} failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Store failed result
                result_entry = {
                    'trial': i,
                    'hyperparameters': hyperparams.copy(),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                self.results.append(result_entry)
                continue
        
        # Save results
        self.save_results()
        
        return self.best_config, self.best_accuracy
    
    def save_results(self):
        """Save tuning results to CSV and JSON."""
        # Save CSV
        csv_path = self.output_dir / 'tuning_results.csv'
        with open(csv_path, 'w', newline='') as f:
            if not self.results:
                return
            
            # Get all unique keys
            fieldnames = ['trial', 'primary_metric', 'accuracy', 'val_accuracy', 'timestamp']
            for result in self.results:
                if 'hyperparameters' in result:
                    fieldnames.extend(result['hyperparameters'].keys())
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'trial': result.get('trial'),
                    'primary_metric': result.get('primary_metric', 0.0),
                    'accuracy': result.get('accuracy', 0.0),
                    'val_accuracy': result.get('val_accuracy', 0.0),
                    'timestamp': result.get('timestamp', ''),
                }
                
                if 'hyperparameters' in result:
                    row.update(result['hyperparameters'])
                
                writer.writerow(row)
        
        print(f"\n✓ Results saved to: {csv_path}")
        
        # Save JSON with full details
        json_path = self.output_dir / 'tuning_results.json'
        with open(json_path, 'w') as f:
            json.dump({
                'best_config': self.best_config,
                'best_accuracy': self.best_accuracy,
                'search_type': self.search_type,
                'model_name': self.model_name,
                'dataset_name': self.dataset_name,
                'total_trials': len(self.results),
                'results': self.results
            }, f, indent=2)
        
        print(f"✓ Full results saved to: {json_path}")
        
        # Save best hyperparameters
        if self.best_config:
            best_path = self.output_dir / 'best_hyperparameters.json'
            with open(best_path, 'w') as f:
                json.dump({
                    'hyperparameters': self.best_config,
                    'accuracy': self.best_accuracy,
                    'model_name': self.model_name,
                    'dataset_name': self.dataset_name,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"✓ Best hyperparameters saved to: {best_path}")
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate markdown report of tuning results."""
        report_path = self.output_dir / 'tuning_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Hyperparameter Tuning Report\n\n")
            f.write(f"**Model**: {self.model_name.upper()}\n")
            f.write(f"**Dataset**: {self.dataset_name}\n")
            f.write(f"**Search Type**: {self.search_type}\n")
            f.write(f"**Total Trials**: {len(self.results)}\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if self.best_config:
                f.write("## Best Configuration\n\n")
                f.write(f"**Validation Accuracy**: {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)\n\n")
                f.write("**Hyperparameters**:\n\n")
                for key, value in sorted(self.best_config.items()):
                    f.write(f"- `{key}`: {value}\n")
                f.write("\n")
            
            # Top 5 configurations
            successful_results = [r for r in self.results if 'primary_metric' in r]
            if successful_results:
                successful_results.sort(key=lambda x: x.get('primary_metric', 0), reverse=True)
                top_5 = successful_results[:5]
                
                f.write("## Top 5 Configurations\n\n")
                f.write("| Rank | Accuracy | Learning Rate | Units | Layers | Dropout | Batch Size |\n")
                f.write("|------|----------|---------------|-------|--------|---------|------------|\n")
                
                for i, result in enumerate(top_5, 1):
                    hp = result.get('hyperparameters', {})
                    acc = result.get('primary_metric', 0.0)
                    f.write(f"| {i} | {acc:.4f} | {hp.get('learning_rate', 'N/A')} | "
                           f"{hp.get('units', 'N/A')} | {hp.get('layers', 'N/A')} | "
                           f"{hp.get('dropout', 'N/A')} | {hp.get('batch_size', 'N/A')} |\n")
                f.write("\n")
            
            # Failed trials
            failed_results = [r for r in self.results if 'error' in r]
            if failed_results:
                f.write(f"## Failed Trials ({len(failed_results)})\n\n")
                for result in failed_results:
                    f.write(f"- **Trial {result.get('trial')}**: {result.get('error', 'Unknown error')}\n")
                f.write("\n")
        
        print(f"✓ Report generated: {report_path}")


def main():
    """Main entry point for hyperparameter tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for prediction models')
    parser.add_argument('--model', type=str, default='dlstm',
                       choices=['lstm', 'gru', 'bilstm', 'dlstm'],
                       help='Model to tune (default: dlstm)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (default: first available)')
    parser.add_argument('--search', type=str, default='random',
                       choices=['grid', 'random'],
                       help='Search type: grid or random (default: random)')
    parser.add_argument('--trials', type=int, default=20,
                       help='Number of random trials (default: 20)')
    parser.add_argument('--output', type=str, default='tuning_results',
                       help='Output directory (default: tuning_results)')
    
    args = parser.parse_args()
    
    # Get paths
    project_root = Path(__file__).parent
    base_config_path = project_root / 'training_config.txt'
    datasets_dir = get_datasets_path()
    
    if not base_config_path.exists():
        print(f"Error: training_config.txt not found at {base_config_path}")
        return 1
    
    # Create tuner
    tuner = HyperparameterTuner(
        model_name=args.model,
        dataset_name=args.dataset,
        search_type=args.search,
        n_random_trials=args.trials,
        output_dir=args.output
    )
    
    # Run tuning
    try:
        best_config, best_accuracy = tuner.tune(base_config_path, datasets_dir)
        
        print("\n" + "=" * 60)
        print("TUNING COMPLETE")
        print("=" * 60)
        print(f"Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        print("\nBest Hyperparameters:")
        for key, value in sorted(best_config.items()):
            print(f"  {key}: {value}")
        print(f"\nResults saved to: {tuner.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTuning interrupted by user")
        print("Saving partial results...")
        tuner.save_results()
        return 1
    except Exception as e:
        print(f"\nError during tuning: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

