#!/usr/bin/env python3
"""
Comprehensive Analysis of Trained Models and PPO Agent
=======================================================
Analyzes prediction models and PPO agent performance.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'PPO approach'))

def analyze_prediction_model(history_path: Path) -> Dict:
    """Analyze a single prediction model's training history."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    model_name = history_path.stem.replace('history_', '').replace('_classification', '').split('_')[0]
    
    # Extract final metrics
    final_train_acc = history['accuracy'][-1] if history['accuracy'] else 0
    final_val_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 0
    final_train_loss = history['loss'][-1] if history['loss'] else float('inf')
    final_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
    
    # Calculate best metrics
    best_train_acc = max(history['accuracy']) if history['accuracy'] else 0
    best_val_acc = max(history['val_accuracy']) if history['val_accuracy'] else 0
    best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
    
    # Training stability (lower std = more stable)
    train_acc_std = np.std(history['accuracy'][-10:]) if len(history['accuracy']) >= 10 else 0
    val_acc_std = np.std(history['val_accuracy'][-10:]) if len(history['val_accuracy']) >= 10 else 0
    
    # Overfitting indicator (train_acc - val_acc)
    overfitting = final_train_acc - final_val_acc
    
    return {
        'model_name': model_name.upper(),
        'epochs': len(history['accuracy']),
        'final_train_accuracy': final_train_acc,
        'final_val_accuracy': final_val_acc,
        'best_train_accuracy': best_train_acc,
        'best_val_accuracy': best_val_acc,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'train_acc_std': train_acc_std,
        'val_acc_std': val_acc_std,
        'overfitting': overfitting,
        'convergence': 'Good' if abs(final_train_acc - best_train_acc) < 2 else 'Needs more training'
    }

def analyze_ppo_agent(checkpoint_dir: Path) -> Dict:
    """Analyze PPO agent from checkpoint directory."""
    info = {
        'checkpoint_dir': str(checkpoint_dir),
        'has_checkpoint': (checkpoint_dir / 'latest.zip').exists(),
        'has_best_model': (checkpoint_dir / 'best' / 'best_model.zip').exists(),
    }
    
    # Try to get model file size
    if info['has_checkpoint']:
        checkpoint_size = (checkpoint_dir / 'latest.zip').stat().st_size / (1024 * 1024)  # MB
        info['checkpoint_size_mb'] = checkpoint_size
    
    if info['has_best_model']:
        best_size = (checkpoint_dir / 'best' / 'best_model.zip').stat().st_size / (1024 * 1024)
        info['best_model_size_mb'] = best_size
    
    return info

def main():
    """Main analysis function."""
    project_path = Path(__file__).parent
    results_path = project_path / 'results'
    checkpoints_path = project_path / 'PPO approach' / 'checkpoints'
    models_path = project_path / 'models'
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    # Analyze Prediction Models
    print("=" * 80)
    print("PART 1: PREDICTION MODELS ANALYSIS")
    print("=" * 80)
    print()
    
    history_files = list(results_path.glob('history_*_classification.json'))
    
    if not history_files:
        print("WARNING: No training history files found!")
        return
    
    model_results = []
    for history_file in history_files:
        try:
            result = analyze_prediction_model(history_file)
            model_results.append(result)
        except Exception as e:
            print(f"WARNING: Error analyzing {history_file.name}: {e}")
    
    # Sort by validation accuracy
    model_results.sort(key=lambda x: x['best_val_accuracy'], reverse=True)
    
    # Print detailed results
    print(f"{'Model':<10} {'Epochs':<8} {'Train Acc':<12} {'Val Acc':<12} {'Best Val':<12} {'Overfit':<10} {'Status':<15}")
    print("-" * 80)
    
    for result in model_results:
        print(f"{result['model_name']:<10} "
              f"{result['epochs']:<8} "
              f"{result['final_train_accuracy']:>6.2f}%     "
              f"{result['final_val_accuracy']:>6.2f}%     "
              f"{result['best_val_accuracy']:>6.2f}%     "
              f"{result['overfitting']:>6.2f}%     "
              f"{result['convergence']:<15}")
    
    print()
    print("=" * 80)
    print("DETAILED METRICS")
    print("=" * 80)
    print()
    
    for result in model_results:
        print(f"\n{result['model_name']} Model:")
        print(f"  Epochs trained: {result['epochs']}")
        print(f"  Final Training Accuracy: {result['final_train_accuracy']:.2f}%")
        print(f"  Final Validation Accuracy: {result['final_val_accuracy']:.2f}%")
        print(f"  Best Training Accuracy: {result['best_train_accuracy']:.2f}%")
        print(f"  Best Validation Accuracy: {result['best_val_accuracy']:.2f}%")
        print(f"  Final Training Loss: {result['final_train_loss']:.4f}")
        print(f"  Final Validation Loss: {result['final_val_loss']:.4f}")
        print(f"  Best Validation Loss: {result['best_val_loss']:.4f}")
        print(f"  Overfitting (train - val): {result['overfitting']:.2f}%")
        print(f"  Training Stability (std): {result['train_acc_std']:.2f}%")
        print(f"  Validation Stability (std): {result['val_acc_std']:.2f}%")
        print(f"  Convergence Status: {result['convergence']}")
    
    # Model ranking
    print()
    print("=" * 80)
    print("MODEL RANKING (by Best Validation Accuracy)")
    print("=" * 80)
    for i, result in enumerate(model_results, 1):
        print(f"{i}. {result['model_name']}: {result['best_val_accuracy']:.2f}% validation accuracy")
    
    # Ensemble recommendation
    print()
    print("=" * 80)
    print("ENSEMBLE RECOMMENDATION")
    print("=" * 80)
    top_2 = model_results[:2]
    print(f"Current ensemble uses: {', '.join([m['model_name'] for m in top_2])}")
    print(f"Combined expected accuracy: ~{np.mean([m['best_val_accuracy'] for m in top_2]):.2f}%")
    print(f"  (Ensemble typically performs better than average)")
    
    # Analyze PPO Agent
    print()
    print("=" * 80)
    print("PART 2: PPO AGENT ANALYSIS")
    print("=" * 80)
    print()
    
    # Find ensemble checkpoint
    ensemble_checkpoint = checkpoints_path / 'ensemble_ADA-EUR_1H_20240101-20251231'
    
    if ensemble_checkpoint.exists():
        ppo_info = analyze_ppo_agent(ensemble_checkpoint)
        print(f"Checkpoint Directory: {ppo_info['checkpoint_dir']}")
        print(f"Latest Checkpoint: {'Found' if ppo_info['has_checkpoint'] else 'Missing'}")
        print(f"Best Model: {'Found' if ppo_info['has_best_model'] else 'Missing'}")
        
        if 'checkpoint_size_mb' in ppo_info:
            print(f"Checkpoint Size: {ppo_info['checkpoint_size_mb']:.2f} MB")
        if 'best_model_size_mb' in ppo_info:
            print(f"Best Model Size: {ppo_info['best_model_size_mb']:.2f} MB")
    else:
        print("WARNING: Ensemble checkpoint directory not found!")
    
    # Check final PPO model
    ppo_models_path = project_path / 'PPO approach' / 'models'
    ppo_model = ppo_models_path / 'ppo_ensemble_ADA-EUR_1H_20240101-20251231.zip'
    
    if ppo_model.exists():
        model_size = ppo_model.stat().st_size / (1024 * 1024)
        print(f"\nFinal PPO Model: Found")
        print(f"  Location: {ppo_model}")
        print(f"  Size: {model_size:.2f} MB")
    else:
        print(f"\nFinal PPO Model: Not found at {ppo_model}")
    
    # Summary and Recommendations
    print()
    print("=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    best_model = model_results[0]
    print(f"Best Prediction Model: {best_model['model_name']} ({best_model['best_val_accuracy']:.2f}% validation accuracy)")
    
    # Find actual ensemble models (DLSTM and BiLSTM based on config)
    ensemble_models = [m for m in model_results if m['model_name'] in ['DLSTM', 'BILSTM']]
    if ensemble_models:
        print(f"Ensemble Configuration: {', '.join([m['model_name'] for m in ensemble_models])}")
        print(f"  Expected ensemble accuracy: ~{np.mean([m['best_val_accuracy'] for m in ensemble_models]):.2f}%")
    else:
        print(f"Ensemble Configuration: {', '.join([m['model_name'] for m in top_2])}")
    print(f"PPO Agent: Trained with ensemble predictions")
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 80)
    
    # Check for overfitting
    for result in model_results:
        if result['overfitting'] > 5:
            print(f"WARNING: {result['model_name']} shows overfitting ({result['overfitting']:.2f}% gap)")
            print(f"  - Consider adding dropout or reducing model complexity")
    
    # Check convergence
    for result in model_results:
        if result['convergence'] == 'Needs more training':
            print(f"WARNING: {result['model_name']} may need more training epochs")
            print(f"  - Current: {result['epochs']} epochs, Best: {result['best_val_accuracy']:.2f}%")
    
    # PPO recommendations
    print()
    print("PPO Agent Recommendations:")
    print("  - Evaluate the agent on test data to measure trading performance")
    print("  - Run backtesting to see actual trading results")
    print("  - Consider tuning reward function if mean reward was 0.00")
    print("  - Monitor Sharpe ratio and drawdown in evaluation")
    
    print()
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()

