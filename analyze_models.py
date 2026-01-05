#!/usr/bin/env python3
"""
Model Analysis Script
=====================
Analyze trained prediction models and PPO agent performance.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import pandas as pd

def analyze_training_history(history_path: Path) -> Dict:
    """Analyze training history and extract key metrics."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Calculate final metrics
    final_train_acc = history['accuracy'][-1] if history['accuracy'] else 0
    final_val_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 0
    final_train_loss = history['loss'][-1] if history['loss'] else float('inf')
    final_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
    
    # Find best validation accuracy
    best_val_acc = max(history['val_accuracy']) if history['val_accuracy'] else 0
    best_val_acc_epoch = history['val_accuracy'].index(best_val_acc) if best_val_acc > 0 else 0
    
    # Calculate training stability (std of last 10 epochs)
    if len(history['val_accuracy']) >= 10:
        stability = np.std(history['val_accuracy'][-10:])
    else:
        stability = np.std(history['val_accuracy']) if history['val_accuracy'] else 0
    
    # Check for overfitting (val_loss > train_loss at end)
    overfitting = final_val_loss > final_train_loss * 1.1
    
    return {
        'final_train_accuracy': final_train_acc,
        'final_val_accuracy': final_val_acc,
        'best_val_accuracy': best_val_acc,
        'best_val_epoch': best_val_acc_epoch,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'stability': stability,
        'overfitting': overfitting,
        'total_epochs': len(history['accuracy']),
        'improvement': final_val_acc - history['val_accuracy'][0] if history['val_accuracy'] else 0
    }

def analyze_all_models(results_dir: Path) -> pd.DataFrame:
    """Analyze all model training histories."""
    models = ['lstm', 'gru', 'bilstm', 'dlstm']
    results = []
    
    for model in models:
        history_file = results_dir / f"history_{model}_ADA-EUR_1H_20240101-20251231_classification.json"
        if history_file.exists():
            metrics = analyze_training_history(history_file)
            metrics['model'] = model.upper()
            results.append(metrics)
        else:
            print(f"⚠ Warning: {history_file} not found")
    
    return pd.DataFrame(results)

def print_analysis_report(df: pd.DataFrame):
    """Print a comprehensive analysis report."""
    print("=" * 80)
    print("MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"\nDataset: ADA-EUR_1H_20240101-20251231")
    print(f"Task: Classification (Fall, Stationary, Rise)")
    print(f"Total Models Analyzed: {len(df)}")
    
    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE METRICS")
    print("=" * 80)
    
    # Sort by validation accuracy
    df_sorted = df.sort_values('final_val_accuracy', ascending=False)
    
    print(f"\n{'Model':<10} {'Val Acc':<12} {'Train Acc':<12} {'Best Val':<12} {'Epochs':<8} {'Stability':<12}")
    print("-" * 80)
    for _, row in df_sorted.iterrows():
        print(f"{row['model']:<10} {row['final_val_accuracy']:>6.2f}%    {row['final_train_accuracy']:>6.2f}%    "
              f"{row['best_val_accuracy']:>6.2f}%    {int(row['total_epochs']):<8} {row['stability']:>6.4f}")
    
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    for _, row in df_sorted.iterrows():
        print(f"\n{row['model']} Model:")
        print(f"  Final Validation Accuracy: {row['final_val_accuracy']:.2f}%")
        print(f"  Final Training Accuracy: {row['final_train_accuracy']:.2f}%")
        print(f"  Best Validation Accuracy: {row['best_val_accuracy']:.2f}% (Epoch {int(row['best_val_epoch']) + 1})")
        print(f"  Total Epochs Trained: {int(row['total_epochs'])}")
        print(f"  Accuracy Improvement: {row['improvement']:.2f}%")
        print(f"  Training Stability (std): {row['stability']:.4f}")
        print(f"  Final Train Loss: {row['final_train_loss']:.4f}")
        print(f"  Final Val Loss: {row['final_val_loss']:.4f}")
        if row['overfitting']:
            print(f"  ⚠ Warning: Potential overfitting detected (val_loss > train_loss)")
        else:
            print(f"  ✓ No significant overfitting")
    
    print("\n" + "=" * 80)
    print("MODEL RANKING")
    print("=" * 80)
    
    # Rank by different metrics
    print("\n1. By Final Validation Accuracy:")
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"   {i}. {row['model']}: {row['final_val_accuracy']:.2f}%")
    
    print("\n2. By Best Validation Accuracy:")
    df_best = df.sort_values('best_val_accuracy', ascending=False)
    for i, (_, row) in enumerate(df_best.iterrows(), 1):
        print(f"   {i}. {row['model']}: {row['best_val_accuracy']:.2f}%")
    
    print("\n3. By Training Stability (lower is better):")
    df_stable = df.sort_values('stability', ascending=True)
    for i, (_, row) in enumerate(df_stable.iterrows(), 1):
        print(f"   {i}. {row['model']}: {row['stability']:.4f} (std)")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    best_model = df_sorted.iloc[0]
    print(f"\n✓ Best Performing Model: {best_model['model']}")
    print(f"  - Validation Accuracy: {best_model['final_val_accuracy']:.2f}%")
    print(f"  - Best Accuracy Achieved: {best_model['best_val_accuracy']:.2f}%")
    
    # Check for overfitting
    overfitting_models = df[df['overfitting']]
    if len(overfitting_models) > 0:
        print(f"\n⚠ Models with potential overfitting:")
        for _, row in overfitting_models.iterrows():
            print(f"  - {row['model']}: Consider early stopping or regularization")
    
    # Check stability
    unstable_models = df[df['stability'] > 2.0]
    if len(unstable_models) > 0:
        print(f"\n⚠ Models with high variance (unstable training):")
        for _, row in unstable_models.iterrows():
            print(f"  - {row['model']}: std={row['stability']:.4f}")
    
    print("\n" + "=" * 80)
    print("PPO TRAINING STATUS")
    print("=" * 80)
    
    ppo_model = Path("PPO approach/models/ppo_dlstm_ADA-EUR_1H_20240101-20251231.zip")
    if ppo_model.exists():
        size_mb = ppo_model.stat().st_size / (1024 * 1024)
        print(f"\n✓ PPO Model Found: {ppo_model.name}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Status: Ready for trading")
    else:
        print("\n⚠ PPO model not found")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    results_dir = Path("results")
    if not results_dir.exists():
        print("Error: results directory not found")
        exit(1)
    
    df = analyze_all_models(results_dir)
    if len(df) == 0:
        print("No model histories found")
        exit(1)
    
    print_analysis_report(df)
    
    # Save summary to CSV
    summary_file = results_dir / "model_analysis_summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved to: {summary_file}")




