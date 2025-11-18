#!/usr/bin/env python3
"""
Evaluate all dev set predictions to find the best model
"""

import os
import glob
from collections import defaultdict

def evaluate_predictions(pred_file, gold_file):
    """Calculate exact match accuracy"""
    with open(pred_file, 'r') as f:
        predictions = [line.strip() for line in f]
    
    with open(gold_file, 'r') as f:
        gold = [line.strip() for line in f]
    
    if len(predictions) != len(gold):
        print(f"Warning: {pred_file} has {len(predictions)} predictions but {len(gold)} gold labels")
        return None
    
    correct = sum(1 for p, g in zip(predictions, gold) if p == g)
    accuracy = correct / len(gold) * 100
    return accuracy

def main():
    # Path to your results directory
    results_dir = "results"
    gold_dev_file = "data/dev.sql"  # Adjust if needed
    
    # Check if gold file exists
    if not os.path.exists(gold_dev_file):
        print(f"Error: Gold dev file not found at {gold_dev_file}")
        print("Looking for data directory...")
        # Try to find it
        possible_paths = ["data/dev.sql", "../data/dev.sql", "./dev.sql"]
        for path in possible_paths:
            if os.path.exists(path):
                gold_dev_file = path
                print(f"Found at: {gold_dev_file}")
                break
    
    # Find all dev prediction files
    dev_files = glob.glob(f"{results_dir}/*_dev_*.sql")
    
    if not dev_files:
        print(f"No dev prediction files found in {results_dir}")
        return
    
    print(f"Found {len(dev_files)} dev prediction files\n")
    print("="*80)
    
    # Group by model configuration
    results = defaultdict(list)
    
    for pred_file in sorted(dev_files):
        filename = os.path.basename(pred_file)
        
        # Extract model name and epoch
        parts = filename.replace('.sql', '').split('_')
        
        # Try to extract epoch number
        epoch = None
        for i, part in enumerate(parts):
            if part == 'epoch' and i+1 < len(parts):
                try:
                    epoch = int(parts[i+1])
                except ValueError:
                    epoch = parts[i+1]
                break
        
        # Create model identifier (everything before epoch)
        model_name = filename.split('_epoch')[0] if '_epoch' in filename else filename.replace('.sql', '')
        
        accuracy = evaluate_predictions(pred_file, gold_dev_file)
        
        if accuracy is not None:
            results[model_name].append((epoch, accuracy, filename))
            print(f"{filename:70s} | Accuracy: {accuracy:.2f}%")
    
    # Print summary by model
    print("\n" + "="*80)
    print("SUMMARY BY MODEL CONFIGURATION:")
    print("="*80 + "\n")
    
    best_overall = None
    best_overall_acc = 0
    
    for model_name, epochs in sorted(results.items()):
        print(f"\n{model_name}:")
        print("-" * 80)
        
        epochs_sorted = sorted(epochs, key=lambda x: (x[0] if isinstance(x[0], int) else 999, x[1]), reverse=False)
        
        for epoch, acc, filename in epochs_sorted:
            epoch_str = f"epoch {epoch}" if epoch is not None else "unknown epoch"
            print(f"  {epoch_str:15s} | {acc:6.2f}% | {filename}")
            
            if acc > best_overall_acc:
                best_overall_acc = acc
                best_overall = filename
        
        # Show best for this model
        best_for_model = max(epochs_sorted, key=lambda x: x[1])
        print(f"  → Best: epoch {best_for_model[0]} with {best_for_model[1]:.2f}%")
    
    print("\n" + "="*80)
    print(f"BEST OVERALL MODEL: {best_overall}")
    print(f"BEST DEV ACCURACY: {best_overall_acc:.2f}%")
    print("="*80)
    
    print("\n\n📋 NEXT STEPS:")
    print("1. Use the best model checkpoint to generate test predictions")
    print("2. Submit the test predictions for evaluation")
    print(f"\nRecommended checkpoint: {best_overall.replace('_dev_', '_checkpoint_').replace('.sql', '')}")

if __name__ == "__main__":
    main()