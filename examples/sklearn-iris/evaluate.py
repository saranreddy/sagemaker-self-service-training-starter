"""Evaluation script for Iris classifier."""
import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    """Evaluate the trained model."""
    model_dir = Path(os.environ.get('SM_MODEL_DIR', '/opt/ml/processing/model'))
    output_dir = Path(os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/processing/evaluation'))
    test_dir = Path(os.environ.get('SM_CHANNEL_TEST', os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/processing/test')))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    model_path = model_dir / 'model.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Loading test data...")
    X_test = np.load(test_dir / 'X.npy')
    y_test = np.load(test_dir / 'y.npy')
    
    print(f"Test data shape: {X_test.shape}")
    
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    print("Calculating metrics...")
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted')),
        'recall': float(recall_score(y_test, y_pred, average='weighted')),
        'f1': float(f1_score(y_test, y_pred, average='weighted'))
    }
    
    print("\nEvaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    metrics_file = output_dir / 'metrics.json'
    print(f"\nSaving metrics to {metrics_file}")
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
