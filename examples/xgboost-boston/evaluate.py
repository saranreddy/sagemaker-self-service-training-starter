"""Evaluation script for Boston Housing regression."""
import json
import os
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    """Evaluate the trained model."""
    model_dir = Path(os.environ.get('SM_MODEL_DIR', '/opt/ml/processing/model'))
    output_dir = Path(os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/processing/evaluation'))
    test_dir = Path(os.environ.get('SM_CHANNEL_TEST', os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/processing/test')))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    model_path = model_dir / 'xgboost-model'
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    
    print("Loading test data...")
    X_test = np.load(test_dir / 'X.npy')
    y_test = np.load(test_dir / 'y.npy')
    
    print(f"Test data shape: {X_test.shape}")
    
    dtest = xgb.DMatrix(X_test)
    
    print("Making predictions...")
    y_pred = bst.predict(dtest)
    
    print("Calculating metrics...")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
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
