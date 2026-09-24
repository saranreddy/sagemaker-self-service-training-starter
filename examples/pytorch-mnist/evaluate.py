"""Evaluation script for MNIST classifier."""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MNISTNet(nn.Module):
    """Simple CNN for MNIST."""
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def main():
    """Evaluate the trained model."""
    model_dir = Path(os.environ.get('SM_MODEL_DIR', '/opt/ml/processing/model'))
    output_dir = Path(os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/processing/evaluation'))
    test_dir = Path(os.environ.get('SM_CHANNEL_TEST', os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/processing/test')))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cpu')
    
    print("Loading model...")
    model = MNISTNet().to(device)
    model_path = model_dir / 'model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("Loading test data...")
    X_test = np.load(test_dir / 'X.npy')
    y_test = np.load(test_dir / 'y.npy')
    
    X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    X_test = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test).long()
    
    print(f"Test data shape: {X_test.shape}")
    
    test_dataset = TensorDataset(X_test, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    print("Making predictions...")
    all_preds = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy().flatten())
    
    y_pred = np.array(all_preds)
    
    print("Calculating metrics...")
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
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
