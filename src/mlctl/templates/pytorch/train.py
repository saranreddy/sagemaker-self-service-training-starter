"""PyTorch training script for MNIST classification (SageMaker script mode)."""

import argparse
import json  # noqa: F401
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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


def parse_args():
    """Parse SageMaker-style arguments."""
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)

    # SageMaker environment variables (also as args for local compatibility)
    parser.add_argument(
        "--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )
    parser.add_argument(
        "--validation",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"
        ),
    )

    return parser.parse_args()


def load_data(data_dir):
    """Load training data from directory."""
    data_path = Path(data_dir)

    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "y.npy")

    X = X.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

    return torch.from_numpy(X), torch.from_numpy(y).long()


def train_epoch(model, device, train_loader, optimizer, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}"  # noqa: E501
            )

    return total_loss / len(train_loader)


def validate(model, device, val_loader):
    """Validate the model."""
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print(
        f"Validation: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({100. * accuracy:.2f}%)"  # noqa: E501
    )

    return val_loss, accuracy


def main():
    """Main training function."""
    args = parse_args()

    device = torch.device("cpu")

    print("Loading training data...")
    X_train, y_train = load_data(args.train)
    print(f"Training data shape: {X_train.shape}")

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if args.validation and Path(args.validation).exists():
        X_val, y_val = load_data(args.validation)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print("\nInitializing model...")
    model = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, device, train_loader, optimizer, epoch)

        if val_loader:
            val_loss, val_accuracy = validate(model, device, val_loader)

    print(f"\nSaving model to {args.model_dir}")
    model_path = Path(args.model_dir) / "model.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)

    print("Training complete!")


if __name__ == "__main__":
    main()
