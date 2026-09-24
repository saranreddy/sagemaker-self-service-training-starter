"""Scikit-learn training script for Iris classification (SageMaker script mode)."""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def parse_args():
    """Parse SageMaker-style arguments."""
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)

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
        default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"),
    )

    return parser.parse_args()


def load_data(data_dir):
    """Load training data from directory."""
    data_path = Path(data_dir)

    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "y.npy")

    return X, y


def main():
    """Main training function."""
    args = parse_args()

    print("Loading training data...")
    X_train, y_train = load_data(args.train)
    print(f"Training data shape: {X_train.shape}")

    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        max_depth=args.max_depth, n_estimators=args.n_estimators, random_state=args.random_state
    )

    clf.fit(X_train, y_train)

    train_accuracy = clf.score(X_train, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")

    if args.validation and Path(args.validation).exists():
        X_val, y_val = load_data(args.validation)
        val_accuracy = clf.score(X_val, y_val)
        print(f"Validation accuracy: {val_accuracy:.4f}")

    print(f"\nSaving model to {args.model_dir}")
    model_path = Path(args.model_dir) / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    print("Training complete!")


if __name__ == "__main__":
    main()
