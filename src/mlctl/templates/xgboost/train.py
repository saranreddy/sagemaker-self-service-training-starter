"""XGBoost training script for Boston Housing regression (SageMaker script mode)."""

import argparse
import json  # noqa: F401
import os
from pathlib import Path

import numpy as np
import xgboost as xgb


def parse_args():
    """Parse SageMaker-style arguments."""
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--objective", type=str, default="reg:squarederror")
    parser.add_argument("--num_round", type=int, default=100)

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

    return X, y


def main():
    """Main training function."""
    args = parse_args()

    print("Loading training data...")
    X_train, y_train = load_data(args.train)
    print(f"Training data shape: {X_train.shape}")

    dtrain = xgb.DMatrix(X_train, label=y_train)

    params = {"max_depth": args.max_depth, "eta": args.eta, "objective": args.objective}

    evals = [(dtrain, "train")]

    if args.validation and Path(args.validation).exists():
        X_val, y_val = load_data(args.validation)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, "validation"))

    print("\nTraining XGBoost model...")
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=evals,
        verbose_eval=10,
    )

    print(f"\nSaving model to {args.model_dir}")
    model_path = Path(args.model_dir) / "xgboost-model"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    bst.save_model(str(model_path))

    print("Training complete!")


if __name__ == "__main__":
    main()
