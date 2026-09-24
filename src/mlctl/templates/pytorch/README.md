# PyTorch MNIST Classifier Example

This example trains a CNN on the MNIST dataset using PyTorch (CPU).

## Project Structure

- `ml.yaml` - Project configuration
- `train.py` - Training script (SageMaker script mode compatible)
- `evaluate.py` - Evaluation script that writes metrics.json
- `requirements.txt` - Python dependencies
- `data/` - Local data directory (created by generate_data.py)

## Quick Start

1. Generate synthetic data:
   ```bash
   python generate_data.py
   ```

2. Validate the project:
   ```bash
   mlctl validate
   ```

3. Run locally:
   ```bash
   mlctl run --local
   ```

4. Submit to SageMaker:
   ```bash
   mlctl submit
   ```

## Quality Gate

The model must achieve at least 95% accuracy on the test set to pass the quality gate and be registered in the model registry.

## Note

This example uses CPU-only PyTorch. For GPU training, update the instance type in ml.yaml to a GPU instance (if allowed by your team's instance allowlist).
