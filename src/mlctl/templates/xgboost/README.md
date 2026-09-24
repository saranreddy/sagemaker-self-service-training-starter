# XGBoost Boston Housing Example

This example trains an XGBoost model for Boston Housing price prediction.

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
   mlctl run
   ```

4. Submit to SageMaker:
   ```bash
   mlctl submit
   ```

## Quality Gate

The model must achieve RMSE ≤ 5.0 on the test set to pass the quality gate and be registered in the model registry.
