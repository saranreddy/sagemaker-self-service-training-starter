"""Code packaging utilities for SageMaker script mode."""

import hashlib
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Any

import boto3


def package_training_code(project_dir: Path, project_name: str) -> Path:
    """Package training code as sourcedir.tar.gz for script mode.

    Includes:
    - train.py
    - requirements.txt (if exists)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "sourcedir.tar.gz"

        with tarfile.open(tar_path, "w:gz") as tar:
            # Add train.py
            train_py = project_dir / "train.py"
            if not train_py.exists():
                raise FileNotFoundError(f"train.py not found in {project_dir}")
            tar.add(train_py, arcname="train.py")

            # Add requirements.txt if it exists
            requirements = project_dir / "requirements.txt"
            if requirements.exists():
                tar.add(requirements, arcname="requirements.txt")

        # Move to persistent location
        output_dir = Path(tempfile.gettempdir()) / "mlctl" / project_name
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "sourcedir.tar.gz"

        import shutil

        shutil.copy(tar_path, final_path)

        return final_path


def package_evaluation_code(project_dir: Path, project_name: str) -> Path:
    """Package evaluation code with wrapper entrypoint.

    Includes:
    - evaluate_entrypoint.sh (wrapper script)
    - evaluate.py
    - requirements.txt (if exists)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "evaluation.tar.gz"

        # Create entrypoint wrapper
        entrypoint_content = """#!/bin/bash
set -e

echo "=== Evaluation Entrypoint ==="
echo "Extracting model..."
cd /opt/ml/processing/model
tar -xzf model.tar.gz

echo "Installing dependencies..."
if [ -f /opt/ml/processing/input/code/requirements.txt ]; then
    pip install -q -r /opt/ml/processing/input/code/requirements.txt
fi

echo "Running evaluation..."
export PYTHONUNBUFFERED=1
python /opt/ml/processing/input/code/evaluate.py

echo "=== Evaluation Complete ==="
"""

        entrypoint_path = Path(tmpdir) / "evaluate_entrypoint.sh"
        entrypoint_path.write_text(entrypoint_content)
        entrypoint_path.chmod(0o755)

        with tarfile.open(tar_path, "w:gz") as tar:
            # Add entrypoint
            tar.add(entrypoint_path, arcname="evaluate_entrypoint.sh")

            # Add evaluate.py
            evaluate_py = project_dir / "evaluate.py"
            if not evaluate_py.exists():
                raise FileNotFoundError(f"evaluate.py not found in {project_dir}")
            tar.add(evaluate_py, arcname="evaluate.py")

            # Add requirements.txt if it exists
            requirements = project_dir / "requirements.txt"
            if requirements.exists():
                tar.add(requirements, arcname="requirements.txt")

        # Move to persistent location
        output_dir = Path(tempfile.gettempdir()) / "mlctl" / project_name
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "evaluation.tar.gz"

        import shutil

        shutil.copy(tar_path, final_path)

        return final_path


def upload_code_packages(
    project_dir: Path,
    project_name: str,
    s3_bucket: str,
    s3_prefix: str,
    region: str,
) -> Dict[str, str]:
    """Package and upload training and evaluation code plus ml.yaml to S3.

    Returns dict with S3 URIs for sourcedir, evaluation packages, and ml.yaml.
    """
    s3_client = boto3.client("s3", region_name=region)

    # Package training code
    sourcedir_path = package_training_code(project_dir, project_name)
    sourcedir_key = f"{s3_prefix}/sourcedir.tar.gz"
    s3_client.upload_file(str(sourcedir_path), s3_bucket, sourcedir_key)
    sourcedir_uri = f"s3://{s3_bucket}/{sourcedir_key}"

    # Package evaluation code
    evaluation_path = package_evaluation_code(project_dir, project_name)
    evaluation_key = f"{s3_prefix}/evaluation.tar.gz"
    s3_client.upload_file(str(evaluation_path), s3_bucket, evaluation_key)
    evaluation_uri = f"s3://{s3_bucket}/{evaluation_key}"

    # Upload ml.yaml
    ml_yaml_path = project_dir / "ml.yaml"
    ml_yaml_key = f"{s3_prefix}/ml.yaml"
    s3_client.upload_file(str(ml_yaml_path), s3_bucket, ml_yaml_key)
    ml_yaml_uri = f"s3://{s3_bucket}/{ml_yaml_key}"

    return {
        "sourcedir": sourcedir_uri,
        "evaluation": evaluation_uri,
        "ml_yaml": ml_yaml_uri,
    }
