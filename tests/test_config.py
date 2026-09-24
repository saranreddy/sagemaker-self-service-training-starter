"""Tests for config module."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from mlctl.config import Config


def test_default_org_config():
    """Test default org config generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(org_config_path=os.path.join(tmpdir, "nonexistent.yaml"))

        assert "frameworks" in config.org_config
        assert "sklearn" in config.org_config["frameworks"]
        assert "xgboost" in config.org_config["frameworks"]
        assert "pytorch" in config.org_config["frameworks"]


def test_load_org_config():
    """Test loading org config from file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "org-config.yaml"

        org_config = {
            "frameworks": {"sklearn": {"version": "1.0"}},
            "execution_role": "arn:aws:iam::123456789012:role/test-role",
            "artifact_bucket": "test-bucket",
        }

        with open(config_path, "w") as f:
            yaml.dump(org_config, f)

        config = Config(org_config_path=str(config_path))

        assert (
            config.org_config["execution_role"]
            == "arn:aws:iam::123456789012:role/test-role"
        )
        assert config.org_config["artifact_bucket"] == "test-bucket"


def test_get_framework_config():
    """Test getting framework configuration."""
    config = Config()

    sklearn_config = config.get_framework_config("sklearn")
    assert "version" in sklearn_config

    with pytest.raises(ValueError):
        config.get_framework_config("unsupported")


def test_is_instance_type_allowed():
    """Test instance type allowlist checking."""
    config = Config()

    assert config.is_instance_type_allowed("ml.m5.large")
    assert not config.is_instance_type_allowed("ml.p3.8xlarge")


def test_resolve_training_image_uri():
    """Test training image URI resolution."""
    config = Config()

    uri = config.resolve_training_image_uri("sklearn", "us-east-1")

    assert "683313688378" in uri
    assert "us-east-1" in uri
    assert "scikit-learn" in uri


def test_resolve_inference_image_uri_pytorch():
    """Test PyTorch uses different inference image."""
    config = Config()

    inference_uri = config.resolve_inference_image_uri("pytorch", "us-east-1")
    training_uri = config.resolve_training_image_uri("pytorch", "us-east-1")

    assert "pytorch-inference" in inference_uri
    assert "763104351884" in inference_uri  # Inference account

    assert "pytorch-training" in training_uri
    assert "763104351884" in training_uri  # PyTorch account (same for all regions)


def test_image_uri_override_string():
    """Test org-config image URI override with string URI."""
    training_image = (
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/custom-sklearn:latest"
    )
    inference_image = (
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
        "custom-sklearn-inference:latest"
    )

    config_data = {
        "execution_role": "arn:aws:iam::123456789012:role/test",
        "artifact_bucket": "test-bucket",
        "frameworks": {
            "sklearn": {
                "training_image": training_image,
                "inference_image": inference_image,
            }
        },
    }

    config = Config(config_data)

    training_uri = config.resolve_training_image_uri("sklearn", "us-east-1")
    inference_uri = config.resolve_inference_image_uri("sklearn", "us-east-1")

    assert training_uri == training_image
    assert inference_uri == inference_image


def test_image_uri_override_region_map():
    """Test org-config image URI override with region map."""
    us_image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/custom-xgboost:latest"
    eu_image = "123456789012.dkr.ecr.eu-west-1.amazonaws.com/custom-xgboost:latest"

    config_data = {
        "execution_role": "arn:aws:iam::123456789012:role/test",
        "artifact_bucket": "test-bucket",
        "frameworks": {
            "xgboost": {
                "training_image": {
                    "us-east-1": us_image,
                    "eu-west-1": eu_image,
                },
            }
        },
    }

    config = Config(config_data)

    us_uri = config.resolve_training_image_uri("xgboost", "us-east-1")
    eu_uri = config.resolve_training_image_uri("xgboost", "eu-west-1")
    fallback_uri = config.resolve_training_image_uri("xgboost", "ap-south-1")

    assert us_uri == us_image
    assert eu_uri == eu_image
    # Falls back to default for ap-south-1
    assert "141502667606" in fallback_uri
