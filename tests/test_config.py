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
            "frameworks": {
                "sklearn": {"container_uri_template": "test-uri", "version": "1.0"}
            },
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
    assert "container_uri_template" in sklearn_config
    assert "version" in sklearn_config

    with pytest.raises(ValueError):
        config.get_framework_config("unsupported")


def test_is_instance_type_allowed():
    """Test instance type allowlist checking."""
    config = Config()

    assert config.is_instance_type_allowed("ml.m5.large")
    assert not config.is_instance_type_allowed("ml.p3.8xlarge")


def test_resolve_container_uri():
    """Test container URI resolution."""
    config = Config()

    uri = config.resolve_container_uri("sklearn", "us-east-1", "123456789012")

    assert "123456789012" in uri
    assert "us-east-1" in uri
    assert "scikit-learn" in uri


def test_load_project_config():
    """Test loading project ml.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config()

        ml_yaml = {
            "name": "test-project",
            "team": "test-team",
            "framework": "sklearn",
            "instance_type": "ml.m5.large",
            "data": {"train": "s3://bucket/train/"},
            "hyperparameters": {"max_depth": "5"},
            "quality_gate": {
                "metric": "accuracy",
                "threshold": 0.9,
                "direction": "maximize",
            },
        }

        ml_yaml_path = Path(tmpdir) / "ml.yaml"
        with open(ml_yaml_path, "w") as f:
            yaml.dump(ml_yaml, f)

        project_config = config.load_project_config(tmpdir)

        assert project_config["name"] == "test-project"
        assert project_config["framework"] == "sklearn"


def test_get_team_config():
    """Test getting team configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "org-config.yaml"

        org_config = {
            "frameworks": {},
            "teams": {
                "data-science": {
                    "execution_role": "arn:aws:iam::123456789012:role/ds-role",
                    "artifact_bucket": "ds-bucket",
                }
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(org_config, f)

        config = Config(org_config_path=str(config_path))

        team_config = config.get_team_config("data-science")
        assert team_config["execution_role"] == "arn:aws:iam::123456789012:role/ds-role"

        assert config.get_team_config("nonexistent") is None
