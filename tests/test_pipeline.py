"""Tests for pipeline module."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from mlctl.config import Config
from mlctl.pipeline import PipelineBuilder


@pytest.fixture
def mock_config():
    """Create a mock config."""
    config = Config()
    config.org_config["execution_role"] = "arn:aws:iam::123456789012:role/test-role"
    config.org_config["artifact_bucket"] = "test-bucket"
    return config


@pytest.fixture
def ml_config():
    """Create a test ML config."""
    return {
        "name": "test-project",
        "team": "test-team",
        "framework": "sklearn",
        "instance_type": "ml.m5.large",
        "data": {
            "train": "s3://bucket/train/",
            "validation": "s3://bucket/validation/",
        },
        "hyperparameters": {"max_depth": "5", "n_estimators": "100"},
        "quality_gate": {
            "metric": "accuracy",
            "threshold": 0.9,
            "direction": "maximize",
        },
    }


def test_build_pipeline_definition(mock_config, ml_config):
    """Test pipeline definition generation."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", "123456789012")

    pipeline_def = builder.build_pipeline_definition()

    assert "Version" in pipeline_def
    assert "Steps" in pipeline_def
    assert len(pipeline_def["Steps"]) >= 3

    step_names = [step["Name"] for step in pipeline_def["Steps"]]
    assert "TrainModel" in step_names
    assert "EvaluateModel" in step_names
    assert "QualityGateCheck" in step_names


def test_training_step(mock_config, ml_config):
    """Test training step generation."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", "123456789012")

    training_step = builder._build_training_step()

    assert training_step["Name"] == "TrainModel"
    assert training_step["Type"] == "Training"
    assert "AlgorithmSpecification" in training_step["Arguments"]
    assert training_step["Arguments"]["ResourceConfig"]["InstanceType"] == "ml.m5.large"

    hyperparameters = training_step["Arguments"]["HyperParameters"]
    assert hyperparameters["max_depth"] == "5"
    assert hyperparameters["n_estimators"] == "100"


def test_evaluation_step(mock_config, ml_config):
    """Test evaluation step generation."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", "123456789012")

    evaluation_step = builder._build_evaluation_step()

    assert evaluation_step["Name"] == "EvaluateModel"
    assert evaluation_step["Type"] == "Processing"
    assert "ProcessingResources" in evaluation_step["Arguments"]
    assert evaluation_step["DependsOn"] == ["TrainModel"]


def test_condition_step_maximize(mock_config, ml_config):
    """Test condition step with maximize direction."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", "123456789012")

    condition_step = builder._build_condition_step()

    assert condition_step["Name"] == "QualityGateCheck"
    assert condition_step["Type"] == "Condition"
    assert "Conditions" in condition_step["Arguments"]

    condition = condition_step["Arguments"]["Conditions"][0]
    assert "GreaterThanOrEqualTo" in condition
    assert condition["GreaterThanOrEqualTo"] == 0.9


def test_condition_step_minimize(mock_config, ml_config):
    """Test condition step with minimize direction."""
    ml_config["quality_gate"]["direction"] = "minimize"
    ml_config["quality_gate"]["threshold"] = 0.1

    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", "123456789012")

    condition_step = builder._build_condition_step()

    condition = condition_step["Arguments"]["Conditions"][0]
    assert "LessThanOrEqualTo" in condition
    assert condition["LessThanOrEqualTo"] == 0.1


def test_register_model_step(mock_config, ml_config):
    """Test model registration step."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", "123456789012")

    register_step = builder._build_register_model_step()

    assert register_step["Name"] == "RegisterModel"
    assert register_step["Type"] == "RegisterModel"
    assert register_step["Arguments"]["ModelApprovalStatus"] == "PendingManualApproval"
    assert register_step["Arguments"]["ModelPackageGroupName"] == "test-project-models"

    metadata = register_step["Arguments"]["CustomerMetadataProperties"]
    assert metadata["ProjectName"] == "test-project"
    assert metadata["Team"] == "test-team"


def test_fail_step(mock_config, ml_config):
    """Test fail step generation."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", "123456789012")

    fail_step = builder._build_fail_step()

    assert fail_step["Name"] == "QualityGateFailed"
    assert fail_step["Type"] == "Fail"
    assert "ErrorMessage" in fail_step["Arguments"]
    assert "accuracy" in fail_step["Arguments"]["ErrorMessage"]


def test_preprocessing_step(mock_config, ml_config):
    """Test preprocessing step generation."""
    ml_config["enable_preprocessing"] = True

    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", "123456789012")

    preprocess_step = builder._build_preprocessing_step()

    assert preprocess_step["Name"] == "PreprocessData"
    assert preprocess_step["Type"] == "Processing"


def test_build_tags(mock_config, ml_config):
    """Test tag generation."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", "123456789012")

    tags = builder._build_tags()

    tag_dict = {tag["Key"]: tag["Value"] for tag in tags}
    assert tag_dict["Project"] == "test-project"
    assert tag_dict["Team"] == "test-team"
    assert tag_dict["Owner"] == "mlctl"


@patch("subprocess.run")
def test_get_git_commit(mock_run, mock_config, ml_config):
    """Test git commit hash retrieval."""
    mock_run.return_value = MagicMock(stdout="abc123def\n")

    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", "123456789012")

    commit = builder._get_git_commit()

    assert commit == "abc123def"


def test_pipeline_name(mock_config, ml_config):
    """Test pipeline name generation."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", "123456789012")

    assert builder.pipeline_name == "test-project-pipeline"
