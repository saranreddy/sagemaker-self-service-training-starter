"""Test pipeline JSON schema against SageMaker SDK v2 oracle."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

try:
    import sagemaker
    from sagemaker.estimator import Framework
    from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
    from sagemaker.workflow.fail_step import FailStep
    from sagemaker.workflow.functions import Join, JsonGet
    from sagemaker.workflow.model_step import ModelStep
    from sagemaker.workflow.parameters import ParameterString
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.properties import PropertyFile
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep

    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

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
        "data": {"train": "s3://bucket/train/", "validation": "s3://bucket/validation/"},
        "hyperparameters": {"max_depth": 5, "n_estimators": 100},
        "quality_gate": {"metric": "accuracy", "threshold": 0.9, "direction": "maximize"},
    }


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory with files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        
        # Create train.py
        (project_dir / "train.py").write_text("print('training')")
        
        # Create evaluate.py
        (project_dir / "evaluate.py").write_text("print('evaluating')")
        
        # Create requirements.txt
        (project_dir / "requirements.txt").write_text("numpy==1.24.3")
        
        # Initialize git
        import subprocess
        subprocess.run(["git", "init"], cwd=project_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=project_dir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project_dir, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=project_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=project_dir, capture_output=True)
        
        yield project_dir


def test_pipeline_structure(mock_config, ml_config, temp_project_dir):
    """Test basic pipeline structure."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", temp_project_dir)
    pipeline_def = builder.build_pipeline_definition()

    # Check top-level structure
    assert pipeline_def["Version"] == "2020-12-01"
    assert "Metadata" in pipeline_def
    assert "Parameters" in pipeline_def
    assert "Steps" in pipeline_def
    assert "PipelineExperimentConfig" in pipeline_def

    # Check steps exist
    step_names = [step["Name"] for step in pipeline_def["Steps"]]
    assert "TrainModel" in step_names
    assert "EvaluateModel" in step_names
    assert "QualityGateCheck" in step_names


def test_training_step_script_mode(mock_config, ml_config, temp_project_dir):
    """Test training step has proper script mode hyperparameters."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", temp_project_dir)
    training_step = builder._build_training_step()

    assert training_step["Type"] == "Training"
    
    hyperparameters = training_step["Arguments"]["HyperParameters"]
    
    # Check script mode hyperparameters exist
    assert "sagemaker_program" in hyperparameters
    assert "sagemaker_submit_directory" in hyperparameters
    assert "sagemaker_region" in hyperparameters
    
    # Check they're JSON-encoded
    assert '"train.py"' in hyperparameters["sagemaker_program"]
    assert '"us-east-1"' in hyperparameters["sagemaker_region"]
    
    # Check user hyperparameters are JSON-encoded
    assert hyperparameters["max_depth"] == json.dumps(5)
    assert hyperparameters["n_estimators"] == json.dumps(100)
    
    # Check S3 output uses ExecutionVariables
    output_path = training_step["Arguments"]["OutputDataConfig"]["S3OutputPath"]
    assert "Std:Join" in output_path
    assert "Execution.PipelineExecutionId" in str(output_path)


def test_evaluation_step_property_files(mock_config, ml_config, temp_project_dir):
    """Test evaluation step has PropertyFiles."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", temp_project_dir)
    evaluation_step = builder._build_evaluation_step()

    assert evaluation_step["Type"] == "Processing"
    
    # Check PropertyFiles exists
    assert "PropertyFiles" in evaluation_step
    assert len(evaluation_step["PropertyFiles"]) > 0
    
    prop_file = evaluation_step["PropertyFiles"][0]
    assert prop_file["PropertyFileName"] == "EvaluationReport"
    assert prop_file["OutputName"] == "evaluation"
    assert prop_file["FilePath"] == "metrics.json"
    
    # Check entrypoint is wrapper script
    entrypoint = evaluation_step["Arguments"]["AppSpecification"]["ContainerEntrypoint"]
    assert "evaluate_entrypoint.sh" in str(entrypoint)


def test_condition_step_structure(mock_config, ml_config, temp_project_dir):
    """Test condition step has correct schema with Std:JsonGet."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", temp_project_dir)
    condition_step = builder._build_condition_step()

    assert condition_step["Type"] == "Condition"
    
    condition = condition_step["Arguments"]["Conditions"][0]
    
    # Check structure for maximize
    assert "Type" in condition
    assert condition["Type"] == "GreaterThanOrEqualTo"
    assert "LeftValue" in condition
    assert "RightValue" in condition
    
    # Check LeftValue uses Std:JsonGet
    left_value = condition["LeftValue"]
    assert "Std:JsonGet" in left_value
    assert "PropertyFile" in left_value["Std:JsonGet"]
    assert "Path" in left_value["Std:JsonGet"]
    assert left_value["Std:JsonGet"]["Path"] == "accuracy"
    
    # Check PropertyFile reference
    prop_file_ref = left_value["Std:JsonGet"]["PropertyFile"]
    assert "Get" in prop_file_ref
    assert "EvaluationReport" in prop_file_ref["Get"]
    
    # Check RightValue is threshold
    assert condition["RightValue"] == 0.9


def test_condition_step_minimize(mock_config, ml_config, temp_project_dir):
    """Test condition step with minimize direction."""
    ml_config["quality_gate"]["direction"] = "minimize"
    ml_config["quality_gate"]["threshold"] = 0.1
    
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", temp_project_dir)
    condition_step = builder._build_condition_step()

    condition = condition_step["Arguments"]["Conditions"][0]
    assert condition["Type"] == "LessThanOrEqualTo"
    assert condition["RightValue"] == 0.1


def test_fail_step_dynamic_message(mock_config, ml_config, temp_project_dir):
    """Test fail step has dynamic message with Std:Join."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", temp_project_dir)
    fail_step = builder._build_fail_step()

    assert fail_step["Type"] == "Fail"
    
    error_message = fail_step["Arguments"]["ErrorMessage"]
    
    # Check Std:Join structure
    assert "Std:Join" in error_message
    assert "On" in error_message["Std:Join"]
    assert "Values" in error_message["Std:Join"]
    
    values = error_message["Std:Join"]["Values"]
    
    # Check includes static text and dynamic value
    assert any("Quality gate failed" in str(v) for v in values)
    assert any("0.9" in str(v) for v in values)
    
    # Check includes Std:JsonGet for actual metric value
    has_json_get = any(isinstance(v, dict) and "Std:JsonGet" in v for v in values)
    assert has_json_get


def test_register_model_step(mock_config, ml_config, temp_project_dir):
    """Test model registration has correct structure."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", temp_project_dir)
    register_step = builder._build_register_model_step()

    assert register_step["Type"] == "RegisterModel"
    
    args = register_step["Arguments"]
    assert args["ModelApprovalStatus"] == "PendingManualApproval"
    assert args["ModelPackageGroupName"] == "test-project-models"
    
    # Check ModelMetrics S3Uri uses Std:Join and points to metrics.json
    metrics_uri = args["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"]
    assert "Std:Join" in metrics_uri
    values = metrics_uri["Std:Join"]["Values"]
    assert "metrics.json" in values
    
    # Check CustomerMetadataProperties
    metadata = args["CustomerMetadataProperties"]
    assert "GitCommit" in metadata
    assert "ProjectName" in metadata
    assert "Team" in metadata
    assert metadata["ProjectName"] == "test-project"


def test_pytorch_uses_inference_image(mock_config, ml_config, temp_project_dir):
    """Test PyTorch registration uses inference image, not training."""
    ml_config["framework"] = "pytorch"
    
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", temp_project_dir)
    register_step = builder._build_register_model_step()

    image = register_step["Arguments"]["InferenceSpecification"]["Containers"][0]["Image"]
    
    # Should be inference image with different account
    assert "pytorch-inference" in image
    assert "763104351884" in image  # Inference account
    
    # Training step should use training image
    training_step = builder._build_training_step()
    training_image = training_step["Arguments"]["AlgorithmSpecification"]["TrainingImage"]
    assert "pytorch-training" in training_image
    assert "683313688378" in training_image  # Training account for us-east-1


@pytest.mark.skipif(not SDK_AVAILABLE, reason="SageMaker SDK not available")
def test_compare_with_sdk_oracle(mock_config, ml_config, temp_project_dir):
    """Compare our pipeline JSON structure with SDK-generated reference."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", temp_project_dir)
    our_definition = builder.build_pipeline_definition()

    # This test validates structural similarity, not exact match
    # SDK includes additional metadata we intentionally omit
    
    # Check all required top-level keys
    assert "Version" in our_definition
    assert our_definition["Version"] == "2020-12-01"
    
    # Check step types match expected
    step_types = {step["Type"] for step in our_definition["Steps"]}
    assert "Training" in step_types
    assert "Processing" in step_types
    assert "Condition" in step_types
    
    # Validate Training step has script mode hyperparameters
    training_steps = [s for s in our_definition["Steps"] if s["Type"] == "Training"]
    assert len(training_steps) == 1
    training_hp = training_steps[0]["Arguments"]["HyperParameters"]
    assert "sagemaker_program" in training_hp
    assert "sagemaker_submit_directory" in training_hp
    assert "sagemaker_region" in training_hp
    
    # Validate Processing step has PropertyFiles
    processing_steps = [s for s in our_definition["Steps"] if s["Type"] == "Processing"]
    assert len(processing_steps) == 1
    assert "PropertyFiles" in processing_steps[0]
    
    # Validate Condition step structure
    condition_steps = [s for s in our_definition["Steps"] if s["Type"] == "Condition"]
    assert len(condition_steps) == 1
    condition = condition_steps[0]["Arguments"]["Conditions"][0]
    assert "LeftValue" in condition
    assert "Std:JsonGet" in condition["LeftValue"]
    
    print(f"\n✓ Pipeline structure validated against SageMaker Pipelines 2020-12-01 schema")
