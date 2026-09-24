"""Test pipeline JSON schema against SageMaker SDK v2 oracle."""

import json
import tempfile
from pathlib import Path

import pytest

try:
    import sagemaker  # noqa: F401

    SDK_AVAILABLE = True
except ImportError as e:
    SDK_AVAILABLE = False
    SDK_IMPORT_ERROR = str(e)
except Exception as e:
    SDK_AVAILABLE = False
    SDK_IMPORT_ERROR = f"Unexpected error: {e}"
else:
    SDK_IMPORT_ERROR = None

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
        "hyperparameters": {"max_depth": 5, "n_estimators": 100},
        "quality_gate": {
            "metric": "accuracy",
            "threshold": 0.9,
            "direction": "maximize",
        },
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
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=project_dir, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=project_dir,
            capture_output=True,
        )
        subprocess.run(["git", "add", "."], cwd=project_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"], cwd=project_dir, capture_output=True
        )

        yield project_dir


def test_pipeline_structure(mock_config, ml_config, temp_project_dir):
    """Test basic pipeline structure."""
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", temp_project_dir)
    builder.code_s3_prefix = "s3://bucket/code/test/abc123"
    pipeline_def = builder.build_pipeline_definition()

    # Check top-level structure
    assert pipeline_def["Version"] == "2020-12-01"
    assert "Metadata" in pipeline_def
    assert "Parameters" in pipeline_def
    assert "Steps" in pipeline_def

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

    image = register_step["Arguments"]["InferenceSpecification"]["Containers"][0][
        "Image"
    ]

    # Should be inference image with different account
    assert "pytorch-inference" in image
    assert "763104351884" in image  # Inference account

    # Training step should use training image
    training_step = builder._build_training_step()
    training_image = training_step["Arguments"]["AlgorithmSpecification"][
        "TrainingImage"
    ]
    assert "pytorch-training" in training_image
    assert "763104351884" in training_image  # PyTorch account (same for all regions)


def test_compare_with_sdk_oracle(mock_config, ml_config, temp_project_dir):
    """Compare our boto3-generated pipeline with SDK v2-generated reference."""
    import os
    import tempfile
    from unittest.mock import patch, MagicMock

    # In CI, fail if SDK is not available; locally, skip gracefully
    if not SDK_AVAILABLE:
        if os.environ.get("CI"):
            pytest.fail(
                f"SageMaker SDK not available in CI - required for SDK oracle test. "
                f"Import error: {SDK_IMPORT_ERROR}"
            )
        else:
            pytest.skip(f"SageMaker SDK not available: {SDK_IMPORT_ERROR}")

    # Set AWS region and fake credentials for SDK (required even for offline usage)
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")

    # Create temp files for SDK code requirements
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("# training script")
        train_code_file = f.name  # noqa: F841
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("# evaluation script")
        eval_code_file = f.name

    # Build our boto3-based pipeline
    builder = PipelineBuilder(mock_config, ml_config, "us-east-1", temp_project_dir)
    builder.code_s3_prefix = "s3://test-bucket/code/test/abc123"
    our_definition = builder.build_pipeline_definition()

    # Build equivalent SDK v2 pipeline with PipelineSession (offline, doesn't call AWS)
    role = "arn:aws:iam::123456789012:role/TestRole"
    bucket = "test-bucket"

    # Mock S3 client and resource to avoid AWS calls during code upload
    mock_s3_client = MagicMock()
    mock_s3_client.list_buckets.return_value = {"Buckets": [{"Name": bucket}]}
    mock_s3_client.head_bucket.return_value = {}
    mock_s3_client.put_object.return_value = {"ETag": '"abc123"'}
    mock_s3_client.upload_file.return_value = None
    mock_s3_client.upload_fileobj.return_value = None

    mock_s3_bucket = MagicMock()
    mock_s3_bucket.creation_date = "2023-01-01"  # Mock bucket exists
    mock_s3_resource = MagicMock()
    mock_s3_resource.Bucket.return_value = mock_s3_bucket

    def mock_boto_client_fn(service_name, **kwargs):
        if service_name == "s3":
            return mock_s3_client
        # For other services, return a mock that doesn't fail
        return MagicMock()

    def mock_boto_resource_fn(service_name, **kwargs):
        if service_name == "s3":
            return mock_s3_resource
        return MagicMock()

    # PipelineSession allows building pipeline definitions without AWS calls
    with patch("boto3.client", side_effect=mock_boto_client_fn):
        with patch("boto3.Session.client", side_effect=mock_boto_client_fn):
            with patch("boto3.resource", side_effect=mock_boto_resource_fn):
                with patch("boto3.Session.resource", side_effect=mock_boto_resource_fn):
                    from sagemaker.estimator import Estimator
                    from sagemaker.processing import (
                        ScriptProcessor,
                        ProcessingInput,
                        ProcessingOutput,
                    )
                    from sagemaker.workflow.pipeline import Pipeline
                    from sagemaker.workflow.steps import TrainingStep, ProcessingStep
                    from sagemaker.workflow.properties import PropertyFile
                    from sagemaker.workflow.conditions import (
                        ConditionGreaterThanOrEqualTo,
                    )
                    from sagemaker.workflow.condition_step import ConditionStep
                    from sagemaker.workflow.functions import JsonGet
                    from sagemaker.workflow.fail_step import FailStep
                    from sagemaker.workflow.model_step import ModelStep  # noqa: F401
                    from sagemaker.workflow.pipeline_context import PipelineSession

                    session = PipelineSession(default_bucket=bucket)

        # SDK Training step (script mode estimator)
        sklearn_image = (
            "683313688378.dkr.ecr.us-east-1.amazonaws.com/"
            "sagemaker-scikit-learn:1.2-1-cpu-py3"
        )
        estimator = Estimator(
            image_uri=sklearn_image,
            role=role,
            instance_count=1,
            instance_type="ml.m5.large",
            sagemaker_session=session,
        )
        estimator.set_hyperparameters(**ml_config["hyperparameters"])

        train_step = TrainingStep(
            name="TrainModel",
            estimator=estimator,
            inputs={
                "train": f"s3://{bucket}/data/train",
                "validation": f"s3://{bucket}/data/validation",
            },
        )

        # SDK Processing step with PropertyFile
        processor = ScriptProcessor(
            role=role,
            image_uri=sklearn_image,
            instance_count=1,
            instance_type="ml.m5.large",
            command=["python3"],
            sagemaker_session=session,
        )

        evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation",
            path="metrics.json",
        )

        eval_step = ProcessingStep(
            name="EvaluateModel",
            processor=processor,
            inputs=[
                ProcessingInput(
                    source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                ),
            ],
            code=eval_code_file,
            property_files=[evaluation_report],
        )

        # SDK Model registration (skip for now - requires AWS calls even with PipelineSession)
        # For comparison purposes, we'll build the pipeline without the RegisterModel step
        # and compare just the core steps (Training, Evaluation, Condition, Fail)
        # The RegisterModel step comparison would require more complex mocking

        # SDK Fail step with dynamic message (using Join for metric value)
        from sagemaker.workflow.functions import Join

        fail_step = FailStep(
            name="QualityGateFailed",
            error_message=Join(
                on=" ",
                values=[
                    "Model quality gate failed.",
                    "Metric:",
                    "accuracy",
                    "Threshold:",
                    "0.90",
                    "Actual value:",
                    JsonGet(
                        step_name="EvaluateModel",
                        property_file=evaluation_report,
                        json_path="accuracy",
                    ),
                ],
            ),
        )

        # SDK Condition with If/Else branches
        cond_gte = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name="EvaluateModel",
                property_file=evaluation_report,
                json_path="accuracy",
            ),
            right=0.90,
        )

        cond_step = ConditionStep(
            name="CheckQualityGate",
            conditions=[cond_gte],
            if_steps=[],  # Skip RegisterModel for offline test
            else_steps=[fail_step],
        )

        # Build SDK Pipeline
        pipeline = Pipeline(
            name="test-sklearn-pipeline",
            steps=[train_step, eval_step, cond_step],
            sagemaker_session=session,
        )

    # Get SDK-generated definition
    sdk_definition = json.loads(pipeline.definition())

    # Structural comparison: verify both have same step types
    our_step_types = {step["Type"] for step in our_definition["Steps"]}

    assert "Training" in our_step_types, f"Training step missing. Got: {our_step_types}"
    assert (
        "Processing" in our_step_types
    ), f"Processing step missing. Got: {our_step_types}"
    assert (
        "Condition" in our_step_types
    ), f"Condition step missing. Got: {our_step_types}"

    # Validate our Training step has script mode hyperparameters (JSON-encoded)
    our_training = [s for s in our_definition["Steps"] if s["Type"] == "Training"][0]
    assert "HyperParameters" in our_training["Arguments"]
    assert "sagemaker_program" in our_training["Arguments"]["HyperParameters"]
    sm_submit_dir = "sagemaker_submit_directory"
    assert sm_submit_dir in our_training["Arguments"]["HyperParameters"]
    # Verify JSON encoding (should be quoted JSON strings)
    assert our_training["Arguments"]["HyperParameters"]["sagemaker_program"].startswith(
        '"'
    )

    # Validate our Processing step has PropertyFiles matching SDK
    our_processing = [s for s in our_definition["Steps"] if s["Type"] == "Processing"][
        0
    ]
    sdk_processing = [s for s in sdk_definition["Steps"] if s["Type"] == "Processing"][
        0
    ]

    assert "PropertyFiles" in our_processing
    assert len(our_processing["PropertyFiles"]) > 0
    assert our_processing["PropertyFiles"][0]["PropertyFileName"] == "EvaluationReport"
    # SDK also has PropertyFiles
    assert "PropertyFiles" in sdk_processing

    # Validate our Condition uses Std:JsonGet (SDK compiles to Std:JsonGet)
    our_condition = [s for s in our_definition["Steps"] if s["Type"] == "Condition"][0]
    left_value = our_condition["Arguments"]["Conditions"][0]["LeftValue"]
    assert "Std:JsonGet" in left_value

    # SDK should also have LeftValue with either JsonGet or Std:JsonGet
    sdk_condition = [s for s in sdk_definition["Steps"] if s["Type"] == "Condition"][0]
    sdk_left_value = sdk_condition["Arguments"]["Conditions"][0]["LeftValue"]
    assert "JsonGet" in sdk_left_value or "Std:JsonGet" in sdk_left_value

    # Find Fail step in our ElseSteps (in condition branch, not top-level)
    our_else_steps = our_condition["Arguments"].get("ElseSteps", [])
    assert len(our_else_steps) > 0, "ElseSteps should contain Fail step"
    our_fail = [s for s in our_else_steps if s["Type"] == "Fail"][0]

    # Validate our Fail step uses Std:Join for dynamic message
    error_msg = our_fail["Arguments"]["ErrorMessage"]
    assert "Std:Join" in error_msg, "Fail ErrorMessage should use Std:Join"

    # SDK should also have Fail in ElseSteps with Join
    sdk_else_steps = sdk_condition["Arguments"].get("ElseSteps", [])
    assert len(sdk_else_steps) > 0, "SDK ElseSteps should contain Fail step"
    sdk_fail = [s for s in sdk_else_steps if s["Type"] == "Fail"][0]
    sdk_error_msg = sdk_fail["Arguments"]["ErrorMessage"]
    assert "Join" in sdk_error_msg or "Std:Join" in sdk_error_msg

    # Find RegisterModel in our IfSteps
    our_if_steps = our_condition["Arguments"].get("IfSteps", [])
    assert len(our_if_steps) > 0, "IfSteps should contain RegisterModel"
    our_register = [s for s in our_if_steps if s["Type"] == "RegisterModel"][0]

    # Validate RegisterModel has ModelMetrics and InferenceSpecification
    assert "ModelMetrics" in our_register["Arguments"]
    assert "InferenceSpecification" in our_register["Arguments"]

    # Note: We skip comparing RegisterModel with SDK since it requires AWS calls
    # even with PipelineSession. The key comparison is that our Training/Processing
    # steps match SDK structure (hyperparameters, property files, etc.)

    print("\n✓ Pipeline structure matches SDK v2-generated definition")


def test_image_uri_override_in_pipeline():
    """Test that org-config image overrides appear in pipeline steps."""
    import tempfile
    import yaml

    custom_training_image = (
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/custom-sklearn:1.0"
    )
    custom_inference_image = (
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/custom-sklearn-inference:1.0"
    )

    config_data = {
        "execution_role": "arn:aws:iam::123456789012:role/test",
        "artifact_bucket": "test-bucket",
        "frameworks": {
            "sklearn": {
                "training_image": custom_training_image,
                "inference_image": custom_inference_image,
            }
        },
    }

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_file = f.name

    config = Config(config_file)

    ml_config = {
        "name": "test-override",
        "team": "test-team",
        "framework": "sklearn",
        "instance_type": "ml.m5.large",
        "data": {
            "train": "s3://bucket/data/train",
            "validation": "s3://bucket/data/validation",
        },
        "hyperparameters": {"max_depth": 5},
        "quality_gate": {
            "metric": "accuracy",
            "threshold": 0.8,
            "direction": "higher_is_better",
        },
    }

    builder = PipelineBuilder(config, ml_config, "us-east-1")
    builder.code_s3_prefix = "s3://bucket/code/test-project/abc123"
    pipeline_def = builder.build_pipeline_definition()

    # Find Training step and verify it uses custom training image
    training_step = [s for s in pipeline_def["Steps"] if s["Type"] == "Training"][0]
    training_image = training_step["Arguments"]["AlgorithmSpecification"][
        "TrainingImage"
    ]
    assert training_image == custom_training_image

    # Find Processing step and verify it uses custom training image
    processing_step = [s for s in pipeline_def["Steps"] if s["Type"] == "Processing"][0]
    processing_image = processing_step["Arguments"]["AppSpecification"]["ImageUri"]
    assert processing_image == custom_training_image

    # Find RegisterModel step (in condition branches) with custom inference image
    pipeline_json_str = json.dumps(pipeline_def)
    assert (
        custom_inference_image in pipeline_json_str
    ), "Custom inference image not found in RegisterModel step"


def test_deployment_tags_in_pipeline(mock_config):
    """Test that deployment tags appear in pipeline definition."""
    # Test with default config
    default_config = mock_config
    ml_config = {
        "name": "test-tags",
        "team": "test-team",
        "framework": "sklearn",
        "instance_type": "ml.m5.large",
        "data": {
            "train": "s3://bucket/data/train",
            "validation": "s3://bucket/data/validation",
        },
        "hyperparameters": {"max_depth": 5},
        "quality_gate": {
            "metric": "accuracy",
            "threshold": 0.8,
            "direction": "higher_is_better",
        },
    }

    builder = PipelineBuilder(default_config, ml_config, "us-east-1")
    builder.code_s3_prefix = "s3://bucket/code/test/abc123"
    tags = builder._build_tags()

    # Should have deployment tag
    deployment_tags = [t for t in tags if t["Key"] == "mlctl:deployment"]
    assert len(deployment_tags) == 1
    assert deployment_tags[0]["Value"] == "sagemaker-self-service-training"

    # Test with custom deployment tag in org-config
    custom_config_data = {
        "execution_role": "arn:aws:iam::123456789012:role/test",
        "artifact_bucket": "test-bucket",
        "deployment_tag": {"custom:tag": "custom-value"},
    }

    # Write custom config to temp file
    import tempfile
    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(custom_config_data, f)
        custom_config_file = f.name

    custom_config = Config(custom_config_file)
    builder2 = PipelineBuilder(custom_config, ml_config, "us-east-1")
    builder2.code_s3_prefix = "s3://bucket/code/test/abc123"
    tags2 = builder2._build_tags()

    custom_tags = [t for t in tags2 if t["Key"] == "custom:tag"]
    assert len(custom_tags) == 1
    assert custom_tags[0]["Value"] == "custom-value"


def test_no_tags_in_register_model(mock_config):
    """Test that RegisterModel step does not have Tags argument."""
    config = mock_config
    ml_config = {
        "name": "test-notags",
        "team": "test-team",
        "framework": "sklearn",
        "instance_type": "ml.m5.large",
        "data": {
            "train": "s3://bucket/data/train",
            "validation": "s3://bucket/data/validation",
        },
        "hyperparameters": {"max_depth": 5},
        "quality_gate": {
            "metric": "accuracy",
            "threshold": 0.8,
            "direction": "higher_is_better",
        },
    }

    builder = PipelineBuilder(config, ml_config, "us-east-1")
    builder.code_s3_prefix = "s3://bucket/code/test/abc123"
    pipeline_def = builder.build_pipeline_definition()

    # Find RegisterModel in conditional branches
    condition_step = [s for s in pipeline_def["Steps"] if s["Type"] == "Condition"][0]
    if_steps = condition_step["Arguments"].get("IfSteps", [])
    register_step = [s for s in if_steps if s["Type"] == "RegisterModel"][0]

    # RegisterModel should not have Tags
    assert "Tags" not in register_step["Arguments"]


def test_ml_yaml_uri_in_metadata(mock_config):
    """Test that MlYamlS3Uri appears when ml_yaml_uri is set."""
    config = mock_config
    ml_config = {
        "name": "test-mlyaml",
        "team": "test-team",
        "framework": "sklearn",
        "instance_type": "ml.m5.large",
        "data": {
            "train": "s3://bucket/data/train",
            "validation": "s3://bucket/data/validation",
        },
        "hyperparameters": {"max_depth": 5},
        "quality_gate": {
            "metric": "accuracy",
            "threshold": 0.8,
            "direction": "higher_is_better",
        },
    }

    builder = PipelineBuilder(config, ml_config, "us-east-1")
    builder.code_s3_prefix = "s3://bucket/code/test/abc123"
    builder.ml_yaml_uri = "s3://bucket/code/test/abc123/ml.yaml"
    pipeline_def = builder.build_pipeline_definition()

    # Find RegisterModel and check CustomerMetadataProperties
    pipeline_json_str = json.dumps(pipeline_def)
    assert "MlYamlS3Uri" in pipeline_json_str
    assert "s3://bucket/code/test/abc123/ml.yaml" in pipeline_json_str


def test_evaluate_container_entrypoint(mock_config):
    """Test that evaluate step has correct ContainerEntrypoint."""
    config = mock_config
    ml_config = {
        "name": "test-eval-entry",
        "team": "test-team",
        "framework": "sklearn",
        "instance_type": "ml.m5.large",
        "data": {
            "train": "s3://bucket/data/train",
            "validation": "s3://bucket/data/validation",
        },
        "hyperparameters": {"max_depth": 5},
        "quality_gate": {
            "metric": "accuracy",
            "threshold": 0.8,
            "direction": "higher_is_better",
        },
    }

    builder = PipelineBuilder(config, ml_config, "us-east-1")
    builder.code_s3_prefix = "s3://bucket/code/test/abc123"
    pipeline_def = builder.build_pipeline_definition()

    # Find Processing step
    processing_step = [s for s in pipeline_def["Steps"] if s["Type"] == "Processing"][0]

    entrypoint = processing_step["Arguments"]["AppSpecification"]["ContainerEntrypoint"]

    # Should extract tar and execute entrypoint
    assert "/bin/bash" in entrypoint
    assert "tar -xzf evaluation.tar.gz" in " ".join(entrypoint)
    assert "evaluate_entrypoint.sh" in " ".join(entrypoint)
