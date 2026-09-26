"""Unit tests for model package group creation."""

import pytest
from botocore.stub import Stubber
from botocore.exceptions import ClientError
import boto3

from mlctl.pipeline import PipelineBuilder


@pytest.fixture
def pipeline_builder():
    """Create a minimal pipeline builder for testing."""
    config = type(
        "Config",
        (),
        {
            "get_artifact_bucket": lambda self, team: "test-bucket",
            "get_deployment_tags": lambda self: {"mlctl:deployment": "test"},
            "org_config": {},
        },
    )()

    ml_config = {
        "name": "test-project",
        "team": "test-team",
        "owner": "test-owner",
        "framework": "sklearn",
        "hyperparameters": {},
        "quality_gate": {
            "metric": "accuracy",
            "threshold": 0.75,
            "direction": "maximize",
        },
    }

    return PipelineBuilder(
        config=config,
        ml_config=ml_config,
        region="us-east-1",
    )


def test_ensure_model_package_group_exists(pipeline_builder):
    """Test when model package group already exists."""
    client = boto3.client("sagemaker", region_name="us-east-1")
    stubber = Stubber(client)

    group_arn = (
        "arn:aws:sagemaker:us-east-1:123456789012:"
        "model-package-group/test-project-models"
    )

    # Group exists, so it returns info
    stubber.add_response(
        "describe_model_package_group",
        {
            "ModelPackageGroupName": "test-project-models",
            "ModelPackageGroupArn": group_arn,
            "CreationTime": "2026-01-01T00:00:00Z",
            "CreatedBy": {},
            "ModelPackageGroupStatus": "Completed",
        },
    )

    # Then add_tags is called to ensure tags are present
    stubber.add_response(
        "add_tags",
        {},
        expected_params={
            "ResourceArn": group_arn,
            "Tags": [
                {"Key": "Project", "Value": "test-project"},
                {"Key": "Team", "Value": "test-team"},
                {"Key": "Owner", "Value": "test-owner"},
                {"Key": "ManagedBy", "Value": "mlctl"},
                {"Key": "mlctl:deployment", "Value": "test"},
            ],
        },
    )

    with stubber:
        pipeline_builder.ensure_model_package_group(client)

    stubber.assert_no_pending_responses()


def test_ensure_model_package_group_validation_exception_does_not_exist(
    pipeline_builder,
):
    """Test creating group when describe returns ValidationException with 'does not exist'."""
    client = boto3.client("sagemaker", region_name="us-east-1")
    stubber = Stubber(client)

    # Describe returns ValidationException with "does not exist" message
    stubber.add_client_error(
        "describe_model_package_group",
        service_error_code="ValidationException",
        service_message=(
            "ModelPackageGroup arn:aws:sagemaker:us-east-1:123456789012:"
            "model-package-group/test-project-models does not exist."
        ),
    )

    # Should create the group with tags
    stubber.add_response(
        "create_model_package_group",
        {
            "ModelPackageGroupArn": (
                "arn:aws:sagemaker:us-east-1:123456789012:"
                "model-package-group/test-project-models"
            ),
        },
        expected_params={
            "ModelPackageGroupName": "test-project-models",
            "ModelPackageGroupDescription": "Models for test-project",
            "Tags": [
                {"Key": "Project", "Value": "test-project"},
                {"Key": "Team", "Value": "test-team"},
                {"Key": "Owner", "Value": "test-owner"},
                {"Key": "ManagedBy", "Value": "mlctl"},
                {"Key": "mlctl:deployment", "Value": "test"},
            ],
        },
    )

    with stubber:
        pipeline_builder.ensure_model_package_group(client)

    stubber.assert_no_pending_responses()


def test_ensure_model_package_group_resource_not_found(pipeline_builder):
    """Test creating group when describe returns ResourceNotFound."""
    client = boto3.client("sagemaker", region_name="us-east-1")
    stubber = Stubber(client)

    # Describe returns ResourceNotFound
    stubber.add_client_error(
        "describe_model_package_group",
        service_error_code="ResourceNotFound",
        service_message="Could not find model package group",
    )

    # Should create the group with tags
    stubber.add_response(
        "create_model_package_group",
        {
            "ModelPackageGroupArn": (
                "arn:aws:sagemaker:us-east-1:123456789012:"
                "model-package-group/test-project-models"
            ),
        },
        expected_params={
            "ModelPackageGroupName": "test-project-models",
            "ModelPackageGroupDescription": "Models for test-project",
            "Tags": [
                {"Key": "Project", "Value": "test-project"},
                {"Key": "Team", "Value": "test-team"},
                {"Key": "Owner", "Value": "test-owner"},
                {"Key": "ManagedBy", "Value": "mlctl"},
                {"Key": "mlctl:deployment", "Value": "test"},
            ],
        },
    )

    with stubber:
        pipeline_builder.ensure_model_package_group(client)

    stubber.assert_no_pending_responses()


def test_ensure_model_package_group_other_client_error_raises(pipeline_builder):
    """Test that other ClientErrors are re-raised."""
    client = boto3.client("sagemaker", region_name="us-east-1")
    stubber = Stubber(client)

    # Describe returns an unexpected error
    stubber.add_client_error(
        "describe_model_package_group",
        service_error_code="AccessDeniedException",
        service_message="User not authorized",
    )

    with stubber:
        with pytest.raises(ClientError) as exc_info:
            pipeline_builder.ensure_model_package_group(client)

        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"

    stubber.assert_no_pending_responses()


def test_ensure_model_package_group_validation_exception_other_message_raises(
    pipeline_builder,
):
    """Test that ValidationException with other messages are re-raised."""
    client = boto3.client("sagemaker", region_name="us-east-1")
    stubber = Stubber(client)

    # Describe returns ValidationException with different message
    stubber.add_client_error(
        "describe_model_package_group",
        service_error_code="ValidationException",
        service_message="Invalid parameter: ModelPackageGroupName",
    )

    with stubber:
        with pytest.raises(ClientError) as exc_info:
            pipeline_builder.ensure_model_package_group(client)

        assert exc_info.value.response["Error"]["Code"] == "ValidationException"
        assert "does not exist" not in exc_info.value.response["Error"]["Message"]

    stubber.assert_no_pending_responses()
