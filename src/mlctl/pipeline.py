"""SageMaker pipeline generation with correct 2020-12-01 schema (boto3-only runtime)."""
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError


class PipelineBuilder:
    """Builds SageMaker pipeline definitions with correct schema."""

    def __init__(
        self, config, ml_config: Dict[str, Any], region: str, project_dir: str = "."
    ):
        self.config = config
        self.ml_config = ml_config
        self.region = region
        self.project_dir = Path(project_dir)
        self.pipeline_name = f"{ml_config['name']}-pipeline"
        self.project_name = ml_config["name"]
        self.git_commit = self._get_git_commit()

    def build_pipeline_definition(self) -> Dict[str, Any]:
        """Build complete pipeline definition JSON with correct schema."""
        steps = []

        training_step = self._build_training_step()
        steps.append(training_step)

        evaluation_step = self._build_evaluation_step()
        steps.append(evaluation_step)

        condition_step = self._build_condition_step()
        steps.append(condition_step)

        pipeline_def = {
            "Version": "2020-12-01",
            "Metadata": {},
            "Parameters": [],
            "PipelineExperimentConfig": {
                "ExperimentName": {"Get": "Execution.PipelineName"},
                "TrialName": {"Get": "Execution.PipelineExecutionId"},
            },
            "Steps": steps,
        }

        return pipeline_def

    def _build_training_step(self) -> Dict[str, Any]:
        """Build training step with proper script mode."""
        from mlctl.image_uris import get_training_image_uri

        training_image = get_training_image_uri(
            self.ml_config["framework"], self.region
        )
        execution_role = self.config.get_execution_role(self.ml_config.get("team"))
        artifact_bucket = self._get_artifact_bucket()

        # Script mode hyperparameters (must be JSON-encoded strings)
        hyperparameters = {}
        for k, v in self.ml_config.get("hyperparameters", {}).items():
            hyperparameters[k] = json.dumps(v)

        # Add required script mode hyperparameters
        code_s3_prefix = self._get_code_s3_prefix()
        hyperparameters["sagemaker_program"] = json.dumps("train.py")
        hyperparameters["sagemaker_submit_directory"] = json.dumps(
            f"s3://{artifact_bucket}/{code_s3_prefix}/sourcedir.tar.gz"
        )
        hyperparameters["sagemaker_region"] = json.dumps(self.region)

        # Use ExecutionVariables for unique output paths
        output_path = {
            "Std:Join": {
                "On": "/",
                "Values": [
                    f"s3://{artifact_bucket}/pipelines",
                    {"Get": "Execution.PipelineExecutionId"},
                    self.project_name,
                    "output",
                ],
            }
        }

        step = {
            "Name": "TrainModel",
            "Type": "Training",
            "Arguments": {
                "AlgorithmSpecification": {
                    "TrainingImage": training_image,
                    "TrainingInputMode": "File",
                },
                "RoleArn": execution_role,
                "OutputDataConfig": {"S3OutputPath": output_path},
                "ResourceConfig": {
                    "InstanceType": self.ml_config["instance_type"],
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 30,
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": self.ml_config.get("max_runtime_seconds", 3600)
                },
                "HyperParameters": hyperparameters,
                "InputDataConfig": [
                    {
                        "ChannelName": "training",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": self.ml_config["data"]["train"],
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                    }
                ],
                "Environment": {"GIT_COMMIT": self.git_commit},
                "Tags": self._build_tags(),
            },
        }

        # Add validation channel if configured
        if "validation" in self.ml_config["data"]:
            step["Arguments"]["InputDataConfig"].append(
                {
                    "ChannelName": "validation",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": self.ml_config["data"]["validation"],
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                }
            )

        return step

    def _build_evaluation_step(self) -> Dict[str, Any]:
        """Build evaluation processing step with PropertyFiles."""
        from mlctl.image_uris import get_training_image_uri

        eval_image = get_training_image_uri(self.ml_config["framework"], self.region)
        execution_role = self.config.get_execution_role(self.ml_config.get("team"))
        artifact_bucket = self._get_artifact_bucket()
        code_s3_prefix = self._get_code_s3_prefix()

        eval_output_path = {
            "Std:Join": {
                "On": "/",
                "Values": [
                    f"s3://{artifact_bucket}/pipelines",
                    {"Get": "Execution.PipelineExecutionId"},
                    self.project_name,
                    "evaluation",
                ],
            }
        }

        # Test data: use test if configured, else validation
        test_data_uri = self.ml_config["data"].get(
            "test", self.ml_config["data"].get("validation")
        )

        processing_inputs = [
            {
                "InputName": "model",
                "S3Input": {
                    "S3Uri": {"Get": "Steps.TrainModel.ModelArtifacts.S3ModelArtifacts"},
                    "LocalPath": "/opt/ml/processing/model",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": f"s3://{artifact_bucket}/{code_s3_prefix}/evaluation.tar.gz",
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ]

        # Add test data if available
        if test_data_uri:
            processing_inputs.append(
                {
                    "InputName": "test",
                    "S3Input": {
                        "S3Uri": test_data_uri,
                        "LocalPath": "/opt/ml/processing/test",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                }
            )

        step = {
            "Name": "EvaluateModel",
            "Type": "Processing",
            "Arguments": {
                "ProcessingResources": {
                    "ClusterConfig": {
                        "InstanceType": self.ml_config["instance_type"],
                        "InstanceCount": 1,
                        "VolumeSizeInGB": 30,
                    }
                },
                "AppSpecification": {
                    "ImageUri": eval_image,
                    "ContainerEntrypoint": [
                        "/bin/bash",
                        "/opt/ml/processing/input/code/evaluate_entrypoint.sh",
                    ],
                },
                "RoleArn": execution_role,
                "ProcessingInputs": processing_inputs,
                "ProcessingOutputConfig": {
                    "Outputs": [
                        {
                            "OutputName": "evaluation",
                            "S3Output": {
                                "S3Uri": eval_output_path,
                                "LocalPath": "/opt/ml/processing/evaluation",
                                "S3UploadMode": "EndOfJob",
                            },
                        }
                    ]
                },
                "Tags": self._build_tags(),
            },
            "PropertyFiles": [
                {
                    "PropertyFileName": "EvaluationReport",
                    "OutputName": "evaluation",
                    "FilePath": "metrics.json",
                }
            ],
        }

        return step

    def _build_condition_step(self) -> Dict[str, Any]:
        """Build condition step with correct Std:JsonGet structure."""
        quality_gate = self.ml_config["quality_gate"]
        metric_name = quality_gate["metric"]
        threshold = quality_gate["threshold"]
        direction = quality_gate["direction"]

        # Build condition with correct schema
        if direction == "maximize":
            condition = {
                "Type": "GreaterThanOrEqualTo",
                "LeftValue": {
                    "Std:JsonGet": {
                        "PropertyFile": {
                            "Get": "Steps.EvaluateModel.PropertyFiles.EvaluationReport"
                        },
                        "Path": metric_name,
                    }
                },
                "RightValue": threshold,
            }
        else:  # minimize
            condition = {
                "Type": "LessThanOrEqualTo",
                "LeftValue": {
                    "Std:JsonGet": {
                        "PropertyFile": {
                            "Get": "Steps.EvaluateModel.PropertyFiles.EvaluationReport"
                        },
                        "Path": metric_name,
                    }
                },
                "RightValue": threshold,
            }

        register_step = self._build_register_model_step()
        fail_step = self._build_fail_step()

        return {
            "Name": "QualityGateCheck",
            "Type": "Condition",
            "Arguments": {
                "Conditions": [condition],
                "IfSteps": [register_step],
                "ElseSteps": [fail_step],
            },
        }

    def _build_register_model_step(self) -> Dict[str, Any]:
        """Build model registration step."""
        from mlctl.image_uris import get_inference_image_uri

        inference_image = get_inference_image_uri(
            self.ml_config["framework"], self.region
        )
        model_package_group_name = f"{self.project_name}-models"
        artifact_bucket = self._get_artifact_bucket()

        # Build metrics S3 URI pointing to metrics.json
        metrics_s3_uri = {
            "Std:Join": {
                "On": "/",
                "Values": [
                    f"s3://{artifact_bucket}/pipelines",
                    {"Get": "Execution.PipelineExecutionId"},
                    self.project_name,
                    "evaluation",
                    "metrics.json",
                ],
            }
        }

        # Customer metadata with ml.yaml and git commit
        customer_metadata = {
            "GitCommit": self.git_commit,
            "ProjectName": self.project_name,
            "Team": self.ml_config["team"],
            "Framework": self.ml_config["framework"],
            "Owner": self.ml_config.get("owner", "mlctl"),
            "QualityGateMetric": self.ml_config["quality_gate"]["metric"],
            "QualityGateThreshold": str(self.ml_config["quality_gate"]["threshold"]),
            "QualityGateDirection": self.ml_config["quality_gate"]["direction"],
        }

        return {
            "Name": "RegisterModel",
            "Type": "RegisterModel",
            "Arguments": {
                "ModelPackageGroupName": model_package_group_name,
                "ModelApprovalStatus": "PendingManualApproval",
                "InferenceSpecification": {
                    "Containers": [
                        {
                            "Image": inference_image,
                            "ModelDataUrl": {
                                "Get": "Steps.TrainModel.ModelArtifacts.S3ModelArtifacts"
                            },
                        }
                    ],
                    "SupportedContentTypes": ["application/json", "text/csv"],
                    "SupportedResponseMIMETypes": ["application/json"],
                    "SupportedRealtimeInferenceInstanceTypes": [
                        "ml.t2.medium",
                        "ml.m5.large",
                        "ml.m5.xlarge",
                    ],
                },
                "ModelMetrics": {
                    "ModelQuality": {
                        "Statistics": {
                            "ContentType": "application/json",
                            "S3Uri": metrics_s3_uri,
                        }
                    }
                },
                "CustomerMetadataProperties": customer_metadata,
                "Tags": self._build_tags(),
            },
        }

    def _build_fail_step(self) -> Dict[str, Any]:
        """Build fail step with dynamic message using Std:Join."""
        quality_gate = self.ml_config["quality_gate"]
        metric_name = quality_gate["metric"]
        threshold = quality_gate["threshold"]
        direction = quality_gate["direction"]

        # Build dynamic error message with actual metric value
        error_message = {
            "Std:Join": {
                "On": "",
                "Values": [
                    f"Quality gate failed. Metric '{metric_name}' did not meet threshold ",
                    str(threshold),
                    f" (direction: {direction}). Actual value: ",
                    {
                        "Std:JsonGet": {
                            "PropertyFile": {
                                "Get": "Steps.EvaluateModel.PropertyFiles.EvaluationReport"
                            },
                            "Path": metric_name,
                        }
                    },
                ],
            }
        }

        return {
            "Name": "QualityGateFailed",
            "Type": "Fail",
            "Arguments": {"ErrorMessage": error_message},
        }

    def _build_tags(self) -> list:
        """Build tags for resources."""
        tags = [
            {"Key": "Project", "Value": self.project_name},
            {"Key": "Team", "Value": self.ml_config["team"]},
            {"Key": "Owner", "Value": self.ml_config.get("owner", "mlctl")},
            {"Key": "ManagedBy", "Value": "mlctl"},
        ]

        for key, value in self.config.org_config.get("required_tags", {}).items():
            if not any(t["Key"] == key for t in tags):
                tags.append({"Key": key, "Value": value})

        return tags

    def _get_artifact_bucket(self) -> str:
        """Get artifact bucket name."""
        bucket = self.config.get_artifact_bucket(self.ml_config.get("team"))
        if not bucket:
            bucket = os.environ.get("MLCTL_ARTIFACT_BUCKET")
        if not bucket:
            raise ValueError(
                "Artifact bucket not configured. Set in org-config.yaml or MLCTL_ARTIFACT_BUCKET env var."
            )
        return bucket

    def _get_code_s3_prefix(self) -> str:
        """Get S3 prefix for code with content hash."""
        code_hash = hashlib.sha256(self.git_commit.encode()).hexdigest()[:12]
        return f"code/{self.project_name}/{self.git_commit[:8]}-{code_hash}"

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.project_dir,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def create_or_update_pipeline(
        self, sagemaker_client
    ) -> str:
        """Create or update the pipeline in SageMaker."""
        pipeline_def = self.build_pipeline_definition()
        execution_role = self.config.get_execution_role(self.ml_config.get("team"))

        try:
            sagemaker_client.describe_pipeline(PipelineName=self.pipeline_name)

            response = sagemaker_client.update_pipeline(
                PipelineName=self.pipeline_name,
                PipelineDefinition=json.dumps(pipeline_def),
                RoleArn=execution_role,
            )

            return response["PipelineArn"]

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFound":
                response = sagemaker_client.create_pipeline(
                    PipelineName=self.pipeline_name,
                    PipelineDefinition=json.dumps(pipeline_def),
                    RoleArn=execution_role,
                    Tags=self._build_tags(),
                )

                return response["PipelineArn"]
            else:
                raise

    def start_pipeline_execution(self, sagemaker_client) -> str:
        """Start a pipeline execution."""
        response = sagemaker_client.start_pipeline_execution(
            PipelineName=self.pipeline_name,
            PipelineExecutionDisplayName=f"{self.project_name}-{self.git_commit[:8]}",
        )

        return response["PipelineExecutionArn"]

    def ensure_model_package_group(self, sagemaker_client):
        """Ensure model package group exists with proper tags."""
        group_name = f"{self.project_name}-models"

        try:
            sagemaker_client.describe_model_package_group(
                ModelPackageGroupName=group_name
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFound":
                sagemaker_client.create_model_package_group(
                    ModelPackageGroupName=group_name,
                    ModelPackageGroupDescription=f"Models for {self.project_name}",
                    Tags=self._build_tags(),
                )
