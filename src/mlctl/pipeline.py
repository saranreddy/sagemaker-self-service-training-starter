"""SageMaker pipeline generation using boto3 (no SDK dependency)."""
import json
import os
import subprocess
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError


class PipelineBuilder:
    """Builds SageMaker pipeline definitions directly with boto3."""
    
    def __init__(self, config, ml_config: Dict[str, Any], region: str, account: str):
        self.config = config
        self.ml_config = ml_config
        self.region = region
        self.account = account
        self.pipeline_name = f"{ml_config['name']}-pipeline"
    
    def build_pipeline_definition(self) -> Dict[str, Any]:
        """Build complete pipeline definition JSON."""
        steps = []
        
        if self.ml_config.get("enable_preprocessing", False):
            preprocess_step = self._build_preprocessing_step()
            steps.append(preprocess_step)
        
        training_step = self._build_training_step()
        steps.append(training_step)
        
        evaluation_step = self._build_evaluation_step()
        steps.append(evaluation_step)
        
        condition_step = self._build_condition_step()
        steps.append(condition_step)
        
        pipeline_def = {
            "Version": "2020-12-01",
            "Parameters": self._build_parameters(),
            "Steps": steps
        }
        
        return pipeline_def
    
    def _build_parameters(self) -> list:
        """Build pipeline parameters."""
        return [
            {
                "Name": "InputDataUri",
                "Type": "String",
                "DefaultValue": self.ml_config["data"]["train"]
            }
        ]
    
    def _build_training_step(self) -> Dict[str, Any]:
        """Build training step."""
        framework = self.ml_config["framework"]
        container_uri = self.config.resolve_container_uri(framework, self.region, self.account)
        execution_role = self.config.get_execution_role(self.ml_config.get("team"))
        
        hyperparameters_str = {
            k: str(v) for k, v in self.ml_config.get("hyperparameters", {}).items()
        }
        
        git_commit = self._get_git_commit()
        
        step = {
            "Name": "TrainModel",
            "Type": "Training",
            "Arguments": {
                "AlgorithmSpecification": {
                    "TrainingImage": container_uri,
                    "TrainingInputMode": "File"
                },
                "RoleArn": execution_role,
                "OutputDataConfig": {
                    "S3OutputPath": f"s3://{self._get_artifact_bucket()}/output"
                },
                "ResourceConfig": {
                    "InstanceType": self.ml_config["instance_type"],
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 30
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                },
                "HyperParameters": hyperparameters_str,
                "InputDataConfig": [
                    {
                        "ChannelName": "train",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": self.ml_config["data"]["train"],
                                "S3DataDistributionType": "FullyReplicated"
                            }
                        }
                    }
                ],
                "Environment": {
                    "GIT_COMMIT": git_commit
                },
                "Tags": self._build_tags()
            }
        }
        
        if "validation" in self.ml_config["data"]:
            step["Arguments"]["InputDataConfig"].append({
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": self.ml_config["data"]["validation"],
                        "S3DataDistributionType": "FullyReplicated"
                    }
                }
            })
        
        return step
    
    def _build_evaluation_step(self) -> Dict[str, Any]:
        """Build evaluation processing step."""
        framework = self.ml_config["framework"]
        container_uri = self.config.resolve_container_uri(framework, self.region, self.account)
        execution_role = self.config.get_execution_role(self.ml_config.get("team"))
        
        step = {
            "Name": "EvaluateModel",
            "Type": "Processing",
            "Arguments": {
                "ProcessingResources": {
                    "ClusterConfig": {
                        "InstanceType": self.ml_config["instance_type"],
                        "InstanceCount": 1,
                        "VolumeSizeInGB": 30
                    }
                },
                "AppSpecification": {
                    "ImageUri": container_uri,
                    "ContainerEntrypoint": [
                        "python3",
                        "/opt/ml/processing/input/code/evaluate.py"
                    ]
                },
                "RoleArn": execution_role,
                "ProcessingInputs": [
                    {
                        "InputName": "model",
                        "S3Input": {
                            "S3Uri": {
                                "Get": "Steps.TrainModel.ModelArtifacts.S3ModelArtifacts"
                            },
                            "LocalPath": "/opt/ml/processing/model",
                            "S3DataType": "S3Prefix",
                            "S3InputMode": "File"
                        }
                    },
                    {
                        "InputName": "code",
                        "S3Input": {
                            "S3Uri": f"s3://{self._get_artifact_bucket()}/code/evaluate.py",
                            "LocalPath": "/opt/ml/processing/input/code",
                            "S3DataType": "S3Prefix",
                            "S3InputMode": "File"
                        }
                    }
                ],
                "ProcessingOutputConfig": {
                    "Outputs": [
                        {
                            "OutputName": "evaluation",
                            "S3Output": {
                                "S3Uri": f"s3://{self._get_artifact_bucket()}/evaluation",
                                "LocalPath": "/opt/ml/processing/evaluation",
                                "S3UploadMode": "EndOfJob"
                            }
                        }
                    ]
                },
                "Tags": self._build_tags()
            },
            "DependsOn": ["TrainModel"]
        }
        
        if "test" in self.ml_config["data"]:
            step["Arguments"]["ProcessingInputs"].append({
                "InputName": "test",
                "S3Input": {
                    "S3Uri": self.ml_config["data"]["test"],
                    "LocalPath": "/opt/ml/processing/test",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            })
        
        return step
    
    def _build_preprocessing_step(self) -> Dict[str, Any]:
        """Build preprocessing step (optional)."""
        framework = self.ml_config["framework"]
        container_uri = self.config.resolve_container_uri(framework, self.region, self.account)
        execution_role = self.config.get_execution_role(self.ml_config.get("team"))
        
        return {
            "Name": "PreprocessData",
            "Type": "Processing",
            "Arguments": {
                "ProcessingResources": {
                    "ClusterConfig": {
                        "InstanceType": self.ml_config["instance_type"],
                        "InstanceCount": 1,
                        "VolumeSizeInGB": 30
                    }
                },
                "AppSpecification": {
                    "ImageUri": container_uri,
                    "ContainerEntrypoint": [
                        "python3",
                        "/opt/ml/processing/input/code/preprocess.py"
                    ]
                },
                "RoleArn": execution_role,
                "ProcessingInputs": [
                    {
                        "InputName": "input",
                        "S3Input": {
                            "S3Uri": self.ml_config["data"]["train"],
                            "LocalPath": "/opt/ml/processing/input",
                            "S3DataType": "S3Prefix",
                            "S3InputMode": "File"
                        }
                    },
                    {
                        "InputName": "code",
                        "S3Input": {
                            "S3Uri": f"s3://{self._get_artifact_bucket()}/code/preprocess.py",
                            "LocalPath": "/opt/ml/processing/input/code",
                            "S3DataType": "S3Prefix",
                            "S3InputMode": "File"
                        }
                    }
                ],
                "ProcessingOutputConfig": {
                    "Outputs": [
                        {
                            "OutputName": "train",
                            "S3Output": {
                                "S3Uri": f"s3://{self._get_artifact_bucket()}/preprocessed/train",
                                "LocalPath": "/opt/ml/processing/train",
                                "S3UploadMode": "EndOfJob"
                            }
                        }
                    ]
                },
                "Tags": self._build_tags()
            }
        }
    
    def _build_condition_step(self) -> Dict[str, Any]:
        """Build condition step with quality gate."""
        quality_gate = self.ml_config["quality_gate"]
        metric_name = quality_gate["metric"]
        threshold = quality_gate["threshold"]
        direction = quality_gate["direction"]
        
        if direction == "maximize":
            operator = "GreaterThanOrEqualTo"
        else:
            operator = "LessThanOrEqualTo"
        
        register_step = self._build_register_model_step()
        fail_step = self._build_fail_step()
        
        return {
            "Name": "QualityGateCheck",
            "Type": "Condition",
            "Arguments": {
                "Conditions": [
                    {
                        "Type": "JsonGet",
                        "JsonPath": f"$.{metric_name}",
                        "PropertyFile": {
                            "PropertyFileName": "EvaluationReport",
                            "S3Uri": {
                                "Get": "Steps.EvaluateModel.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri"
                            },
                            "FilePath": "metrics.json"
                        },
                        operator: threshold
                    }
                ],
                "IfSteps": [register_step],
                "ElseSteps": [fail_step]
            },
            "DependsOn": ["EvaluateModel"]
        }
    
    def _build_register_model_step(self) -> Dict[str, Any]:
        """Build model registration step."""
        execution_role = self.config.get_execution_role(self.ml_config.get("team"))
        model_package_group_name = f"{self.ml_config['name']}-models"
        
        return {
            "Name": "RegisterModel",
            "Type": "RegisterModel",
            "Arguments": {
                "ModelPackageGroupName": model_package_group_name,
                "ModelApprovalStatus": "PendingManualApproval",
                "InferenceSpecification": {
                    "Containers": [
                        {
                            "Image": self.config.resolve_container_uri(
                                self.ml_config["framework"],
                                self.region,
                                self.account
                            ),
                            "ModelDataUrl": {
                                "Get": "Steps.TrainModel.ModelArtifacts.S3ModelArtifacts"
                            }
                        }
                    ],
                    "SupportedContentTypes": ["application/json"],
                    "SupportedResponseMIMETypes": ["application/json"],
                    "SupportedRealtimeInferenceInstanceTypes": [
                        "ml.t2.medium",
                        "ml.m5.large"
                    ]
                },
                "ModelMetrics": {
                    "ModelQuality": {
                        "Statistics": {
                            "ContentType": "application/json",
                            "S3Uri": {
                                "Get": "Steps.EvaluateModel.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri"
                            }
                        }
                    }
                },
                "CustomerMetadataProperties": {
                    "GitCommit": self._get_git_commit(),
                    "ProjectName": self.ml_config["name"],
                    "Team": self.ml_config["team"],
                    "Framework": self.ml_config["framework"]
                },
                "Tags": self._build_tags()
            }
        }
    
    def _build_fail_step(self) -> Dict[str, Any]:
        """Build fail step for quality gate failure."""
        quality_gate = self.ml_config["quality_gate"]
        
        message = (
            f"Model quality gate failed. "
            f"Metric '{quality_gate['metric']}' did not meet threshold {quality_gate['threshold']} "
            f"(direction: {quality_gate['direction']})."
        )
        
        return {
            "Name": "QualityGateFailed",
            "Type": "Fail",
            "Arguments": {
                "ErrorMessage": message
            }
        }
    
    def _build_tags(self) -> list:
        """Build tags for resources."""
        tags = [
            {"Key": "Project", "Value": self.ml_config["name"]},
            {"Key": "Team", "Value": self.ml_config["team"]},
            {"Key": "Owner", "Value": "mlctl"},
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
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return "unknown"
    
    def create_or_update_pipeline(self) -> str:
        """Create or update the pipeline in SageMaker."""
        sagemaker_client = boto3.client('sagemaker', region_name=self.region)
        
        pipeline_def = self.build_pipeline_definition()
        execution_role = self.config.get_execution_role(self.ml_config.get("team"))
        
        try:
            sagemaker_client.describe_pipeline(PipelineName=self.pipeline_name)
            
            response = sagemaker_client.update_pipeline(
                PipelineName=self.pipeline_name,
                PipelineDefinition=json.dumps(pipeline_def),
                RoleArn=execution_role
            )
            
            return response["PipelineArn"]
        
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFound':
                response = sagemaker_client.create_pipeline(
                    PipelineName=self.pipeline_name,
                    PipelineDefinition=json.dumps(pipeline_def),
                    RoleArn=execution_role,
                    Tags=self._build_tags()
                )
                
                return response["PipelineArn"]
            else:
                raise
    
    def start_pipeline_execution(self) -> str:
        """Start a pipeline execution."""
        sagemaker_client = boto3.client('sagemaker', region_name=self.region)
        
        response = sagemaker_client.start_pipeline_execution(
            PipelineName=self.pipeline_name,
            PipelineExecutionDisplayName=f"{self.ml_config['name']}-{self._get_git_commit()[:8]}"
        )
        
        return response["PipelineExecutionArn"]
