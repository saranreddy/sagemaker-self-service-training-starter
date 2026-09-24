"""Configuration management for mlctl."""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


class Config:
    """Manages org-level and project-level configuration."""
    
    def __init__(self, org_config_path: Optional[str] = None):
        self.org_config_path = org_config_path or self._find_org_config()
        self.org_config = self._load_org_config()
    
    def _find_org_config(self) -> str:
        """Find org config in standard locations."""
        search_paths = [
            "org-config.yaml",
            os.path.expanduser("~/.mlctl/org-config.yaml"),
            "/etc/mlctl/org-config.yaml",
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        return "org-config.yaml"
    
    def _load_org_config(self) -> Dict[str, Any]:
        """Load org-level configuration."""
        if not os.path.exists(self.org_config_path):
            return self._default_org_config()
        
        with open(self.org_config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _default_org_config(self) -> Dict[str, Any]:
        """Return default org configuration for standalone mode."""
        return {
            "frameworks": {
                "sklearn": {
                    "container_uri_template": "{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
                    "default_instance_type": "ml.m5.large",
                    "version": "1.2-1"
                },
                "xgboost": {
                    "container_uri_template": "{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.7-1",
                    "default_instance_type": "ml.m5.large",
                    "version": "1.7-1"
                },
                "pytorch": {
                    "container_uri_template": "{account}.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.1.0-cpu-py310",
                    "default_instance_type": "ml.m5.large",
                    "version": "2.1.0"
                }
            },
            "default_instance_type": "ml.m5.large",
            "allowed_instance_types": [
                "ml.m5.large",
                "ml.m5.xlarge",
                "ml.m5.2xlarge",
                "ml.m5.4xlarge",
                "ml.c5.xlarge",
                "ml.c5.2xlarge",
                "ml.c5.4xlarge"
            ],
            "required_tags": {
                "Project": "sagemaker-self-service-training"
            },
            "teams": {}
        }
    
    def load_project_config(self, project_dir: str = ".") -> Dict[str, Any]:
        """Load ml.yaml from project directory."""
        ml_yaml_path = Path(project_dir) / "ml.yaml"
        
        if not ml_yaml_path.exists():
            raise FileNotFoundError(f"ml.yaml not found in {project_dir}")
        
        with open(ml_yaml_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_framework_config(self, framework: str) -> Dict[str, Any]:
        """Get framework configuration."""
        if framework not in self.org_config["frameworks"]:
            raise ValueError(f"Unsupported framework: {framework}")
        
        return self.org_config["frameworks"][framework]
    
    def get_team_config(self, team: str) -> Optional[Dict[str, Any]]:
        """Get team configuration if available."""
        return self.org_config.get("teams", {}).get(team)
    
    def is_instance_type_allowed(self, instance_type: str, team: Optional[str] = None) -> bool:
        """Check if instance type is allowed."""
        if team:
            team_config = self.get_team_config(team)
            if team_config and "allowed_instance_types" in team_config:
                return instance_type in team_config["allowed_instance_types"]
        
        return instance_type in self.org_config.get("allowed_instance_types", [])
    
    def resolve_container_uri(self, framework: str, region: str, account: str) -> str:
        """Resolve container URI for framework."""
        fw_config = self.get_framework_config(framework)
        template = fw_config["container_uri_template"]
        
        return template.format(account=account, region=region)
    
    def get_execution_role(self, team: Optional[str] = None) -> Optional[str]:
        """Get execution role ARN."""
        if team:
            team_config = self.get_team_config(team)
            if team_config and "execution_role" in team_config:
                return team_config["execution_role"]
        
        return self.org_config.get("execution_role")
    
    def get_artifact_bucket(self, team: Optional[str] = None) -> Optional[str]:
        """Get artifact S3 bucket."""
        if team:
            team_config = self.get_team_config(team)
            if team_config and "artifact_bucket" in team_config:
                return team_config["artifact_bucket"]
        
        return self.org_config.get("artifact_bucket")
