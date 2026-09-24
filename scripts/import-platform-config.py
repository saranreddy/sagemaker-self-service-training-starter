#!/usr/bin/env python3
"""Import team configurations from sagemaker-multi-team-platform-starter.

Usage:
    cd /path/to/sagemaker-multi-team-platform-starter/terraform
    terraform output -json > outputs.json
    
    cd /path/to/sagemaker-self-service-training-starter
    python scripts/import-platform-config.py outputs.json > org-config.yaml
"""
import argparse
import json
import sys

import yaml


def main():
    """Import platform config and generate org-config.yaml."""
    parser = argparse.ArgumentParser(
        description="Import team configs from multi-team platform starter"
    )
    parser.add_argument(
        "platform_outputs",
        help="Path to terraform output -json from the platform starter"
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)"
    )
    
    args = parser.parse_args()
    
    try:
        with open(args.platform_outputs, 'r') as f:
            outputs = json.load(f)
    except Exception as e:
        print(f"Error reading platform outputs: {e}", file=sys.stderr)
        sys.exit(1)
    
    org_config = {
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
    
    team_configs = outputs.get("team_configs", {}).get("value", {})
    
    for team_name, team_config in team_configs.items():
        org_config["teams"][team_name] = {
            "execution_role": team_config.get("sagemaker_execution_role_arn"),
            "artifact_bucket": team_config.get("team_bucket_name"),
            "allowed_instance_types": team_config.get("allowed_instance_types", [])
        }
    
    print("# org-config.yaml")
    print("# Generated from sagemaker-multi-team-platform-starter outputs")
    print("# Teams imported:", ", ".join(org_config["teams"].keys()))
    print()
    
    yaml.dump(org_config, sys.stdout, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
