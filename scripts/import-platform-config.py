#!/usr/bin/env python3
"""Import team configurations from sagemaker-multi-team-platform-starter.

Usage:
    cd /path/to/sagemaker-multi-team-platform-starter/terraform
    terraform output -json > outputs.json
    
    cd /path/to/sagemaker-self-service-training-starter
    python scripts/import-platform-config.py outputs.json [--allowlist ml.m5.large,ml.m5.xlarge] > org-config.yaml

Note: The companion platform starter (saranreddy/sagemaker-multi-team-platform-starter @ 68d7704)
outputs team_details.value[team].{execution_role_arn, s3_bucket, ...} but does NOT export
allowed_instance_types. Pass them via --allowlist or document them separately.
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
        "platform_outputs", help="Path to terraform output -json from the platform starter"
    )
    parser.add_argument(
        "--region", default="us-east-1", help="AWS region (default: us-east-1)"
    )
    parser.add_argument(
        "--allowlist",
        help="Comma-separated list of allowed instance types (not in platform outputs)",
    )

    args = parser.parse_args()

    try:
        with open(args.platform_outputs, "r") as f:
            outputs = json.load(f)
    except Exception as e:
        print(f"Error reading platform outputs: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse allowlist if provided
    default_allowlist = [
        "ml.m5.large",
        "ml.m5.xlarge",
        "ml.m5.2xlarge",
        "ml.m5.4xlarge",
        "ml.c5.xlarge",
        "ml.c5.2xlarge",
        "ml.c5.4xlarge",
    ]

    if args.allowlist:
        default_allowlist = [t.strip() for t in args.allowlist.split(",")]

    # Build org config
    org_config = {
        "frameworks": {
            "sklearn": {"default_instance_type": "ml.m5.large", "version": "1.2-1"},
            "xgboost": {"default_instance_type": "ml.m5.large", "version": "1.7-1"},
            "pytorch": {"default_instance_type": "ml.m5.large", "version": "2.1.0"},
        },
        "default_instance_type": "ml.m5.large",
        "allowed_instance_types": default_allowlist,
        "required_tags": {"Project": "sagemaker-self-service-training"},
        "teams": {},
    }

    # Extract team details from platform outputs
    # Platform structure: team_details.value[team_name] = {...}
    team_details = outputs.get("team_details", {}).get("value", {})

    if not team_details:
        print(
            "ERROR: No teams found in platform outputs. Check outputs structure.",
            file=sys.stderr,
        )
        print(
            "Expected: team_details.value[team] = {execution_role_arn, s3_bucket, ...}",
            file=sys.stderr,
        )
        sys.exit(1)

    for team_name, team_config in team_details.items():
        org_config["teams"][team_name] = {
            "execution_role": team_config.get("execution_role_arn"),
            "artifact_bucket": team_config.get("s3_bucket"),
            "allowed_instance_types": default_allowlist,  # Not in platform, use default
        }

    print("# org-config.yaml")
    print("# Generated from sagemaker-multi-team-platform-starter outputs")
    print(f"# Teams imported: {', '.join(org_config['teams'].keys())}")
    print(
        "# Note: allowed_instance_types not exported by platform, using provided/default list"
    )
    print(
        "# Note: Platform default allowlist may not include ml.m5.large - verify/adjust as needed"
    )
    print()

    yaml.dump(org_config, sys.stdout, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
