"""Validate project configuration."""

import sys

import click

from mlctl.config import Config
from mlctl.validation import ProjectValidator


@click.command()
@click.option("--project-dir", default=".", help="Project directory to validate")
@click.option(
    "--offline", is_flag=True, help="Skip S3 path validation (for offline use)"
)
@click.option("--org-config", help="Path to org-config.yaml")
def validate(project_dir: str, offline: bool, org_config: str):
    """Validate ml.yaml configuration and project structure."""
    config = Config(org_config_path=org_config)
    validator = ProjectValidator(config, project_dir, offline)

    success = validator.validate_all()

    sys.exit(0 if success else 1)
