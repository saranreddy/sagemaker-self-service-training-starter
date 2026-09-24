"""Submit pipeline to SageMaker."""
import json
import os
import sys
from pathlib import Path

import boto3
import click
from rich.console import Console

from mlctl.config import Config
from mlctl.pipeline import PipelineBuilder
from mlctl.validation import ProjectValidator

console = Console()


@click.command()
@click.option(
    '--project-dir',
    default='.',
    help='Project directory'
)
@click.option(
    '--org-config',
    help='Path to org-config.yaml'
)
@click.option(
    '--skip-validation',
    is_flag=True,
    help='Skip validation before submit'
)
def submit(project_dir: str, org_config: str, skip_validation: bool):
    """Submit pipeline to SageMaker (create/update and start execution)."""
    config = Config(org_config_path=org_config)
    
    if not skip_validation:
        console.print("[bold]Validating project...[/bold]")
        validator = ProjectValidator(config, project_dir, offline=False)
        if not validator.validate_all():
            console.print("\n[red]Validation failed. Fix errors before submitting.[/red]")
            sys.exit(1)
        console.print()
    
    ml_config = config.load_project_config(project_dir)
    
    sts_client = boto3.client('sts')
    identity = sts_client.get_caller_identity()
    account = identity['Account']
    
    session = boto3.session.Session()
    region = session.region_name or 'us-east-1'
    
    console.print(f"[bold]Preparing pipeline submission...[/bold]")
    console.print(f"  Account: {account}")
    console.print(f"  Region: {region}")
    console.print(f"  Project: {ml_config['name']}")
    console.print(f"  Team: {ml_config['team']}")
    console.print(f"  Framework: {ml_config['framework']}")
    console.print()
    
    _upload_code(config, ml_config, project_dir, region)
    
    console.print("[bold]Creating/updating pipeline...[/bold]")
    
    builder = PipelineBuilder(config, ml_config, region, account)
    
    try:
        pipeline_arn = builder.create_or_update_pipeline()
        console.print(f"[green]✓ Pipeline ready: {pipeline_arn}[/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Pipeline creation/update failed: {e}[/red]")
        sys.exit(1)
    
    console.print("[bold]Starting pipeline execution...[/bold]")
    
    try:
        execution_arn = builder.start_pipeline_execution()
        console.print(f"[green bold]✓ Pipeline execution started[/green bold]")
        console.print(f"\n[cyan]Execution ARN:[/cyan]\n{execution_arn}")
        console.print(f"\nMonitor with: mlctl status --execution-arn {execution_arn}")
        console.print(f"Stream logs with: mlctl logs --project {ml_config['name']}")
    except Exception as e:
        console.print(f"[red]✗ Pipeline execution failed: {e}[/red]")
        sys.exit(1)


def _upload_code(config: Config, ml_config: dict, project_dir: str, region: str):
    """Upload project code to S3."""
    bucket = config.get_artifact_bucket(ml_config.get("team"))
    if not bucket:
        bucket = os.environ.get("MLCTL_ARTIFACT_BUCKET")
    
    if not bucket:
        console.print("[red]Artifact bucket not configured[/red]")
        sys.exit(1)
    
    s3_client = boto3.client('s3', region_name=region)
    project_path = Path(project_dir).resolve()
    
    code_files = ["train.py", "evaluate.py"]
    if ml_config.get("enable_preprocessing", False):
        code_files.append("preprocess.py")
    
    console.print("[bold]Uploading code to S3...[/bold]")
    
    for code_file in code_files:
        local_path = project_path / code_file
        if local_path.exists():
            s3_key = f"code/{code_file}"
            s3_client.upload_file(str(local_path), bucket, s3_key)
            console.print(f"  [green]✓[/green] Uploaded {code_file}")
    
    ml_yaml_path = project_path / "ml.yaml"
    if ml_yaml_path.exists():
        s3_key = f"projects/{ml_config['name']}/ml.yaml"
        s3_client.upload_file(str(ml_yaml_path), bucket, s3_key)
        console.print(f"  [green]✓[/green] Uploaded ml.yaml")
    
    console.print()
