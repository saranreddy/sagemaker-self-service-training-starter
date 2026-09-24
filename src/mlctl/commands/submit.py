"""Submit pipeline to SageMaker."""

import json
import os
import sys
from pathlib import Path

import boto3
import click
from rich.console import Console

from mlctl.config import Config
from mlctl.packaging import upload_code_packages
from mlctl.pipeline import PipelineBuilder
from mlctl.validation import ProjectValidator

console = Console()


@click.command()
@click.option("--project-dir", default=".", help="Project directory")
@click.option("--org-config", help="Path to org-config.yaml")
@click.option("--skip-validation", is_flag=True, help="Skip validation before submit")
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def submit(project_dir: str, org_config: str, skip_validation: bool, output: str):
    """Submit pipeline to SageMaker (create/update and start execution)."""
    config = Config(org_config_path=org_config)

    if not skip_validation:
        if output == "text":
            console.print("[bold]Validating project...[/bold]")
        else:
            # Use stderr console for validation output in JSON mode
            from rich.console import Console as RichConsole

            validation_console = RichConsole(stderr=True)
            # Temporarily replace global console
            import mlctl.validation as validation_module

            original_console = validation_module.console
            validation_module.console = validation_console

        validator = ProjectValidator(config, project_dir, offline=False)
        validation_passed = validator.validate_all()

        if output == "json" and not skip_validation:
            # Restore original console
            validation_module.console = original_console

        if not validation_passed:
            if output == "text":
                console.print(
                    "\n[red]Validation failed. Fix errors before submitting.[/red]"
                )
            else:
                print(json.dumps({"error": "validation failed"}), file=sys.stderr)
            sys.exit(1)
        if output == "text":
            console.print()

    ml_config = config.load_project_config(project_dir)

    try:
        sts_client = boto3.client("sts")
        identity = sts_client.get_caller_identity()
        account = identity["Account"]
    except Exception as e:
        if output == "json":
            print(
                json.dumps({"error": f"AWS credentials error: {str(e)}"}),
                file=sys.stderr,
            )
        else:
            console.print(f"[red]AWS credentials error: {e}[/red]")
        sys.exit(1)

    session = boto3.session.Session()
    region = session.region_name or "us-east-1"

    if output == "text":
        console.print("[bold]Preparing pipeline submission...[/bold]")
        console.print(f"  Account: {account}")
        console.print(f"  Region: {region}")
        console.print(f"  Project: {ml_config['name']}")
        console.print(f"  Team: {ml_config['team']}")
        console.print(f"  Framework: {ml_config['framework']}")
        console.print()

    # Package and upload code
    bucket = config.get_artifact_bucket(ml_config.get("team"))
    if not bucket:
        bucket = os.environ.get("MLCTL_ARTIFACT_BUCKET")

    if not bucket:
        if output == "text":
            console.print("[red]Artifact bucket not configured[/red]")
        else:
            print(
                json.dumps({"error": "Artifact bucket not configured"}), file=sys.stderr
            )
        sys.exit(1)

    builder = PipelineBuilder(config, ml_config, region, Path(project_dir))

    if output == "text":
        console.print("[bold]Packaging and uploading code...[/bold]")

    try:
        # Package code first to get file paths for content hashing
        from mlctl.packaging import package_training_code, package_evaluation_code

        sourcedir_path = package_training_code(Path(project_dir), ml_config["name"])
        evaluation_path = package_evaluation_code(Path(project_dir), ml_config["name"])
        ml_yaml_path = Path(project_dir) / "ml.yaml"

        # Generate S3 prefix based on actual content
        content_paths = [str(sourcedir_path), str(evaluation_path), str(ml_yaml_path)]
        s3_prefix = builder.get_code_s3_prefix_for_content(content_paths)

        # Set s3_prefix on builder for pipeline definition
        builder.code_s3_prefix = s3_prefix

        # Now upload with the content-based prefix
        code_uris = upload_code_packages(
            Path(project_dir), ml_config["name"], bucket, s3_prefix, region
        )
        # Set ml.yaml URI on builder for CustomerMetadataProperties
        builder.ml_yaml_uri = code_uris["ml_yaml"]

        if output == "text":
            console.print("  [green]✓[/green] Uploaded sourcedir.tar.gz")
            console.print("  [green]✓[/green] Uploaded evaluation.tar.gz")
            console.print("  [green]✓[/green] Uploaded ml.yaml")
            console.print()
    except Exception as e:
        if output == "text":
            console.print(f"[red]✗ Code upload failed: {e}[/red]")
        else:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

    # Create SageMaker client
    sagemaker_client = boto3.client("sagemaker", region_name=region)

    # Ensure model package group exists
    if output == "text":
        console.print("[bold]Ensuring model package group exists...[/bold]")

    try:
        builder.ensure_model_package_group(sagemaker_client)
        if output == "text":
            console.print(
                f"  [green]✓[/green] Model package group ready: {ml_config['name']}-models"
            )
            console.print()
    except Exception as e:
        if output == "text":
            console.print(f"[red]✗ Model package group creation failed: {e}[/red]")
        else:
            print(
                json.dumps({"error": f"Model package group creation failed: {e}"}),
                file=sys.stderr,
            )
        sys.exit(1)

    # Create or update pipeline
    if output == "text":
        console.print("[bold]Creating/updating pipeline...[/bold]")

    try:
        pipeline_arn = builder.create_or_update_pipeline(sagemaker_client)
        if output == "text":
            console.print(f"[green]✓ Pipeline ready: {pipeline_arn}[/green]\n")
    except Exception as e:
        if output == "text":
            console.print(f"[red]✗ Pipeline creation/update failed: {e}[/red]")
        else:
            print(
                json.dumps({"error": f"Pipeline creation/update failed: {e}"}),
                file=sys.stderr,
            )
        sys.exit(1)

    # Start pipeline execution
    if output == "text":
        console.print("[bold]Starting pipeline execution...[/bold]")

    try:
        execution_arn = builder.start_pipeline_execution(sagemaker_client)

        if output == "json":
            print(
                json.dumps(
                    {
                        "execution_arn": execution_arn,
                        "pipeline_arn": pipeline_arn,
                        "project": ml_config["name"],
                        "region": region,
                    }
                )
            )
        else:
            console.print(
                "[green bold]✓ Pipeline execution started[/green bold]"
            )
            console.print(f"\n[cyan]Execution ARN:[/cyan]\n{execution_arn}")
            console.print(
                f"\nMonitor with: mlctl status --execution-arn {execution_arn}"
            )
            console.print(f"Stream logs with: mlctl logs --project {ml_config['name']}")

    except Exception as e:
        if output == "text":
            console.print(f"[red]✗ Pipeline execution failed: {e}[/red]")
        else:
            print(
                json.dumps({"error": f"Pipeline execution failed: {e}"}),
                file=sys.stderr,
            )
        sys.exit(1)
