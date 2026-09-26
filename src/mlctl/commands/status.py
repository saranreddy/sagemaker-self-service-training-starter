"""Check pipeline execution status."""

import sys

import boto3
import click
from rich.console import Console
from rich.table import Table
from rich.markup import escape

console = Console()


@click.command()
@click.option("--execution-arn", help="Pipeline execution ARN")
@click.option("--project", help="Project name (shows latest execution)")
def status(execution_arn: str, project: str):
    """Check pipeline execution status."""
    if not execution_arn and not project:
        console.print("[red]Provide either --execution-arn or --project[/red]")
        sys.exit(1)

    session = boto3.session.Session()
    region = session.region_name or "us-east-1"
    sagemaker_client = boto3.client("sagemaker", region_name=region)

    if project:
        pipeline_name = f"{project}-pipeline"

        try:
            response = sagemaker_client.list_pipeline_executions(
                PipelineName=pipeline_name,
                MaxResults=1,
                SortBy="CreationTime",
                SortOrder="Descending",
            )

            if not response.get("PipelineExecutionSummaries"):
                console.print(
                    f"[yellow]No executions found for project '{project}'[/yellow]"
                )
                sys.exit(0)

            execution_arn = response["PipelineExecutionSummaries"][0][
                "PipelineExecutionArn"
            ]

        except Exception as e:
            console.print(
                f"[red]Failed to find pipeline for project '{project}': {e}[/red]"
            )
            sys.exit(1)

    try:
        response = sagemaker_client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )

        console.print("[bold]Pipeline Execution Status[/bold]\n")
        console.print(f"[cyan]ARN:[/cyan] {execution_arn}")
        console.print(f"[cyan]Status:[/cyan] {response['PipelineExecutionStatus']}")
        console.print(
            f"[cyan]Display Name:[/cyan] {response.get('PipelineExecutionDisplayName', 'N/A')}"
        )

        if "CreationTime" in response:
            console.print(f"[cyan]Created:[/cyan] {response['CreationTime']}")

        if "LastModifiedTime" in response:
            console.print(f"[cyan]Last Modified:[/cyan] {response['LastModifiedTime']}")

        if response.get("FailureReason"):
            console.print(
                f"\n[red bold]Failure Reason:[/red bold]\n{escape(response['FailureReason'])}"
            )

        steps_response = sagemaker_client.list_pipeline_execution_steps(
            PipelineExecutionArn=execution_arn, MaxResults=100
        )

        if steps_response.get("PipelineExecutionSteps"):
            console.print("\n[bold]Steps:[/bold]")

            table = Table()
            table.add_column("Step Name")
            table.add_column("Type")
            table.add_column("Status")
            table.add_column("Started")

            # Collect failed step reasons to print after the table
            failed_step_reasons = []

            for step in steps_response["PipelineExecutionSteps"]:
                step_name = step["StepName"]
                step_type = step.get("StepDisplayName", step.get("StepName", "Unknown"))
                status = step["StepStatus"]
                started = step.get("StartTime", "N/A")

                table.add_row(step_name, step_type, status, str(started))

                # Collect FailureReason for failed steps
                if status == "Failed" and step.get("FailureReason"):
                    failed_step_reasons.append((step_name, step["FailureReason"]))

            console.print(table)

            # Print failed step reasons after the table
            if failed_step_reasons:
                console.print()
                for step_name, reason in failed_step_reasons:
                    console.print(f"[red bold]{step_name} failure:[/red bold]")
                    console.print(f"  {escape(reason)}")

    except Exception as e:
        console.print(f"[red]Failed to get execution status: {e}[/red]")
        sys.exit(1)
