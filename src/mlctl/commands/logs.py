"""Stream training job logs."""

import sys
import time

import boto3
import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--project", required=True, help="Project name")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
def logs(project: str, follow: bool):
    """Stream CloudWatch logs for training jobs."""
    session = boto3.session.Session()
    region = session.region_name or "us-east-1"

    sagemaker_client = boto3.client("sagemaker", region_name=region)
    logs_client = boto3.client("logs", region_name=region)

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

        steps_response = sagemaker_client.list_pipeline_execution_steps(
            PipelineExecutionArn=execution_arn, MaxResults=100
        )

        training_job_name = None
        for step in steps_response.get("PipelineExecutionSteps", []):
            if step.get("StepName") == "TrainModel":
                metadata = step.get("Metadata", {})
                training_metadata = metadata.get("TrainingJob", {})
                training_job_name = training_metadata.get("Arn", "").split("/")[-1]
                break

        if not training_job_name:
            console.print("[yellow]No training job found in latest execution[/yellow]")
            sys.exit(0)  # noqa:F541

        console.print(
            f"[bold]Streaming logs for training job:[/bold] {training_job_name}\n"
        )

        log_group = "/aws/sagemaker/TrainingJobs"
        log_stream = f"{training_job_name}/algo-1-{int(time.time())}"

        try:
            response = logs_client.describe_log_streams(
                logGroupName=log_group,
                logStreamNamePrefix=training_job_name,
                orderBy="LastEventTime",
                descending=True,
                limit=1,
            )

            if response.get("logStreams"):
                log_stream = response["logStreams"][0]["logStreamName"]
        except Exception:
            pass

        next_token = None

        while True:
            try:
                kwargs = {
                    "logGroupName": log_group,  # noqa:F541
                    "logStreamName": log_stream,
                    "startFromHead": True,
                }

                if next_token:
                    kwargs["nextToken"] = next_token

                response = logs_client.get_log_events(**kwargs)

                for event in response.get("events", []):
                    console.print(event["message"].rstrip())

                next_token = response.get("nextForwardToken")

                if not follow:
                    break

                if not response.get("events"):
                    time.sleep(2)

            except logs_client.exceptions.ResourceNotFoundException:
                console.print(f"[yellow]Log stream not found: {log_stream}[/yellow]")

                if not follow:
                    break

                time.sleep(5)

            except KeyboardInterrupt:
                console.print("\n[yellow]Log streaming interrupted[/yellow]")
                break

            except Exception as e:
                console.print(f"[red]Error streaming logs: {e}[/red]")
                if not follow:
                    sys.exit(1)
                time.sleep(5)

    except Exception as e:
        console.print(f"[red]Failed to get logs: {e}[/red]")
        sys.exit(1)
