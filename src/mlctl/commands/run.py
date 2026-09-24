"""Run training locally."""

import json
import os
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console

from mlctl.config import Config

console = Console()


@click.command()
@click.option("--project-dir", default=".", help="Project directory")
@click.option("--local", is_flag=True, default=True, help="Run in local mode (default)")
@click.option("--docker", is_flag=True, help="Use Docker-based SageMaker local mode")
@click.option("--org-config", help="Path to org-config.yaml")
def run(project_dir: str, local: bool, docker: bool, org_config: str):
    """Run training and evaluation locally."""
    if docker:
        console.print("[yellow]Docker-based local mode not yet implemented[/yellow]")
        console.print("Use --local for plain Python execution")
        sys.exit(1)

    config = Config(org_config_path=org_config)
    ml_config = config.load_project_config(project_dir)

    project_path = Path(project_dir).resolve()

    console.print("[bold]Running local training...[/bold]\n")

    model_dir = project_path / "local_output" / "model"
    output_dir = project_path / "local_output" / "output"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "SM_MODEL_DIR": str(model_dir),
            "SM_OUTPUT_DIR": str(output_dir),
            "SM_CHANNEL_TRAIN": str(project_path / "data" / "train"),
            "SM_CHANNEL_VALIDATION": str(project_path / "data" / "validation"),
            "SM_NUM_GPUS": "0",
        }
    )

    for key, value in ml_config.get("hyperparameters", {}).items():
        env[f"SM_HP_{key.upper()}"] = str(value)

    console.print("[cyan]Step 1: Training[/cyan]")
    train_script = project_path / "train.py"

    if not train_script.exists():
        console.print(f"[red]train.py not found in {project_dir}[/red]")
        sys.exit(1)

    try:
        result = subprocess.run(
            [sys.executable, str(train_script)],
            cwd=project_path,
            env=env,
            capture_output=False,
            check=True,
        )
        console.print("[green]✓ Training completed[/green]\n")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Training failed with exit code {e.returncode}[/red]")
        sys.exit(1)

    console.print("[cyan]Step 2: Evaluation[/cyan]")
    evaluate_script = project_path / "evaluate.py"

    if not evaluate_script.exists():
        console.print(f"[red]evaluate.py not found in {project_dir}[/red]")
        sys.exit(1)

    evaluation_dir = project_path / "local_output" / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    env["SM_OUTPUT_DATA_DIR"] = str(evaluation_dir)

    try:
        result = subprocess.run(
            [sys.executable, str(evaluate_script)],
            cwd=project_path,
            env=env,
            capture_output=False,
            check=True,
        )
        console.print("[green]✓ Evaluation completed[/green]\n")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Evaluation failed with exit code {e.returncode}[/red]")
        sys.exit(1)

    metrics_file = evaluation_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        console.print("[bold]Evaluation Metrics:[/bold]")
        for metric_name, value in metrics.items():
            console.print(f"  {metric_name}: {value}")

        quality_gate = ml_config["quality_gate"]
        gate_metric = quality_gate["metric"]
        threshold = quality_gate["threshold"]
        direction = quality_gate["direction"]

        if gate_metric in metrics:
            metric_value = metrics[gate_metric]

            if direction == "maximize":
                passed = metric_value >= threshold
            else:
                passed = metric_value <= threshold

            console.print(f"\n[bold]Quality Gate:[/bold] {gate_metric} {direction}")
            console.print(f"  Threshold: {threshold}")
            console.print(f"  Actual: {metric_value}")

            if passed:
                console.print("[green bold]✓ Quality gate PASSED[/green bold]")
            else:
                console.print("[red bold]✗ Quality gate FAILED[/red bold]")
                sys.exit(1)
        else:
            console.print(
                f"\n[yellow]Warning: Quality gate metric '{gate_metric}' not found in metrics.json[/yellow]"
            )
    else:
        console.print(
            "[yellow]Warning: metrics.json not found in evaluation output[/yellow]"
        )

    console.print(f"\n[green bold]✓ Local run completed successfully[/green bold]")
    console.print(f"Output saved to: {project_path / 'local_output'}")
