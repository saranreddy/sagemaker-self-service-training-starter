"""Initialize a new ML project."""

import shutil
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.option(
    "--framework",
    type=click.Choice(["sklearn", "xgboost", "pytorch"]),
    required=True,
    help="ML framework to use",
)
@click.option("--output", default=".", help="Output directory for project")
def init(framework: str, output: str):
    """Initialize a new ML project with working example code."""
    output_dir = Path(output)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    templates_dir = Path(__file__).parent.parent / "templates" / framework

    if not templates_dir.exists():
        console.print(f"[red]Template for {framework} not found[/red]")
        return

    files_to_copy = [
        "ml.yaml",
        "train.py",
        "evaluate.py",
        "requirements.txt",
        "README.md",
    ]

    for file_name in files_to_copy:
        src = templates_dir / file_name
        dst = output_dir / file_name

        if dst.exists():
            console.print(f"[yellow]Skipping {file_name} (already exists)[/yellow]")
            continue

        if src.exists():
            shutil.copy(src, dst)
            console.print(f"[green]Created {file_name}[/green]")

    console.print(
        f"\n[green bold]✓ Project initialized with {framework} template[/green bold]"
    )
    console.print("\nNext steps:")
    console.print("  1. Review and customize ml.yaml")
    console.print("  2. Run: mlctl validate")  # noqa:F541
    console.print("  3. Test locally: mlctl run --local")
    console.print("  4. Submit to SageMaker: mlctl submit")
