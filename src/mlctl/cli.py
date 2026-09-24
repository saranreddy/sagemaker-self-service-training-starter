"""CLI entry point for mlctl."""

import click
from rich.console import Console

from mlctl.commands import init, validate, run, submit, status, logs, list_projects

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """mlctl - Self-service SageMaker training pipelines.

    Bring your train.py and get a tested, evaluated, registered model version.
    """
    pass


main.add_command(init.init)
main.add_command(validate.validate)
main.add_command(run.run)
main.add_command(submit.submit)
main.add_command(status.status)
main.add_command(logs.logs)
main.add_command(list_projects.list_projects)


if __name__ == "__main__":
    main()
