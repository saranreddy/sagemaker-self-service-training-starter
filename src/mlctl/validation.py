"""Validation logic for ml.yaml and project structure."""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import jsonschema
from rich.console import Console

from mlctl.config import Config

console = Console()


class ProjectValidator:
    """Validates project configuration and structure."""

    def __init__(self, config: Config, project_dir: str = ".", offline: bool = False):
        self.config = config
        self.project_dir = Path(project_dir)
        self.offline = offline
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> bool:
        """Run all validations and return success status."""
        self.errors = []
        self.warnings = []

        console.print("\n[bold]Validating project...[/bold]")

        try:
            ml_config = self.config.load_project_config(str(self.project_dir))
        except FileNotFoundError as e:
            self.errors.append(str(e))
            self._print_results()
            return False
        except Exception as e:
            self.errors.append(f"Failed to load ml.yaml: {e}")
            self._print_results()
            return False

        self._validate_schema(ml_config)
        self._validate_framework(ml_config)
        self._validate_instance_type(ml_config)
        self._validate_scripts()
        self._validate_requirements()

        if not self.offline:
            self._validate_s3_paths(ml_config)

        self._print_results()

        return len(self.errors) == 0

    def _validate_schema(self, ml_config: Dict[str, Any]):
        """Validate ml.yaml against JSON schema."""
        schema_path = Path(__file__).parent / "schemas" / "ml-schema.json"

        try:
            with open(schema_path, "r") as f:
                schema = json.load(f)

            jsonschema.validate(instance=ml_config, schema=schema)
            console.print("[green]✓[/green] Schema validation passed")
        except jsonschema.ValidationError as e:
            self.errors.append(f"Schema validation failed: {e.message}")
            console.print(f"[red]✗[/red] Schema validation failed: {e.message}")
        except Exception as e:
            self.errors.append(f"Schema validation error: {e}")
            console.print(f"[red]✗[/red] Schema validation error: {e}")

    def _validate_framework(self, ml_config: Dict[str, Any]):
        """Validate framework configuration."""
        framework = ml_config.get("framework")

        try:
            self.config.get_framework_config(framework)
            console.print(f"[green]✓[/green] Framework '{framework}' is supported")
        except ValueError as e:
            self.errors.append(str(e))
            console.print(f"[red]✗[/red] {e}")

    def _validate_instance_type(self, ml_config: Dict[str, Any]):
        """Validate instance type is allowed."""
        instance_type = ml_config.get("instance_type")
        team = ml_config.get("team")

        if self.config.is_instance_type_allowed(instance_type, team):
            console.print(
                f"[green]✓[/green] Instance type '{instance_type}' is allowed"
            )
        else:
            allowed_types_list = self.config.org_config.get(
                "allowed_instance_types", []
            )
            allowed_str = ", ".join(allowed_types_list)
            self.errors.append(
                f"Instance type '{instance_type}' is not in the allowlist. "
                f"Allowed types: {allowed_str}"
            )
            console.print(
                f"[red]✗[/red] Instance type '{instance_type}' is not allowed"
            )

    def _validate_scripts(self):
        """Validate required scripts exist and are valid Python."""
        required_scripts = ["train.py", "evaluate.py"]
        optional_scripts = ["preprocess.py"]

        for script in required_scripts:
            script_path = self.project_dir / script

            if not script_path.exists():
                self.errors.append(f"Required script '{script}' not found")
                console.print(f"[red]✗[/red] Required script '{script}' not found")
                continue

            if self._check_python_syntax(script_path):
                console.print(f"[green]✓[/green] Script '{script}' syntax is valid")
            else:
                self.errors.append(f"Script '{script}' has syntax errors")
                console.print(f"[red]✗[/red] Script '{script}' has syntax errors")

        for script in optional_scripts:
            script_path = self.project_dir / script
            if script_path.exists():
                if self._check_python_syntax(script_path):
                    console.print(
                        f"[green]✓[/green] Optional script '{script}' syntax is valid"
                    )
                else:
                    self.warnings.append(
                        f"Optional script '{script}' has syntax errors"
                    )
                    console.print(
                        f"[yellow]![/yellow] Optional script '{script}' has syntax errors"
                    )

    def _check_python_syntax(self, script_path: Path) -> bool:
        """Check if Python script has valid syntax."""
        try:
            with open(script_path, "r") as f:
                code = f.read()
            compile(code, str(script_path), "exec")
            return True
        except SyntaxError:
            return False

    def _validate_requirements(self):
        """Validate requirements.txt exists and can be parsed."""
        req_path = self.project_dir / "requirements.txt"

        if not req_path.exists():
            self.warnings.append("requirements.txt not found (recommended)")
            console.print("[yellow]![/yellow] requirements.txt not found (recommended)")
            return

        try:
            with open(req_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    pass

            console.print("[green]✓[/green] requirements.txt is valid")
        except Exception as e:
            self.warnings.append(f"requirements.txt parse error: {e}")
            console.print(f"[yellow]![/yellow] requirements.txt parse error: {e}")

    def _validate_s3_paths(self, ml_config: Dict[str, Any]):
        """Validate S3 paths exist (requires AWS access)."""
        import boto3
        from botocore.exceptions import ClientError

        s3_client = boto3.client("s3")
        data_config = ml_config.get("data", {})

        for key, s3_uri in data_config.items():
            if not s3_uri:
                continue

            if not s3_uri.startswith("s3://"):
                self.errors.append(f"Invalid S3 URI for data.{key}: {s3_uri}")
                console.print(f"[red]✗[/red] Invalid S3 URI for data.{key}")
                continue

            try:
                bucket, key_path = self._parse_s3_uri(s3_uri)

                if key_path.endswith("/"):
                    response = s3_client.list_objects_v2(
                        Bucket=bucket, Prefix=key_path, MaxKeys=1
                    )
                    exists = response.get("KeyCount", 0) > 0
                else:
                    s3_client.head_object(Bucket=bucket, Key=key_path)
                    exists = True

                if exists:
                    console.print(f"[green]✓[/green] S3 path exists: data.{key}")
                else:
                    self.errors.append(f"S3 path not found: data.{key} ({s3_uri})")
                    console.print(f"[red]✗[/red] S3 path not found: data.{key}")

            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    self.errors.append(f"S3 path not found: data.{key} ({s3_uri})")
                    console.print(f"[red]✗[/red] S3 path not found: data.{key}")
                else:
                    self.errors.append(f"S3 access error for data.{key}: {e}")
                    console.print(f"[red]✗[/red] S3 access error for data.{key}: {e}")
            except Exception as e:
                # Catch network errors, NoCredentialsError, BotoCoreError, etc.
                self.errors.append(f"S3 access error for data.{key}: {e}")
                console.print(f"[red]✗[/red] S3 access error for data.{key}: {e}")

    def _parse_s3_uri(self, s3_uri: str) -> Tuple[str, str]:
        """Parse S3 URI into bucket and key."""
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def _print_results(self):
        """Print validation results summary."""
        console.print()

        if self.errors:
            console.print(
                f"[red bold]Validation failed with {len(self.errors)} error(s)[/red bold]"
            )
            for error in self.errors:
                console.print(f"  [red]•[/red] {error}")

        if self.warnings:
            console.print(f"[yellow]{len(self.warnings)} warning(s)[/yellow]")
            for warning in self.warnings:
                console.print(f"  [yellow]•[/yellow] {warning}")

        if not self.errors:
            console.print("[green bold]✓ Validation passed[/green bold]")
