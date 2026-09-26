"""Unit tests for AWS tag validation."""

import re
from pathlib import Path

import pytest

from mlctl.pipeline import PipelineBuilder


class TestTagKeyValidation:
    """Test tag key validation."""

    @pytest.fixture
    def pipeline_builder(self):
        """Create a minimal pipeline builder for testing."""
        config = type(
            "Config",
            (),
            {
                "get_artifact_bucket": lambda self, team: "test-bucket",
                "get_deployment_tag": lambda self: {"mlctl:deployment": "test"},
                "get_required_tags": lambda self: {},
            },
        )()

        ml_config = {
            "name": "test-project",
            "team": "test-team",
            "owner": "test-owner",
            "framework": "sklearn",
        }

        return PipelineBuilder(
            config=config,
            ml_config=ml_config,
            region="us-east-1",
        )

    def test_valid_tag_keys(self, pipeline_builder):
        """Test that valid tag keys pass validation."""
        valid_keys = [
            "Project",
            "Team",
            "Owner",
            "mlctl:deployment",
            "Cost-Center",
            "Environment_Name",
            "Project/Team",
            "Email@Domain",
            "A+B=C.D:E/F@G",
            "a" * 128,
        ]
        for key in valid_keys:
            pipeline_builder._validate_tag_key(key)

    def test_empty_key_raises(self, pipeline_builder):
        """Test that empty key raises ValueError."""
        with pytest.raises(ValueError, match="Tag key cannot be empty"):
            pipeline_builder._validate_tag_key("")

    def test_too_long_key_raises(self, pipeline_builder):
        """Test that key over 128 chars raises ValueError."""
        long_key = "a" * 129
        with pytest.raises(ValueError, match="Tag key exceeds 128 chars"):
            pipeline_builder._validate_tag_key(long_key)

    def test_aws_prefix_case_insensitive_raises(self, pipeline_builder):
        """Test that keys starting with aws: in any case raise ValueError."""
        invalid_keys = [
            "aws:something",
            "AWS:Something",
            "Aws:Something",
            "AwS:test",
        ]
        for key in invalid_keys:
            with pytest.raises(ValueError, match="cannot start with 'aws:'"):
                pipeline_builder._validate_tag_key(key)

    def test_invalid_characters_raise(self, pipeline_builder):
        """Test that keys with invalid characters raise ValueError."""
        invalid_keys = [
            "Project\n",
            "Team\t",
            "Owner\r",
            "Key;Value",
            "Key<Value",
            "Key>Value",
            "Key{Value}",
            "Key[Value]",
            "Key*Value",
            "Key?Value",
            "Key&Value",
        ]
        for key in invalid_keys:
            with pytest.raises(ValueError, match="contains invalid characters"):
                pipeline_builder._validate_tag_key(key)


class TestAWSTagValidation:
    """Test that all tags comply with AWS tag requirements."""

    # AWS tag regex: letters, numbers, spaces, and + - = . _ : / @
    TAG_VALUE_REGEX = re.compile(r"^[a-zA-Z0-9\s+\-=._:/@]*$")
    TAG_KEY_MAX_LENGTH = 128
    TAG_VALUE_MAX_LENGTH = 256

    def test_tag_sanitization(self):
        """Test that _sanitize_tag_value removes disallowed characters."""
        # Test disallowed characters are replaced
        assert PipelineBuilder._sanitize_tag_value("test;value") == "test-value"
        assert PipelineBuilder._sanitize_tag_value("test!value") == "test-value"
        assert PipelineBuilder._sanitize_tag_value("test,value") == "test-value"
        assert PipelineBuilder._sanitize_tag_value("test(value)") == "test-value-"

        # Test allowed characters pass through
        assert PipelineBuilder._sanitize_tag_value("test-value") == "test-value"
        assert PipelineBuilder._sanitize_tag_value("test_value") == "test_value"
        assert PipelineBuilder._sanitize_tag_value("test+value") == "test+value"
        assert PipelineBuilder._sanitize_tag_value("test=value") == "test=value"
        assert PipelineBuilder._sanitize_tag_value("test.value") == "test.value"
        assert (
            PipelineBuilder._sanitize_tag_value("test:value") == "test:value"
        )  # Colon is allowed
        assert PipelineBuilder._sanitize_tag_value("test/value") == "test/value"
        assert PipelineBuilder._sanitize_tag_value("test@value") == "test@value"
        assert PipelineBuilder._sanitize_tag_value("Test Value 123") == "Test Value 123"

        # Test length truncation
        long_value = "a" * 300
        sanitized = PipelineBuilder._sanitize_tag_value(long_value)
        assert len(sanitized) == 256

        # Test empty string
        assert PipelineBuilder._sanitize_tag_value("") == ""

        # Test tab and newline are replaced with hyphen
        assert PipelineBuilder._sanitize_tag_value("test\tvalue") == "test-value"
        assert PipelineBuilder._sanitize_tag_value("test\nvalue") == "test-value"
        assert PipelineBuilder._sanitize_tag_value("test\r\nvalue") == "test--value"

    def test_terraform_tag_literals(self):
        """Test that all terraform tag literals match AWS requirements."""
        terraform_dir = Path(__file__).parent.parent / "terraform"

        # Parse all .tf files for tag literals
        tag_violations = []

        for tf_file in terraform_dir.glob("*.tf"):
            content = tf_file.read_text()

            # Look specifically for tags = { ... } blocks
            in_tags_block = False
            brace_depth = 0

            for line_num, line in enumerate(content.split("\n"), 1):
                stripped = line.strip()

                # Detect tags block start
                if "tags" in stripped and "=" in stripped and "{" in stripped:
                    in_tags_block = True
                    brace_depth = 1
                    continue

                if in_tags_block:
                    # Track brace depth
                    brace_depth += stripped.count("{") - stripped.count("}")

                    if brace_depth == 0:
                        in_tags_block = False
                        continue

                    # Extract tag values from lines like: Name = "value"
                    if "=" in stripped:
                        matches = re.findall(r'=\s*"([^"]+)"', stripped)
                        for value in matches:
                            # Skip variable references
                            if "${" in value:
                                continue

                            # Check if value matches AWS tag regex
                            if not self.TAG_VALUE_REGEX.match(value):
                                tag_violations.append(
                                    {
                                        "file": tf_file.name,
                                        "line": line_num,
                                        "value": value,
                                        "line_content": stripped,
                                    }
                                )

                            if len(value) > self.TAG_VALUE_MAX_LENGTH:
                                tag_violations.append(
                                    {
                                        "file": tf_file.name,
                                        "line": line_num,
                                        "value": value,
                                        "reason": f"exceeds {self.TAG_VALUE_MAX_LENGTH} chars",
                                    }
                                )

        if tag_violations:
            msg = "Found AWS tag violations in Terraform files:\n"
            for v in tag_violations:
                reason = v.get("reason", "invalid characters")
                content = v.get("line_content", v["value"])
                msg += f"  {v['file']}:{v['line']} - {reason}: {content}\n"
            pytest.fail(msg)

    def test_pipeline_build_tags_output(self):
        """Test that _build_tags produces valid AWS tags."""
        import tempfile
        from mlctl.config import Config

        # Create a temporary org-config.yaml file
        config_content = """
execution_role: arn:aws:iam::123456789012:role/test-role
artifact_bucket: test-bucket
frameworks:
  sklearn:
    version: '1.2-1'
    default_instance_type: ml.m5.large
default_instance_type: ml.m5.large
deployment_tag:
  mlctl:deployment: test-deployment
required_tags:
  Project: test-project
  Environment: test
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            ml_config = {
                "name": "test-project",
                "team": "test-team",
                "owner": "test-owner",
                "framework": "sklearn",
                "instance_type": "ml.m5.large",
                "data": {
                    "train": "s3://bucket/train/",
                    "validation": "s3://bucket/val/",
                },
                "hyperparameters": {},
                "quality_gate": {
                    "metric": "accuracy",
                    "threshold": 0.8,
                    "direction": "maximize",
                },
            }

            config = Config(org_config_path=config_path)
            builder = PipelineBuilder(config, ml_config, "us-east-1", ".")

            tags = builder._build_tags()

            # Validate all tags
            for tag in tags:
                key = tag["Key"]
                value = tag["Value"]

                # Check key length
                assert (
                    len(key) <= self.TAG_KEY_MAX_LENGTH
                ), f"Tag key '{key}' exceeds {self.TAG_KEY_MAX_LENGTH} chars"

                # Check value length
                assert (
                    len(value) <= self.TAG_VALUE_MAX_LENGTH
                ), f"Tag value for key '{key}' exceeds {self.TAG_VALUE_MAX_LENGTH} chars"

                # Check value matches AWS regex
                assert self.TAG_VALUE_REGEX.match(
                    value
                ), f"Tag value for key '{key}' contains invalid characters: '{value}'"
        finally:
            import os

            os.unlink(config_path)

    def test_pipeline_build_tags_with_special_chars(self):
        """Test that _build_tags sanitizes special characters from user input."""
        import tempfile
        from mlctl.config import Config

        config_content = """
execution_role: arn:aws:iam::123456789012:role/test-role
artifact_bucket: test-bucket
frameworks:
  sklearn:
    version: '1.2-1'
    default_instance_type: ml.m5.large
default_instance_type: ml.m5.large
deployment_tag:
  mlctl:deployment: 'test;deployment'
required_tags: {}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            ml_config = {
                "name": "test;project!",  # Contains disallowed chars
                "team": "test,team",  # Contains comma
                "owner": "test(owner)",  # Contains parentheses
                "framework": "sklearn",
                "instance_type": "ml.m5.large",
                "data": {
                    "train": "s3://bucket/train/",
                    "validation": "s3://bucket/val/",
                },
                "hyperparameters": {},
                "quality_gate": {
                    "metric": "accuracy",
                    "threshold": 0.8,
                    "direction": "maximize",
                },
            }

            config = Config(org_config_path=config_path)
            builder = PipelineBuilder(config, ml_config, "us-east-1", ".")

            tags = builder._build_tags()

            # All tag values should be sanitized
            for tag in tags:
                value = tag["Value"]
                assert self.TAG_VALUE_REGEX.match(
                    value
                ), f"Tag value contains invalid characters: '{value}'"

                # Check specific sanitizations
                if tag["Key"] == "Project":
                    assert value == "test-project-"
                elif tag["Key"] == "Team":
                    assert value == "test-team"
                elif tag["Key"] == "Owner":
                    assert value == "test-owner-"
        finally:
            import os

            os.unlink(config_path)

    def test_customer_metadata_sanitization(self):
        """Test that customer metadata in model registration is sanitized."""
        import tempfile
        from mlctl.config import Config

        config_content = """
execution_role: arn:aws:iam::123456789012:role/test-role
artifact_bucket: test-bucket
frameworks:
  sklearn:
    version: '1.2-1'
    default_instance_type: ml.m5.large
default_instance_type: ml.m5.large
required_tags: {}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            ml_config = {
                "name": "test;project",
                "team": "test,team",
                "owner": "test!owner",
                "framework": "sklearn",
                "instance_type": "ml.m5.large",
                "data": {
                    "train": "s3://bucket/train/",
                    "validation": "s3://bucket/val/",
                },
                "hyperparameters": {},
                "quality_gate": {
                    "metric": "accuracy",
                    "threshold": 0.8,
                    "direction": "maximize",
                },
            }

            config = Config(org_config_path=config_path)
            builder = PipelineBuilder(config, ml_config, "us-east-1", ".")

            # Build the pipeline definition to get the RegisterModel step
            pipeline_def = builder.build_pipeline_definition()

            # Find the RegisterModel step within the condition
            condition_step = next(
                s for s in pipeline_def["Steps"] if s["Name"] == "QualityGateCheck"
            )
            register_step = condition_step["Arguments"]["IfSteps"][0]

            customer_metadata = register_step["Arguments"]["CustomerMetadataProperties"]

            # All metadata values should match AWS tag regex
            for key, value in customer_metadata.items():
                assert self.TAG_VALUE_REGEX.match(
                    value
                ), f"CustomerMetadataProperties['{key}'] contains invalid characters: '{value}'"
        finally:
            import os

            os.unlink(config_path)
