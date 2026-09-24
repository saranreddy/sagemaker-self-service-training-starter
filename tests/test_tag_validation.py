"""Unit tests for AWS tag validation."""

import re
from pathlib import Path

import pytest

from mlctl.pipeline import PipelineBuilder


class TestAWSTagValidation:
    """Test that all tags comply with AWS tag requirements."""

    # AWS tag regex: letters, numbers, spaces, and + - = . _ : / @
    TAG_VALUE_REGEX = re.compile(r'^[a-zA-Z0-9\s+\-=._:/@]*$')
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
        assert PipelineBuilder._sanitize_tag_value("test:value") == "test-value"
        assert PipelineBuilder._sanitize_tag_value("test/value") == "test/value"
        assert PipelineBuilder._sanitize_tag_value("test@value") == "test@value"
        assert PipelineBuilder._sanitize_tag_value("Test Value 123") == "Test Value 123"
        
        # Test length truncation
        long_value = "a" * 300
        sanitized = PipelineBuilder._sanitize_tag_value(long_value)
        assert len(sanitized) == 256
        
        # Test empty string
        assert PipelineBuilder._sanitize_tag_value("") == ""

    def test_terraform_tag_literals(self):
        """Test that all terraform tag literals match AWS requirements."""
        terraform_dir = Path(__file__).parent.parent / "terraform"
        
        # Parse all .tf files for tag literals
        tag_violations = []
        
        for tf_file in terraform_dir.glob("*.tf"):
            content = tf_file.read_text()
            
            # Find tag blocks (simplified pattern)
            # This catches: tags = { ... } and default_tags { tags = { ... } }
            # Looking for lines like: Key = "value"
            for line_num, line in enumerate(content.split('\n'), 1):
                # Match tag assignment patterns
                if '=' in line and ('"' in line or "'" in line):
                    # Extract quoted values
                    matches = re.findall(r'["\']([^"\']+)["\']', line)
                    for value in matches:
                        # Skip variable references and empty strings
                        if value.startswith('${') or not value:
                            continue
                        
                        # Check if this looks like a tag value (not a key or other code)
                        if not self.TAG_VALUE_REGEX.match(value):
                            tag_violations.append({
                                'file': tf_file.name,
                                'line': line_num,
                                'value': value,
                                'line_content': line.strip()
                            })
                        
                        if len(value) > self.TAG_VALUE_MAX_LENGTH:
                            tag_violations.append({
                                'file': tf_file.name,
                                'line': line_num,
                                'value': value,
                                'reason': f'exceeds {self.TAG_VALUE_MAX_LENGTH} chars'
                            })
        
        if tag_violations:
            msg = "Found AWS tag violations in Terraform files:\n"
            for v in tag_violations:
                msg += f"  {v['file']}:{v['line']} - {v.get('reason', 'invalid characters')}: {v.get('line_content', v['value'])}\n"
            pytest.fail(msg)

    def test_pipeline_build_tags_output(self):
        """Test that _build_tags produces valid AWS tags."""
        from mlctl.config import Config
        
        # Create a test config
        config_dict = {
            'execution_role': 'arn:aws:iam::123456789012:role/test-role',
            'artifact_bucket': 'test-bucket',
            'frameworks': {
                'sklearn': {
                    'version': '1.2-1',
                    'default_instance_type': 'ml.m5.large'
                }
            },
            'default_instance_type': 'ml.m5.large',
            'deployment_tag': {
                'mlctl:deployment': 'test-deployment'
            },
            'required_tags': {
                'Project': 'test-project',
                'Environment': 'test'
            }
        }
        
        ml_config = {
            'name': 'test-project',
            'team': 'test-team',
            'owner': 'test-owner',
            'framework': 'sklearn',
            'instance_type': 'ml.m5.large',
            'data': {
                'train': 's3://bucket/train/',
                'validation': 's3://bucket/val/'
            },
            'hyperparameters': {},
            'quality_gate': {
                'metric': 'accuracy',
                'threshold': 0.8,
                'direction': 'maximize'
            }
        }
        
        config = Config(config_dict, org_config=config_dict)
        builder = PipelineBuilder(config, ml_config, 'us-east-1', '.')
        
        tags = builder._build_tags()
        
        # Validate all tags
        for tag in tags:
            key = tag['Key']
            value = tag['Value']
            
            # Check key length
            assert len(key) <= self.TAG_KEY_MAX_LENGTH, f"Tag key '{key}' exceeds {self.TAG_KEY_MAX_LENGTH} chars"
            
            # Check value length
            assert len(value) <= self.TAG_VALUE_MAX_LENGTH, f"Tag value for key '{key}' exceeds {self.TAG_VALUE_MAX_LENGTH} chars"
            
            # Check value matches AWS regex
            assert self.TAG_VALUE_REGEX.match(value), f"Tag value for key '{key}' contains invalid characters: '{value}'"

    def test_pipeline_build_tags_with_special_chars(self):
        """Test that _build_tags sanitizes special characters from user input."""
        from mlctl.config import Config
        
        config_dict = {
            'execution_role': 'arn:aws:iam::123456789012:role/test-role',
            'artifact_bucket': 'test-bucket',
            'frameworks': {
                'sklearn': {
                    'version': '1.2-1',
                    'default_instance_type': 'ml.m5.large'
                }
            },
            'default_instance_type': 'ml.m5.large',
            'deployment_tag': {
                'mlctl:deployment': 'test;deployment'  # Contains semicolon
            },
            'required_tags': {}
        }
        
        ml_config = {
            'name': 'test;project!',  # Contains disallowed chars
            'team': 'test,team',  # Contains comma
            'owner': 'test(owner)',  # Contains parentheses
            'framework': 'sklearn',
            'instance_type': 'ml.m5.large',
            'data': {
                'train': 's3://bucket/train/',
                'validation': 's3://bucket/val/'
            },
            'hyperparameters': {},
            'quality_gate': {
                'metric': 'accuracy',
                'threshold': 0.8,
                'direction': 'maximize'
            }
        }
        
        config = Config(config_dict, org_config=config_dict)
        builder = PipelineBuilder(config, ml_config, 'us-east-1', '.')
        
        tags = builder._build_tags()
        
        # All tag values should be sanitized
        for tag in tags:
            value = tag['Value']
            assert self.TAG_VALUE_REGEX.match(value), f"Tag value contains invalid characters: '{value}'"
            
            # Check specific sanitizations
            if tag['Key'] == 'Project':
                assert 'test-project-' == value
            elif tag['Key'] == 'Team':
                assert 'test-team' == value
            elif tag['Key'] == 'Owner':
                assert 'test-owner-' == value

    def test_customer_metadata_sanitization(self):
        """Test that customer metadata in model registration is sanitized."""
        from mlctl.config import Config
        
        config_dict = {
            'execution_role': 'arn:aws:iam::123456789012:role/test-role',
            'artifact_bucket': 'test-bucket',
            'frameworks': {
                'sklearn': {
                    'version': '1.2-1',
                    'default_instance_type': 'ml.m5.large'
                }
            },
            'default_instance_type': 'ml.m5.large',
            'required_tags': {}
        }
        
        ml_config = {
            'name': 'test;project',
            'team': 'test,team',
            'owner': 'test!owner',
            'framework': 'sklearn',
            'instance_type': 'ml.m5.large',
            'data': {
                'train': 's3://bucket/train/',
                'validation': 's3://bucket/val/'
            },
            'hyperparameters': {},
            'quality_gate': {
                'metric': 'accuracy',
                'threshold': 0.8,
                'direction': 'maximize'
            }
        }
        
        config = Config(config_dict, org_config=config_dict)
        builder = PipelineBuilder(config, ml_config, 'us-east-1', '.')
        
        # Build the pipeline definition to get the RegisterModel step
        pipeline_def = builder.build_pipeline_definition()
        
        # Find the RegisterModel step within the condition
        condition_step = next(s for s in pipeline_def['Steps'] if s['Name'] == 'QualityGateCheck')
        register_step = condition_step['Arguments']['IfSteps'][0]
        
        customer_metadata = register_step['Arguments']['CustomerMetadataProperties']
        
        # All metadata values should match AWS tag regex
        for key, value in customer_metadata.items():
            assert self.TAG_VALUE_REGEX.match(value), f"CustomerMetadataProperties['{key}'] contains invalid characters: '{value}'"
