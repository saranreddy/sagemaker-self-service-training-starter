"""Tests for validation module."""
import tempfile
from pathlib import Path

import pytest
import yaml

from mlctl.config import Config
from mlctl.validation import ProjectValidator


@pytest.fixture
def valid_project():
    """Create a valid test project."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        
        ml_yaml = {
            "name": "test-project",
            "team": "test-team",
            "framework": "sklearn",
            "instance_type": "ml.m5.large",
            "data": {
                "train": "s3://bucket/train/"
            },
            "hyperparameters": {
                "max_depth": "5"
            },
            "quality_gate": {
                "metric": "accuracy",
                "threshold": 0.9,
                "direction": "maximize"
            }
        }
        
        with open(project_dir / "ml.yaml", 'w') as f:
            yaml.dump(ml_yaml, f)
        
        (project_dir / "train.py").write_text("print('training')")
        (project_dir / "evaluate.py").write_text("print('evaluating')")
        (project_dir / "requirements.txt").write_text("numpy==1.24.3")
        
        yield str(project_dir)


def test_schema_validation_valid(valid_project):
    """Test schema validation with valid config."""
    config = Config()
    validator = ProjectValidator(config, valid_project, offline=True)
    
    ml_config = config.load_project_config(valid_project)
    validator._validate_schema(ml_config)
    
    assert len(validator.errors) == 0


def test_schema_validation_invalid():
    """Test schema validation with invalid config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        
        ml_yaml = {
            "name": "test-project",
            "framework": "sklearn"
        }
        
        with open(project_dir / "ml.yaml", 'w') as f:
            yaml.dump(ml_yaml, f)
        
        config = Config()
        validator = ProjectValidator(config, str(project_dir), offline=True)
        
        ml_config = config.load_project_config(str(project_dir))
        validator._validate_schema(ml_config)
        
        assert len(validator.errors) > 0


def test_framework_validation():
    """Test framework validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        
        ml_yaml = {
            "name": "test-project",
            "team": "test-team",
            "framework": "unsupported",
            "instance_type": "ml.m5.large",
            "data": {"train": "s3://bucket/train/"},
            "hyperparameters": {},
            "quality_gate": {
                "metric": "accuracy",
                "threshold": 0.9,
                "direction": "maximize"
            }
        }
        
        with open(project_dir / "ml.yaml", 'w') as f:
            yaml.dump(ml_yaml, f)
        
        config = Config()
        validator = ProjectValidator(config, str(project_dir), offline=True)
        
        ml_config = config.load_project_config(str(project_dir))
        validator._validate_framework(ml_config)
        
        assert len(validator.errors) > 0


def test_instance_type_validation():
    """Test instance type allowlist validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        
        ml_yaml = {
            "name": "test-project",
            "team": "test-team",
            "framework": "sklearn",
            "instance_type": "ml.p3.8xlarge",
            "data": {"train": "s3://bucket/train/"},
            "hyperparameters": {},
            "quality_gate": {
                "metric": "accuracy",
                "threshold": 0.9,
                "direction": "maximize"
            }
        }
        
        with open(project_dir / "ml.yaml", 'w') as f:
            yaml.dump(ml_yaml, f)
        
        config = Config()
        validator = ProjectValidator(config, str(project_dir), offline=True)
        
        ml_config = config.load_project_config(str(project_dir))
        validator._validate_instance_type(ml_config)
        
        assert len(validator.errors) > 0


def test_scripts_validation(valid_project):
    """Test script validation."""
    config = Config()
    validator = ProjectValidator(config, valid_project, offline=True)
    
    validator._validate_scripts()
    
    assert len(validator.errors) == 0


def test_missing_scripts():
    """Test validation with missing scripts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        
        ml_yaml = {
            "name": "test-project",
            "team": "test-team",
            "framework": "sklearn",
            "instance_type": "ml.m5.large",
            "data": {"train": "s3://bucket/train/"},
            "hyperparameters": {},
            "quality_gate": {
                "metric": "accuracy",
                "threshold": 0.9,
                "direction": "maximize"
            }
        }
        
        with open(project_dir / "ml.yaml", 'w') as f:
            yaml.dump(ml_yaml, f)
        
        config = Config()
        validator = ProjectValidator(config, str(project_dir), offline=True)
        
        validator._validate_scripts()
        
        assert len(validator.errors) >= 2


def test_python_syntax_check():
    """Test Python syntax checking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        
        valid_script = project_dir / "valid.py"
        valid_script.write_text("print('hello')")
        
        invalid_script = project_dir / "invalid.py"
        invalid_script.write_text("print('hello'")
        
        config = Config()
        validator = ProjectValidator(config, str(project_dir), offline=True)
        
        assert validator._check_python_syntax(valid_script) is True
        assert validator._check_python_syntax(invalid_script) is False
