"""Unit tests for container dependency pin validation."""

import re
from pathlib import Path


class TestContainerPins:
    """Test that example/template requirements match container versions."""

    # Container framework versions from SageMaker container repos
    CONTAINER_VERSIONS = {
        "sklearn": {
            "scikit-learn": "1.2.1",
            "numpy": "1.24.1",
        },
        "xgboost": {
            "xgboost": "1.7.4",
            "numpy": "1.24.1",
            "scikit-learn": "1.2.1",
        },
        "pytorch": {
            "torch": "2.1.0",
            "numpy": "1.24.1",
            "scikit-learn": "1.2.1",
        },
    }

    def _parse_requirements(self, req_file: Path) -> dict:
        """Parse requirements.txt and extract container pins (python<3.11)."""
        content = req_file.read_text()
        pins = {}

        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Match: package==version; python_version < "3.11"
            # or: package>=version,<max; python_version < "3.11"
            match = re.match(
                r'^([a-z0-9-]+)==([0-9.]+);\s*python_version\s*<\s*"3\.11"$',
                line,
            )
            if not match:
                match = re.match(
                    r'^([a-z0-9-]+)>=([0-9.]+),<[0-9.]+;\s*python_version\s*<\s*"3\.11"$',
                    line,
                )
            if match:
                package, version = match.groups()
                pins[package] = version

        return pins

    def test_sklearn_example_pins(self):
        """Test sklearn example container pins match image versions."""
        req_file = (
            Path(__file__).parent.parent / "examples/sklearn-iris/requirements.txt"
        )
        pins = self._parse_requirements(req_file)

        for package, expected_version in self.CONTAINER_VERSIONS["sklearn"].items():
            assert (
                package in pins
            ), f"{package} container pin not found in {req_file.name}"
            assert (
                pins[package] == expected_version
            ), f"{package} pin {pins[package]} != container version {expected_version}"

    def test_xgboost_example_pins(self):
        """Test xgboost example container pins match image versions."""
        req_file = (
            Path(__file__).parent.parent / "examples/xgboost-boston/requirements.txt"
        )
        pins = self._parse_requirements(req_file)

        for package, expected_version in self.CONTAINER_VERSIONS["xgboost"].items():
            assert (
                package in pins
            ), f"{package} container pin not found in {req_file.name}"
            assert (
                pins[package] == expected_version
            ), f"{package} pin {pins[package]} != container version {expected_version}"

    def test_pytorch_example_pins(self):
        """Test pytorch example container pins match image versions."""
        req_file = (
            Path(__file__).parent.parent / "examples/pytorch-mnist/requirements.txt"
        )
        pins = self._parse_requirements(req_file)

        for package, expected_version in self.CONTAINER_VERSIONS["pytorch"].items():
            assert (
                package in pins
            ), f"{package} container pin not found in {req_file.name}"
            assert (
                pins[package] == expected_version
            ), f"{package} pin {pins[package]} != container version {expected_version}"

    def test_sklearn_template_pins(self):
        """Test sklearn template container pins match image versions."""
        req_file = (
            Path(__file__).parent.parent
            / "src/mlctl/templates/sklearn/requirements.txt"
        )
        pins = self._parse_requirements(req_file)

        for package, expected_version in self.CONTAINER_VERSIONS["sklearn"].items():
            assert (
                package in pins
            ), f"{package} container pin not found in {req_file.name}"
            assert (
                pins[package] == expected_version
            ), f"{package} pin {pins[package]} != container version {expected_version}"

    def test_xgboost_template_pins(self):
        """Test xgboost template container pins match image versions."""
        req_file = (
            Path(__file__).parent.parent
            / "src/mlctl/templates/xgboost/requirements.txt"
        )
        pins = self._parse_requirements(req_file)

        for package, expected_version in self.CONTAINER_VERSIONS["xgboost"].items():
            assert (
                package in pins
            ), f"{package} container pin not found in {req_file.name}"
            assert (
                pins[package] == expected_version
            ), f"{package} pin {pins[package]} != container version {expected_version}"

    def test_pytorch_template_pins(self):
        """Test pytorch template container pins match image versions."""
        req_file = (
            Path(__file__).parent.parent
            / "src/mlctl/templates/pytorch/requirements.txt"
        )
        pins = self._parse_requirements(req_file)

        for package, expected_version in self.CONTAINER_VERSIONS["pytorch"].items():
            assert (
                package in pins
            ), f"{package} container pin not found in {req_file.name}"
            assert (
                pins[package] == expected_version
            ), f"{package} pin {pins[package]} != container version {expected_version}"

    def test_all_examples_have_local_pins(self):
        """Test that all examples also have local (3.11+) pins."""
        examples = [
            "examples/sklearn-iris/requirements.txt",
            "examples/xgboost-boston/requirements.txt",
            "examples/pytorch-mnist/requirements.txt",
        ]

        for example in examples:
            req_file = Path(__file__).parent.parent / example
            content = req_file.read_text()

            # Should have at least one line with python_version >= "3.11"
            assert (
                'python_version >= "3.11"' in content
            ), f"{example} missing local (3.11+) pins"

            # Should have at least one line with python_version < "3.11"
            assert (
                'python_version < "3.11"' in content
            ), f"{example} missing container (<3.11) pins"
