"""Unit tests for container dependency pin validation.

Container version sources:
- sklearn 1.2-1: github.com/aws/sagemaker-scikit-learn-container v1.2-1
- xgboost 1.7-1: github.com/aws/sagemaker-xgboost-container v1.7-1
- pytorch DLC v1.21: github.com/aws/deep-learning-containers v1.21
"""

from pathlib import Path
from packaging.requirements import Requirement
from packaging.markers import default_environment


class TestContainerPins:
    """Test that example/template requirements match container versions."""

    # Container framework versions from SageMaker container repos
    # Only framework packages are validated (sklearn, xgboost, torch)
    # Other packages (numpy, scipy) vary by image and Python version
    CONTAINER_VERSIONS = {
        "sklearn": {
            "scikit-learn": "1.2.1",
        },
        "xgboost": {
            "xgboost": "1.7.4",
        },
        "pytorch": {
            "torch": "2.1.0",
        },
    }

    # Framework package names to check
    FRAMEWORK_PACKAGES = {
        "sklearn": ["scikit-learn"],
        "xgboost": ["xgboost"],
        "pytorch": ["torch"],
    }

    def _parse_requirements(self, req_file: Path, framework: str) -> dict:
        """Parse requirements.txt and extract container pins for container Pythons.

        Returns dict mapping package name to matched version for each container Python version.
        """
        content = req_file.read_text()

        # Container images use Python 3.8, 3.9, or 3.10
        container_pythons = ["3.8", "3.9", "3.10"]
        framework_packages = self.FRAMEWORK_PACKAGES[framework]

        # Track which packages have container pins and which don't
        container_pins = {}
        unmarked_framework_lines = []

        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                req = Requirement(line)
                package_name = req.name.lower()

                # Check if this is a framework package
                if package_name in framework_packages:
                    # Check if it has a marker for Python < 3.11
                    if req.marker:
                        # Evaluate for each container Python version
                        versions_matched = []
                        for py_version in container_pythons:
                            env = default_environment()
                            env["python_version"] = py_version
                            if req.marker.evaluate(env):
                                versions_matched.append(py_version)

                        if versions_matched:
                            # Extract the version that the specifier pins to
                            # We expect exact pins (==) for containers
                            container_pins[package_name] = (
                                req.specifier,
                                versions_matched,
                            )
                    else:
                        # Framework package without marker
                        unmarked_framework_lines.append(line)
            except Exception:
                # Skip unparseable lines
                pass

        # Fail if framework packages have no container marker
        assert not unmarked_framework_lines, (
            f"Framework packages must have python_version < '3.11' marker. "
            f"Found unmarked lines: {unmarked_framework_lines}"
        )

        return container_pins

    def test_sklearn_example_pins(self):
        """Test sklearn example container pins match image versions."""
        req_file = (
            Path(__file__).parent.parent / "examples/sklearn-iris/requirements.txt"
        )
        pins = self._parse_requirements(req_file, "sklearn")

        for package, expected_version in self.CONTAINER_VERSIONS["sklearn"].items():
            assert (
                package in pins
            ), f"{package} container pin not found in {req_file.name}"

            specifier, matched_pythons = pins[package]

            # Check that the pin applies to container Python versions
            assert matched_pythons == [
                "3.8",
                "3.9",
                "3.10",
            ], f"{package} marker does not match container Pythons (3.8-3.10)"

            # Check that exactly the container version is pinned
            assert (
                expected_version in specifier
            ), f"{package} specifier {specifier} does not include container version {expected_version}"  # noqa: E501

            # For exact pins, verify no other version satisfies
            if str(specifier).startswith("=="):
                test_versions = ["1.2.0", "1.2.1", "1.2.2", "1.3.0"]
                matching = [v for v in test_versions if v in specifier]
                assert matching == [
                    expected_version
                ], f"{package} pin {specifier} matches multiple versions: {matching}"

    def test_xgboost_example_pins(self):
        """Test xgboost example container pins match image versions."""
        req_file = (
            Path(__file__).parent.parent / "examples/xgboost-boston/requirements.txt"
        )
        pins = self._parse_requirements(req_file, "xgboost")

        for package, expected_version in self.CONTAINER_VERSIONS["xgboost"].items():
            assert (
                package in pins
            ), f"{package} container pin not found in {req_file.name}"

            specifier, matched_pythons = pins[package]

            # Check that the pin applies to container Python versions
            assert matched_pythons == [
                "3.8",
                "3.9",
                "3.10",
            ], f"{package} marker does not match container Pythons (3.8-3.10)"

            # Check that exactly the container version is pinned
            assert (
                expected_version in specifier
            ), f"{package} specifier {specifier} does not include container version {expected_version}"  # noqa: E501

    def test_pytorch_example_pins(self):
        """Test pytorch example container pins match image versions."""
        req_file = (
            Path(__file__).parent.parent / "examples/pytorch-mnist/requirements.txt"
        )
        pins = self._parse_requirements(req_file, "pytorch")

        for package, expected_version in self.CONTAINER_VERSIONS["pytorch"].items():
            assert (
                package in pins
            ), f"{package} container pin not found in {req_file.name}"

            specifier, matched_pythons = pins[package]

            # Check that the pin applies to container Python versions
            assert matched_pythons == [
                "3.8",
                "3.9",
                "3.10",
            ], f"{package} marker does not match container Pythons (3.8-3.10)"

            # Check that exactly the container version is pinned
            assert (
                expected_version in specifier
            ), f"{package} specifier {specifier} does not include container version {expected_version}"  # noqa: E501

    def test_sklearn_template_pins(self):
        """Test sklearn template container pins match image versions."""
        req_file = (
            Path(__file__).parent.parent
            / "src/mlctl/templates/sklearn/requirements.txt"
        )
        pins = self._parse_requirements(req_file, "sklearn")

        for package, expected_version in self.CONTAINER_VERSIONS["sklearn"].items():
            assert (
                package in pins
            ), f"{package} container pin not found in {req_file.name}"

            specifier, matched_pythons = pins[package]
            assert matched_pythons == ["3.8", "3.9", "3.10"]
            assert expected_version in specifier

    def test_xgboost_template_pins(self):
        """Test xgboost template container pins match image versions."""
        req_file = (
            Path(__file__).parent.parent
            / "src/mlctl/templates/xgboost/requirements.txt"
        )
        pins = self._parse_requirements(req_file, "xgboost")

        for package, expected_version in self.CONTAINER_VERSIONS["xgboost"].items():
            assert (
                package in pins
            ), f"{package} container pin not found in {req_file.name}"

            specifier, matched_pythons = pins[package]
            assert matched_pythons == ["3.8", "3.9", "3.10"]
            assert expected_version in specifier

    def test_pytorch_template_pins(self):
        """Test pytorch template container pins match image versions."""
        req_file = (
            Path(__file__).parent.parent
            / "src/mlctl/templates/pytorch/requirements.txt"
        )
        pins = self._parse_requirements(req_file, "pytorch")

        for package, expected_version in self.CONTAINER_VERSIONS["pytorch"].items():
            assert (
                package in pins
            ), f"{package} container pin not found in {req_file.name}"

            specifier, matched_pythons = pins[package]
            assert matched_pythons == ["3.8", "3.9", "3.10"]
            assert expected_version in specifier

    def test_all_examples_and_templates_have_local_pins(self):
        """Test that all examples and templates have local (3.11+) pins."""
        files = [
            "examples/sklearn-iris/requirements.txt",
            "examples/xgboost-boston/requirements.txt",
            "examples/pytorch-mnist/requirements.txt",
            "src/mlctl/templates/sklearn/requirements.txt",
            "src/mlctl/templates/xgboost/requirements.txt",
            "src/mlctl/templates/pytorch/requirements.txt",
        ]

        for file_path in files:
            req_file = Path(__file__).parent.parent / file_path
            content = req_file.read_text()

            # Should have at least one line with python_version >= "3.11"
            has_local = (
                'python_version >= "3.11"' in content
                or "python_version >= '3.11'" in content
            )
            assert has_local, f"{file_path} missing local (3.11+) pins"

            # Should have at least one line with python_version < "3.11"
            has_container = (
                'python_version < "3.11"' in content
                or "python_version < '3.11'" in content
            )
            assert has_container, f"{file_path} missing container (<3.11) pins"
