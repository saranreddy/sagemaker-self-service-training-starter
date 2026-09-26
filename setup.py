"""Setup configuration for mlctl."""
from setuptools import setup, find_packages
import os

# Read version from __init__.py (relative to this file)
version = {}
init_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "src", "mlctl", "__init__.py"
)
with open(init_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)
            break

readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
with open(readme_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlctl-sagemaker",
    version=version["__version__"],
    author="SageMaker Self-Service Training Starter Contributors",
    description="CLI for self-service SageMaker training pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saranreddy/sagemaker-self-service-training-starter",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.9",
    install_requires=[
        "boto3==1.34.84",
        "botocore==1.34.84",
        "pyyaml==6.0.2",
        "jsonschema==4.21.1",
        "click==8.1.7",
        "rich==13.7.1",
    ],
    entry_points={
        "console_scripts": [
            "mlctl=mlctl.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mlctl": ["templates/**/*", "schemas/*.json"],
    },
)
