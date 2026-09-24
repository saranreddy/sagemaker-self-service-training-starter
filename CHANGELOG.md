# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-24

### Added
- Initial release of sagemaker-self-service-training-starter
- `mlctl` CLI with commands: init, validate, run, submit, status, logs, list
- Support for scikit-learn, XGBoost, and PyTorch (CPU) frameworks using AWS Deep Learning Containers
- Local training mode (`mlctl run --local`) with plain Python (passes hyperparameters as CLI args, checks quality gate)
- SageMaker pipeline generation using boto3 (train → evaluate → quality gate → model registry)
  - SageMaker script mode for training with JSON-encoded hyperparameters
  - Processing job for evaluation with entrypoint wrapper for dependency installation
  - ConditionStep using `Std:JsonGet` for metric comparison
  - FailStep with dynamic error message using `Std:Join` and `Std:JsonGet` showing actual metric value
  - RegisterModel with ModelMetrics, CustomerMetadataProperties (git commit, ml.yaml fields), and PyTorch inference image
- Required evaluation step writing metrics.json
- Model registration as PendingManualApproval after passing quality gate
- Standalone Terraform infrastructure (execution role with least privilege, S3 bucket, no account-wide log groups)
- Integration with sagemaker-multi-team-platform-starter via `import-platform-config.py` script
- Org-level configuration for framework versions, instance allowlists, and team settings
- Dynamic image URI resolution by framework and region (separate PyTorch inference image account)
- Three working example projects with synthetic data generation:
  - sklearn-iris: RandomForest classification (quality gate: accuracy ≥ 0.75)
  - xgboost-boston: XGBoost regression (quality gate: RMSE ≤ 5.0)
  - pytorch-mnist: CNN classification on class-separable synthetic images (quality gate: accuracy ≥ 0.95)
- GitHub Actions CI: terraform fmt/validate/tflint, Python unit tests (3.9-3.12), schema validation, local runs
- Comprehensive smoke test with scoped cleanup (tagged resources only, temp work dir, ERR/EXIT trap, retries)
- Project-level ml.yaml configuration with strict JSON schema validation (owner, max_runtime_seconds, no preprocessing)
- Git commit tracking in S3 code paths and model registry metadata
- Per-project model package groups with automatic tagging and creation before pipeline operations
- CloudWatch log streaming for training jobs
- Pipeline status tracking and execution listing
- Unit tests with SageMaker Python SDK v2 as a test oracle for schema validation (dev dependency only)

### Implementation Details
- Pipeline definitions generated directly with boto3 for stability and testability
- Code packaging into sourcedir.tar.gz and evaluation.tar.gz with content hashing
- S3 paths include git commit hash for uniqueness
- IAM role includes sagemaker:AddTags, execution role does not manage pipeline lifecycle
- pre-destroy.sh and smoke-test.sh only affect project-tagged or recorded-name resources
- Terraform lock file regenerated with multi-platform hashes (darwin_arm64, darwin_amd64, linux_amd64, linux_arm64)

### Known Limitations
- Docker-based local mode (`--docker` flag) is not implemented; defaults to plain Python
- Preprocessing step (`preprocess.py`) is not included; data prep is expected before S3 upload or within train.py
- Full end-to-end smoke test designed for macOS/bash 3.2 but not yet executed on live AWS account
- Training and processing job records cannot be deleted via API (remain in console history, no cost)
