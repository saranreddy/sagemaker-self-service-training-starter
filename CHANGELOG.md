# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-24

### Added
- Initial release of sagemaker-self-service-training-starter
- `mlctl` CLI with commands: init, validate, run, submit, status, logs, list
- Support for scikit-learn, XGBoost, and PyTorch (CPU) frameworks
- Local training mode (`mlctl run --local`) with plain Python or optional Docker
- SageMaker pipeline generation with train → evaluate → quality gate → model registry
- Required evaluation step writing metrics.json
- Model registration as PendingManualApproval after passing quality gate
- Standalone Terraform infrastructure (execution role, S3 bucket, CloudWatch logs)
- Integration path for sagemaker-multi-team-platform-starter
- Org-level configuration for framework versions, instance allowlists, and team settings
- Three working example projects: sklearn (iris classification), xgboost (boston housing), pytorch (mnist)
- Synthetic data generation for all examples
- GitHub Actions CI: terraform validation, Python unit tests, example validation and local runs
- Comprehensive smoke test suite for end-to-end validation
- Project-level ml.yaml configuration with strict validation
- Quality gate implementation (ConditionStep with metric threshold)
- Git commit tracking and ml.yaml attachment in model registry metadata
- Per-project model package groups with automatic tagging
- CloudWatch log streaming for training jobs
- Pipeline status tracking and execution listing
