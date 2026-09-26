# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-09-26

### Fixed
- **Model package group creation blocker**: Fixed `ensure_model_package_group` to handle AWS SageMaker's actual `ValidationException` error code (with message containing "does not exist") instead of only `ResourceNotFound`. The group is now created with deployment tags so pre-destroy cleanup can find and delete it. Added unit tests with botocore Stubber covering exists, ValidationException does-not-exist → create with tags, and other ClientError → raises.
- **Version number consistency**: Single-sourced version from `__init__.py` in CLI. `mlctl --version` now correctly shows 0.1.2 (was hardcoded 0.1.0).
- **Terraform output command**: Changed README documentation from `terraform output org_config_yaml` to `terraform output -raw org_config_yaml` to avoid heredoc wrapper in output file.
- **Pipeline failure visibility**: `mlctl status` now displays each failed step's `FailureReason` below the step table for better debugging.
- **Team tag reading**: `mlctl list` now reads the `Team` value from pipeline tags instead of showing hardcoded "unknown".
- **Smoke test failure reporting**: `scripts/smoke-test.sh` now prints failed step `FailureReason` when pipeline fails, detects account-level service limit errors (ResourceLimitExceeded for training/processing quotas at 0 Instances), and prints actionable guidance to request quota increases via AWS Service Quotas console. Instance type is now configurable via `SMOKE_INSTANCE_TYPE` environment variable (defaults to ml.m5.large).
- **Project name validation**: Terraform `var.project_name` validation regex now caps at 40 characters (was 42) to ensure the generated bucket name (`${project_name}-artifacts-${account_id}`) stays under S3's 63-character limit. Updated error message and CHANGELOG to reflect 3-40 character range.
- **Tag key validation improvements**: `_validate_tag_key` now uses `re.fullmatch` instead of `re.match` to reject trailing newlines, and performs case-insensitive check for `aws:` prefix. Added comprehensive unit tests for tag key validation including empty key, length limits, aws: prefix variations, and invalid characters (tabs, newlines, semicolons).
- **PyTorch container dependency pins**: Removed `numpy` and `scikit-learn` container pins (python_version < "3.11") from PyTorch example and template requirements.txt files. These were downgrading the DLC's installed versions (numpy 1.26.4→1.24.4, sklearn 1.5.1→1.2.1). Now only `torch==2.1.0` and `torchvision==0.16.0` are pinned for containers, matching the image's other packages.
- **AWS CLI version documentation**: Updated README to document that both AWS CLI v1 and v2 are supported (was "v2 only").

### Added
- **SageMaker quota check in doctor**: `scripts/doctor.sh` now checks SageMaker service quotas for ml.m5.large training and processing instance types via `aws service-quotas`. If quotas are 0, prints a warning (not a failure) with guidance to request increases. Degrades gracefully if service-quotas access is denied or AWS CLI too old.
- **Container pin validation improvements**: Rewrote `tests/test_container_pins.py` to use `packaging.requirements.Requirement` and `packaging.markers` for proper environment marker evaluation across Python 3.8, 3.9, 3.10. Now rejects unmarked framework lines (sklearn/xgboost/torch) and validates that container pins match container Python versions. Accepts both single-quote and double-quote marker forms. Updated `CONTAINER_VERSIONS` table to only validate framework packages (removed incorrect numpy/sklearn values for xgboost and pytorch images). Added source links to container repos and DLC releases. Test now covers templates in addition to examples.
- **Documentation enhancements**: Added explanation of `python_version` marker split in README (container vs local pins), documented `brew install libomp` requirement for local xgboost on macOS, and updated CI job list to mention Python 3.13 and PyTorch dry-run job.

### Changed
- CHANGELOG 0.1.1 entries for "container framework versions" and "3-42 chars" corrected to match actual behavior.

## [0.1.1] - 2026-09-24

### Fixed
- **S3 tag validation error**: Fixed `terraform/s3.tf` tag value containing semicolon (not allowed in AWS S3/SageMaker tags). Changed "force_destroy is enabled for demo purposes; disable for production" to use hyphen instead.
- **Tag sanitization**: Added `_sanitize_tag_value()` method in `pipeline.py` to remove disallowed characters from all user-derived tag values (project name, team, owner, git commit, deployment tags, required tags, and CustomerMetadataProperties). AWS tags only allow letters, numbers, spaces (not tabs/newlines), and `+ - = . _ : / @`. Values are truncated to 256 chars. Added `_validate_tag_key()` to validate tag keys (max 128 chars, no `aws:` prefix, no invalid characters).
- **Container dependency compatibility**: Fixed critical issue where loosened dependency pins upgraded packages inside SageMaker containers, breaking model inference. Example and template `requirements.txt` files now use environment markers to keep container Pythons (3.8-3.10) at container framework versions and only upgrade for local Python 3.11+:
  - sklearn: `==1.2.1` for containers, `>=1.3,<1.6` for local
  - numpy: `>=1.24.1,<1.25` for containers, `>=1.26,<2.2` for local
  - xgboost: `==1.7.4` for containers, `>=1.7.6,<2.2` for local
  - torch: `==2.1.0` for containers, `>=2.1.0,<2.7` for local (note: no Intel Mac wheels for torch on Python 3.13)
- **Train channel naming**: Renamed train.py argument from `--train` to `--training` for consistency with pipeline channel name and `SM_CHANNEL_TRAINING` environment variable. Updated all examples, templates, and README documentation.
- **Non-git checkout handling**: Changed git commit value from `"unknown"` to `"none"` when not in a git repository. Pipeline execution display names now use content hash as identifier for non-git checkouts. Documented behavior in README under "Git Commit Tracking".
- **Makefile improvements**: 
  - `make apply` and `make destroy` now honor `AUTO_APPROVE=1` to skip confirmation prompts
  - `make apply` creates `terraform.tfvars` from example if missing (exits with guidance on first run)
  - `make destroy` now stops by default if pre-destroy cleanup fails (set `FORCE=1` to continue anyway)
  - Updated help text to document `AUTO_APPROVE` and `FORCE` variables

### Added
- **Tag validation unit tests**: New `tests/test_tag_validation.py` with comprehensive tests for AWS tag compliance:
  - `_sanitize_tag_value()` behavior with disallowed characters and length limits
  - Terraform tag literals validation (parses `.tf` files)
  - `_build_tags()` output validation
  - CustomerMetadataProperties sanitization in RegisterModel step
- **Container pin validation tests**: New `tests/test_container_pins.py` validates that example and template requirements.txt container-side pins match SageMaker container framework versions
- **Python 3.13 support**: Added to classifiers in `setup.py` and CI test matrix. Local-run CI jobs now test on both Python 3.9 and 3.13. Added PyTorch dry-run test job.
- **Terraform validation**: Added validation block to `var.project_name` requiring lowercase alphanumeric and hyphens (3-42 chars)

### Changed
- CI local-run jobs now install from actual `requirements.txt` files instead of hard-coded versions
- `scripts/smoke-test.sh` now installs from example `requirements.txt` files
- Updated `pyyaml` from 6.0.1 to 6.0.2 (adds Python 3.13 wheel support)
- Updated README to document `AUTO_APPROVE` and `FORCE` make variables, and the tfvars auto-copy behavior

## [0.1.0] - 2026-09-24

### Added
- Initial release of sagemaker-self-service-training-starter
- `mlctl` CLI with commands: init, validate, run, submit, status, logs, list
- Support for scikit-learn, XGBoost, and PyTorch (CPU) frameworks using SageMaker framework containers
- Local training mode (`mlctl run`) with plain Python (passes hyperparameters as CLI args, checks quality gate)
- SageMaker pipeline generation using boto3 (train → evaluate → quality gate → model registry)
  - SageMaker script mode for training with JSON-encoded hyperparameters
  - Processing job for evaluation with entrypoint wrapper for dependency installation
  - ConditionStep using `Std:JsonGet` for metric comparison
  - FailStep with dynamic error message using `Std:Join` and `Std:JsonGet` showing actual metric value
  - RegisterModel with ModelMetrics, CustomerMetadataProperties (git commit, ml.yaml fields), and PyTorch inference image
- Required evaluation step writing metrics.json
- Model registration as PendingManualApproval after passing quality gate
- Standalone Terraform infrastructure (execution role with least privilege, S3 bucket, no account-wide log groups, deployment-scoped cleanup tags)
- Integration with sagemaker-multi-team-platform-starter via `import-platform-config.py` script (reads `team_details`, requires `--allowlist`)
- Org-level configuration for framework image URIs (overridable per framework in org-config), instance allowlists, deployment tags, and team settings
- Dynamic image URI resolution by framework and region (sklearn/xgboost accounts vary by region, PyTorch uses 763104351884)
- Three working example projects with synthetic data generation:
  - sklearn-iris: RandomForest classification (quality gate: accuracy ≥ 0.75)
  - xgboost-boston: XGBoost regression (quality gate: RMSE ≤ 5.0)
  - pytorch-mnist: CNN classification on class-separable synthetic images (quality gate: accuracy ≥ 0.95)
- GitHub Actions CI: terraform fmt/validate/tflint, Python unit tests (3.9-3.12), schema validation, local runs
- Comprehensive smoke test with scoped cleanup (tagged resources only, temp work dir, ERR/EXIT trap, retries)
- Project-level ml.yaml configuration with strict JSON schema validation (owner, max_runtime_seconds, no preprocessing)
- Git commit and ml.yaml S3 URI tracking in model registry CustomerMetadataProperties
- Per-project model package groups with automatic tagging (deployment-scoped) and creation before pipeline operations
- CloudWatch log streaming for training jobs
- Pipeline status tracking and execution listing
- Unit tests with real SageMaker Python SDK v2 oracle comparison of pipeline structure (dev dependency only)

### Implementation Details
- Pipeline definitions generated directly with boto3 for stability and testability
- Code packaging into sourcedir.tar.gz, evaluation.tar.gz, and ml.yaml upload
- S3 paths unique per submit (hash of freshly built archives) for uncommitted edits / no-git scenarios
- IAM role includes sagemaker:AddTags; execution role does NOT have pipeline create/update/start (caller credentials used)
- Deployment-scoped tagging (`mlctl:deployment`) for safe pre-destroy cleanup across accounts
- pre-destroy.sh deletes resources by deployment tag, collects job names before pipeline deletion, waits for stopped executions, removes log streams
- smoke-test.sh cleanup runs on success and failure, uses EXIT trap, records per-execution S3 prefixes only, bash 3.2/5.x compatible
- Terraform lock file regenerated with multi-platform hashes (darwin_arm64, darwin_amd64, linux_amd64, linux_arm64)

### Known Limitations
- Docker-based local mode is not implemented
- Preprocessing step (`preprocess.py`) is not included; data prep is expected before S3 upload or within train.py
- Full end-to-end smoke test designed for macOS/bash 3.2 but not yet executed on live AWS account
- Training and processing job records cannot be deleted via API (remain in console history, no cost)
- Caller (data scientist) needs pipeline create/update/start, model package group create/describe, S3 put, and iam:PassRole permissions (root user has these by default)
