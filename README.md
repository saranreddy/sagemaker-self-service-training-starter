# SageMaker Self-Service Training Starter

**Bring your `train.py`**: Self-service SageMaker training pipelines for many data scientists, via a small CLI and one shared, tested pipeline template.

[![CI](https://github.com/saranreddy/sagemaker-self-service-training-starter/workflows/CI/badge.svg)](https://github.com/saranreddy/sagemaker-self-service-training-starter/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note**: This is v0.1.0. The design has been validated via CI (unit tests, schema validation, local runs). The full end-to-end smoke test with live SageMaker API calls is designed for macOS/bash 3.2 and has not yet been executed on a live AWS account. Please report any issues you encounter.

## Who Is This For?

This starter is for **organizations with many data scientists (20-100+) sharing a small MLOps team (2-5 engineers)**. It solves the bottleneck where:

- Data scientists wait weeks for MLOps to build custom pipelines
- Copy-pasted pipeline code drifts across 30+ project repos
- Framework upgrades require coordinated changes everywhere
- Platform fixes must be manually applied to each pipeline

With this starter, a data scientist brings a working `train.py` and gets a tested, evaluated, registered model version **the same day**, while MLOps maintains **one shared, tested pipeline template**.

## When NOT to Use This

- **Single small team (3-8 people)**: Use [sagemaker-mlops-pipeline-starter](https://github.com/saranreddy/sagemaker-mlops-pipeline-starter) instead. The overhead of a shared template doesn't pay off at small scale.
- **Large distributed or LLM training**: This starter targets single-node training jobs (ml.m5/c5/g4dn instances). Distributed training requires custom pipeline steps.
- **Non-SageMaker orgs**: If you're on Kubeflow, Vertex AI, or AzureML, this won't help.
- **Streaming feature pipelines**: This is for batch training → evaluation → registry. Feature engineering and real-time inference are out of scope for v0.1.0.
- **You need deployment automation**: This starter **stops at the model registry** with status `PendingManualApproval`. Deployment is manual or separate.

## What You Get

### For Data Scientists

1. **`mlctl init --framework sklearn`** → Working example project that runs locally
2. **`mlctl validate`** → Instant feedback on config, scripts, S3 paths, allowlists
3. **`mlctl run`** → Run training + evaluation locally on a small data sample (plain Python, no Docker)
4. **`mlctl submit`** → Pipeline created/updated and executed; model registered if quality gate passes

One `ml.yaml` file declares the project. No pipeline code to maintain.

### For MLOps Engineers

1. **One Terraform module** → Execution role, S3 bucket, least-privilege IAM policies
2. **One org-level config** → Framework image URIs (with defaults per region), instance allowlists, required tags, deployment-scoped cleanup tags, per-team settings
3. **One pipeline template** → Train → Evaluate → Quality Gate → Register (with git commit and ml.yaml tracking)
4. **Image URI customization** → Override framework image URIs in `org-config.yaml` to pin specific versions or use custom images
5. **Integration with the multi-team platform starter** → Easy import of team roles, buckets, allowlists if [sagemaker-multi-team-platform-starter](https://github.com/saranreddy/sagemaker-multi-team-platform-starter) is deployed

## Quick Start

### For the MLOps Engineer (One-Time Setup)

**Prerequisites**: Terraform 1.5+, AWS CLI v2, AWS credentials, Python 3.9+, `jq`, `make`, `bash` 3.2+

1. **Clone and deploy the infrastructure:**

   ```bash
   git clone https://github.com/saranreddy/sagemaker-self-service-training-starter.git
   cd sagemaker-self-service-training-starter
   make doctor          # Check prerequisites
   cd terraform
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars if needed
   terraform init
   terraform apply
   terraform output org_config_yaml > ../org-config.yaml
   ```

2. **Distribute `org-config.yaml`** to data scientists (via git, S3, or copy to `~/.mlctl/org-config.yaml`)

3. **(Optional) Import settings from the multi-team platform starter:**

   If you've deployed [sagemaker-multi-team-platform-starter](https://github.com/saranreddy/sagemaker-multi-team-platform-starter):

   ```bash
   cd /path/to/sagemaker-multi-team-platform-starter/terraform
   terraform output -json > outputs.json
   # Now import team settings:
   cd /path/to/sagemaker-self-service-training-starter
   python scripts/import-platform-config.py outputs.json > org-config.yaml
   ```

   This reads the platform's team roles, buckets, and allowlists and generates the corresponding `org-config.yaml` automatically.

4. **Run smoke tests** (validates the live infrastructure):

   ```bash
   make smoke
   ```

   This will:
   - Upload synthetic sklearn data to S3
   - Submit a pipeline that passes the quality gate
   - Verify model registration with `PendingManualApproval`
   - Submit a pipeline with a failing quality gate
   - Verify the execution stops at the `QualityGateFailed` step
   - Clean up all created resources

### For the Data Scientist

**Prerequisites**: Python 3.9+, AWS CLI v2 configured with credentials, `org-config.yaml` from MLOps

1. **Install `mlctl`:**

   ```bash
   pip install git+https://github.com/saranreddy/sagemaker-self-service-training-starter.git
   # Or from a local clone:
   pip install -e /path/to/sagemaker-self-service-training-starter
   ```

2. **Initialize a new project:**

   ```bash
   mkdir my-model
   cd my-model
   mlctl init --framework sklearn
   # Creates: ml.yaml, train.py, evaluate.py, requirements.txt, README.md
   ```

3. **Customize for your use case:**

   - Edit `ml.yaml`: set S3 data paths, hyperparameters, quality gate
   - Update `train.py` and `evaluate.py` (keep the SageMaker environment variables contract)
   - Add dependencies to `requirements.txt`

4. **Validate locally:**

   ```bash
   mlctl validate           # Full validation (checks S3 paths)
   mlctl validate --offline # Skip S3 checks (for offline dev)
   ```

5. **Test locally:**

   ```bash
   # Generate a small sample of your data in data/train/ and data/validation/
   mlctl run
   # Runs train.py then evaluate.py with SageMaker env vars, checks quality gate
   ```

6. **Submit to SageMaker:**

   ```bash
   mlctl submit
   # Output: Pipeline execution ARN
   ```

7. **Monitor:**

   ```bash
   mlctl status --project my-model
   mlctl logs --project my-model --follow
   ```

8. **Check the model registry:**

   When the quality gate passes, the model is registered as `PendingManualApproval` in the model package group `<project-name>-models`.

## The `ml.yaml` Contract

```yaml
name: my-model               # Project name (lowercase, hyphens ok, 2-64 chars)
team: data-science           # Team name (used to look up role/bucket/allowlist)
framework: sklearn           # sklearn | xgboost | pytorch
instance_type: ml.m5.large   # Must be in the team's allowlist

data:
  train: s3://bucket/path/to/train/
  validation: s3://bucket/path/to/val/  # optional
  test: s3://bucket/path/to/test/       # optional (used by evaluate.py)

hyperparameters:              # Passed as --arg value and SM_HP_ARG env vars
  max_depth: "5"
  n_estimators: "100"

quality_gate:
  metric: accuracy            # Must appear in metrics.json from evaluate.py
  threshold: 0.90
  direction: maximize         # maximize | minimize
```

## The `train.py` and `evaluate.py` Contract

### `train.py`

SageMaker script mode contract:

- **Arguments**: `--<hyperparam>` for each key in `ml.yaml` hyperparameters, plus `--model-dir`, `--train`, `--validation`
- **Environment variables**: `SM_MODEL_DIR`, `SM_CHANNEL_TRAIN`, `SM_CHANNEL_VALIDATION`, `SM_HP_<KEY>`
- **Output**: Save model artifacts to `$SM_MODEL_DIR` (default `/opt/ml/model`)

See `examples/sklearn-iris/train.py` for a complete example.

### `evaluate.py`

- **Input**: Trained model in `$SM_MODEL_DIR` (or `/opt/ml/processing/model` when run as a processing job)
- **Output**: `metrics.json` in `$SM_OUTPUT_DATA_DIR` (or `/opt/ml/processing/evaluation`)

```json
{
  "accuracy": 0.95,
  "precision": 0.93,
  "recall": 0.94
}
```

The quality gate reads the specified metric from this file.

## How It Works

1. **`mlctl submit`** uploads your code and `ml.yaml` to S3
2. **Pipeline is created/updated** using the shared template:
   - **TrainModel**: Training job running `train.py` with your hyperparameters in SageMaker script mode
   - **EvaluateModel**: Processing job running `evaluate.py`, produces `metrics.json`
   - **QualityGateCheck**: ConditionStep reading the metric from `metrics.json`
     - **Pass** → **RegisterModel**: Create model package version as `PendingManualApproval`, attach evaluation metrics, git commit, ml.yaml metadata, tags
     - **Fail** → **QualityGateFailed**: Fail step with clear message including the actual metric value
3. **Pipeline execution starts** and you monitor via `mlctl status` / `mlctl logs`

## Frameworks and Containers (v0.1.0)

| Framework    | Default Container                                             | CPU/GPU  |
|--------------|---------------------------------------------------------------|----------|
| `sklearn`    | `sagemaker-scikit-learn:1.2-1-cpu-py3`                        | CPU      |
| `xgboost`    | `sagemaker-xgboost:1.7-1`                                     | CPU      |
| `pytorch`    | `pytorch-training:2.1.0-cpu-py310`                            | CPU      |

GPU instances (e.g., `ml.g4dn.xlarge`) are supported if your team's allowlist permits them.

Container images are resolved per framework and region:
- **sklearn/xgboost**: AWS-provided SageMaker containers (account varies by region)
- **pytorch**: AWS Deep Learning Container (account 763104351884, all regions)
- **Custom images**: Override in `org-config.yaml` under `frameworks.<framework>.training_image` and `inference_image`

PyTorch uses a separate inference image for model registration.

## Cost

- **SageMaker Pipelines**: No charge for the pipeline itself
- **Training/Processing Jobs**: You pay for instance minutes (e.g., ml.m5.large ≈ $0.115/hour in us-east-1)
- **S3 and CloudWatch Logs**: Negligible for small experiments
- **Smoke test**: ~5-10 minutes of ml.m5.large = $0.10-0.20

Nothing idles. Costs are incurred only during active pipeline executions.

## GitHub Actions Template for Project Repos

Data scientists can add CI to their ML project repos with this `.github/workflows/ml-ci.yml`:

```yaml
name: ML CI

on:
  pull_request:
    branches: [ main ]
  push:
    branches: [ main ]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install git+https://github.com/saranreddy/sagemaker-self-service-training-starter.git
      - run: mlctl validate --offline
      - run: mlctl run
        if: github.event_name == 'pull_request'

  submit:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    permissions:
      id-token: write
      contents: read
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GithubActionsRole
          aws-region: us-east-1
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install git+https://github.com/saranreddy/sagemaker-self-service-training-starter.git
      - run: mlctl submit
```

This validates on every PR and submits to SageMaker on merge to `main` via GitHub OIDC.

## Terraform Outputs

After `terraform apply`, use `terraform output -json` to get:

- `execution_role_arn`: SageMaker execution role
- `artifact_bucket_name`: S3 bucket for code/data/models
- `region`, `account_id`
- `org_config_yaml`: Ready-to-use `org-config.yaml` content

## Teardown

```bash
make destroy
```

This runs `scripts/pre-destroy.sh` to clean up:

- SageMaker pipelines **tagged with the deployment-scoped tag** (e.g., `mlctl:deployment=sagemaker-self-service-training`)
- Model packages and model package groups **tagged with the deployment tag**
- CloudWatch log streams for jobs from tagged pipelines (in `/aws/sagemaker/TrainingJobs` and `/aws/sagemaker/ProcessingJobs`)
- Running pipeline executions are stopped first and waited for before deletion

Then runs `terraform destroy`.

**Important Notes**:
- `pre-destroy.sh` only deletes resources tagged with the deployment-scoped tag from `org-config.yaml` (`deployment_tag` key)
- SageMaker training and processing **job records** cannot be deleted via API. They remain visible in the console history but do not incur charges.
- The smoke test tracks its own resources by name and cleans them up on both success and failure.

## v0.1.0 Scope Exclusions

The following are **intentionally excluded** from v0.1.0 and may appear in future releases:

- **Deployment**: Models stop at `PendingManualApproval`. Deployment to endpoints is separate.
- **Preprocessing step**: `preprocess.py` is not included in v0.1.0. Data preprocessing is expected to be done before uploading to S3, or within `train.py`.
- **Hyperparameter tuning**: No `HyperparameterTuner` step.
- **Distributed/multi-GPU training**: Single-node jobs only.
- **SageMaker Feature Store**: Not integrated.
- **Data quality monitoring**: No `DataQualityCheck` steps.
- **Model explainability**: No Clarify integration.
- **Notebook conversion**: Bring a `.py` script, not a `.ipynb`.
- **Built-in algorithms**: Framework containers (sklearn, xgboost, pytorch) only; no built-in image classification, etc.
- **Custom Docker images**: Uses AWS Deep Learning Containers; custom images require code changes to `src/mlctl/image_uris.py`.

## Examples

Three complete working examples are included in `examples/`:

1. **`sklearn-iris/`**: Random Forest classifier on synthetic Iris-like data (quality gate: accuracy ≥ 0.75)
2. **`xgboost-boston/`**: XGBoost regression on synthetic Boston Housing-like data (quality gate: RMSE ≤ 5.0)
3. **`pytorch-mnist/`**: CNN on synthetic class-separable MNIST-like image data (quality gate: accuracy ≥ 0.95)

Each includes:

- `ml.yaml`, `train.py`, `evaluate.py`, `requirements.txt`
- `generate_data.py` to create synthetic data (no external downloads)
- README with instructions

Run any example locally:

```bash
cd examples/sklearn-iris
python generate_data.py
mlctl validate --offline
mlctl run
```

## Integration with Multi-Team Platform Starter

If you've deployed [sagemaker-multi-team-platform-starter](https://github.com/saranreddy/sagemaker-multi-team-platform-starter), you can import its team configurations:

```bash
# In the multi-team platform repo:
cd terraform
terraform output -json > /tmp/platform-outputs.json

# In this repo:
python scripts/import-platform-config.py /tmp/platform-outputs.json > org-config.yaml
```

This reads the platform's `team_details` output and generates an `org-config.yaml` with per-team:

- `execution_role` (from the team's SageMaker role)
- `artifact_bucket` (from the team's S3 bucket)
- `allowed_instance_types` (must be provided via `--allowlist` or defaults; platform does not export this)

Projects with `team: data-science` will automatically use the data-science team's role, bucket, and allowlist.

## Development

### Running Tests

```bash
make test    # Python unit tests
make lint    # flake8 and black
```

### CI

GitHub Actions runs on every push and PR:

- Terraform fmt/validate/tflint
- Python unit tests (3.9, 3.10, 3.11, 3.12)
- Validate all example projects offline
- Run sklearn and xgboost examples locally (pytorch skipped due to large torch install)

## Design Decisions

### Why boto3 Instead of SageMaker Python SDK?

The SageMaker Python SDK v2 → v3 migration caused breaking changes across many projects. By building pipeline definitions directly with boto3 and pinned container URI maps, we get:

- **Stability**: SDK upgrades don't break our pipeline generation
- **Testability**: Pipeline definitions are JSON; easy golden-file unit tests
- **Lightweight**: No heavyweight SDK dependency (boto3 is already everywhere)
- **Transparency**: The pipeline definition is explicit, not hidden behind SDK abstractions

Trade-off: We lose the SDK's helper methods. We accept this for the stability and testability gains.

### Why Plain Python Local Mode by Default?

Docker-based SageMaker local mode is powerful but adds setup friction (Docker Desktop on Mac, etc.). Most data scientists want to quickly test on a small sample. Plain Python with `SM_*` env vars achieves this with zero Docker dependencies.

Docker-based local mode is not implemented in v0.1.0.

### Why Stop at the Model Registry?

Deployment strategies vary wildly:

- Some orgs deploy to SageMaker endpoints
- Some export to Kubernetes with KServe
- Some batch-score and cache results in S3
- Some require manual approval and staging promotions

Rather than build one opinionated deployment path, we stop at `PendingManualApproval` and let each org wire up their deployment automation separately.

## Known Considerations for the Live Smoke Test

The smoke test (`make smoke`) is designed for **macOS with bash 3.2** (also works on Linux with bash 4+) and requires:

- **Valid AWS credentials** with permissions to SageMaker, S3, IAM (read-only for `sts:GetCallerIdentity`), CloudWatch Logs
- **Deployed infrastructure** (`make apply` must succeed first)
- **Region us-east-1** (or edit `terraform/terraform.tfvars` to change region; `smoke-test.sh` reads from Terraform outputs)
- **~10-15 minutes** for pipeline executions (2 pipelines: one pass with threshold 0.70, one fail with impossible threshold 1.01)
- **`jq` installed** for JSON parsing of `mlctl submit --output json`

**Root user compatibility**: If the deployer is the AWS account **root user**, they cannot `sts:AssumeRole`. The Terraform design passes the execution role ARN to SageMaker (which does not require the caller to assume it), so this works correctly.

**Cleanup**: The smoke test runs cleanup on both success and failure via an EXIT trap. It deletes only the resources it created: pipelines, model packages, model package groups, all S3 prefixes (including `smoke-test/`, `code/<project>/`, `pipelines/`), and log streams by job name. It does not affect other projects or pipelines in the account.

## Contributing

Contributions welcome! Open an issue or PR.

## License

MIT License - see [LICENSE](LICENSE)

## Related Projects

- [sagemaker-mlops-pipeline-starter](https://github.com/saranreddy/sagemaker-mlops-pipeline-starter) - For small teams who want per-project pipelines
- [sagemaker-multi-team-platform-starter](https://github.com/saranreddy/sagemaker-multi-team-platform-starter) - Multi-tenant platform with per-team VPCs, roles, budgets

## Support

This is a community-maintained starter template, not an official AWS product. For issues, open a GitHub issue or PR.
