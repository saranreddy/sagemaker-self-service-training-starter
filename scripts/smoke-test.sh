#!/usr/bin/env bash
set -euo pipefail

echo "=== SageMaker Self-Service Training Smoke Test ==="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Track resources for cleanup
CREATED_PIPELINES=()
CREATED_EXECUTIONS=()
CREATED_MODEL_GROUPS=()
CREATED_S3_PREFIXES=()

CLEANUP_DONE=false
WORK_DIR=""

cleanup() {
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi
    
    echo ""
    echo "=== Cleanup ==="
    
    # Stop running executions (bash 3.2 safe)
    if [ ${#CREATED_EXECUTIONS[@]} -gt 0 ]; then
        for exec_arn in "${CREATED_EXECUTIONS[@]}"; do
            echo "Stopping execution: $exec_arn"
            aws sagemaker stop-pipeline-execution --pipeline-execution-arn "$exec_arn" 2>/dev/null || true
        done
        
        # Wait for executions to reach terminal state before deleting pipelines
        echo "Waiting for executions to stop..."
        for exec_arn in "${CREATED_EXECUTIONS[@]}"; do
            local max_stop_wait=60
            local stop_elapsed=0
            while [ $stop_elapsed -lt $max_stop_wait ]; do
                status=$(aws sagemaker describe-pipeline-execution \
                    --pipeline-execution-arn "$exec_arn" \
                    --query 'PipelineExecutionStatus' \
                    --output text 2>/dev/null || echo "UNKNOWN")
                
                if [[ "$status" =~ ^(Succeeded|Failed|Stopped)$ ]]; then
                    break
                fi
                
                sleep 2
                stop_elapsed=$((stop_elapsed + 2))
            done
        done
    fi
    
    # Collect job names from pipeline executions before deleting
    local JOB_NAMES=()
    if [ ${#CREATED_EXECUTIONS[@]} -gt 0 ]; then
        for exec_arn in "${CREATED_EXECUTIONS[@]}"; do
            echo "Collecting job names from $exec_arn..."
            steps=$(aws sagemaker list-pipeline-execution-steps \
                --pipeline-execution-arn "$exec_arn" \
                --query 'PipelineExecutionSteps[].Metadata' \
                --output json 2>/dev/null || echo "[]")
            
            # Extract training job names
            training_jobs=$(echo "$steps" | jq -r '.[] | select(.TrainingJob != null) | .TrainingJob.Arn' | sed 's|.*/||' || true)
            for job in $training_jobs; do
                [ -n "$job" ] && JOB_NAMES+=("$job")
            done
            
            # Extract processing job names
            processing_jobs=$(echo "$steps" | jq -r '.[] | select(.ProcessingJob != null) | .ProcessingJob.Arn' | sed 's|.*/||' || true)
            for job in $processing_jobs; do
                [ -n "$job" ] && JOB_NAMES+=("$job")
            done
        done
    fi
    
    # Delete model packages
    if [ ${#CREATED_MODEL_GROUPS[@]} -gt 0 ]; then
        for group in "${CREATED_MODEL_GROUPS[@]}"; do
            echo "Deleting model packages in group: $group"
            packages=$(aws sagemaker list-model-packages \
                --model-package-group-name "$group" \
                --query 'ModelPackageSummaryList[].ModelPackageArn' \
                --output text 2>/dev/null || true)
            
            for arn in $packages; do
                [ -n "$arn" ] && aws sagemaker delete-model-package --model-package-name "$arn" 2>/dev/null || true
            done
            
            echo "Deleting model package group: $group"
            aws sagemaker delete-model-package-group --model-package-group-name "$group" 2>/dev/null || true
        done
    fi
    
    # Delete pipelines
    if [ ${#CREATED_PIPELINES[@]} -gt 0 ]; then
        for pipeline in "${CREATED_PIPELINES[@]}"; do
            echo "Deleting pipeline: $pipeline"
            aws sagemaker delete-pipeline --pipeline-name "$pipeline" 2>/dev/null || true
        done
    fi
    
    # Delete log streams
    if [ ${#JOB_NAMES[@]} -gt 0 ]; then
        for job_name in "${JOB_NAMES[@]}"; do
            for log_group in "/aws/sagemaker/TrainingJobs" "/aws/sagemaker/ProcessingJobs"; do
                streams=$(aws logs describe-log-streams \
                    --log-group-name "$log_group" \
                    --log-stream-name-prefix "$job_name" \
                    --query 'logStreams[].logStreamName' \
                    --output text 2>/dev/null || true)
                
                for stream in $streams; do
                    [ -n "$stream" ] && aws logs delete-log-stream \
                        --log-group-name "$log_group" \
                        --log-stream-name "$stream" 2>/dev/null || true
                done
            done
        done
    fi
    
    # Delete S3 prefixes
    if [ ${#CREATED_S3_PREFIXES[@]} -gt 0 ]; then
        for prefix in "${CREATED_S3_PREFIXES[@]}"; do
            echo "Deleting S3 prefix: $prefix"
            aws s3 rm "$prefix" --recursive 2>/dev/null || true
        done
    fi
    
    # Clean up work directory
    if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then
        rm -rf "$WORK_DIR"
    fi
    
    CLEANUP_DONE=true
    echo "Cleanup complete"
}

fail() {
    local msg="$1"
    echo ""
    echo "FAIL: $msg"
    cleanup
    exit 1
}

trap 'fail "Script failed at line $LINENO"' ERR
trap cleanup EXIT

# Check prerequisites
command -v jq >/dev/null || fail "jq not found"
command -v aws >/dev/null || fail "aws CLI not found"

# Create venv and install dependencies
cd "$ROOT_DIR"
if [ ! -d "venv" ]; then
    python3 -m venv venv || fail "Failed to create venv"
fi

source venv/bin/activate || fail "Failed to activate venv"

# Install mlctl and example dependencies
pip install -q -e . || fail "Failed to install mlctl"
pip install -q "scikit-learn>=1.2.2,<1.6" "numpy>=1.24.3,<2.2" "xgboost>=1.7.6,<2.2" || fail "Failed to install example deps"

# Get AWS details
cd terraform
if [ ! -f "terraform.tfstate" ]; then
    fail "Terraform not deployed. Run 'make apply' first."
fi

EXECUTION_ROLE=$(terraform output -raw execution_role_arn) || fail "Failed to get execution role"
ARTIFACT_BUCKET=$(terraform output -raw artifact_bucket_name) || fail "Failed to get artifact bucket"
REGION=$(terraform output -raw region) || fail "Failed to get region"

export MLCTL_ARTIFACT_BUCKET="$ARTIFACT_BUCKET"
export AWS_DEFAULT_REGION="$REGION"

echo "Configuration:"
echo "  Execution Role: $EXECUTION_ROLE"
echo "  Artifact Bucket: $ARTIFACT_BUCKET"
echo "  Region: $REGION"
echo ""

# Create temp work directory
WORK_DIR=$(mktemp -d)

cp -r "$ROOT_DIR/examples/sklearn-iris"/* "$WORK_DIR/"
cd "$WORK_DIR"

# Get deployment tag from terraform output
cd "$ROOT_DIR/terraform"
DEPLOYMENT_TAG=$(terraform output -raw deployment_tag 2>/dev/null || echo "mlctl:deployment=sagemaker-self-service-training")
TAG_KEY=$(echo "$DEPLOYMENT_TAG" | cut -d= -f1)
TAG_VALUE=$(echo "$DEPLOYMENT_TAG" | cut -d= -f2-)
cd "$WORK_DIR"

# Create org-config.yaml in temp dir with deployment_tag
cat > org-config.yaml <<EOF
execution_role: $EXECUTION_ROLE
artifact_bucket: $ARTIFACT_BUCKET

frameworks:
  sklearn:
    default_instance_type: "ml.m5.large"
    version: "1.2-1"

default_instance_type: "ml.m5.large"

allowed_instance_types:
  - ml.m5.large
  - ml.m5.xlarge

required_tags:
  Project: sagemaker-self-service-training

deployment_tag:
  $TAG_KEY: $TAG_VALUE

teams: {}
EOF

echo "=== Test 1: Generate and upload data ==="
python3 generate_data.py || fail "Data generation failed"

PROJECT_NAME="smoke-test-iris"


# Record S3 prefixes that will be created (execution-specific prefixes added after submit)
CREATED_S3_PREFIXES+=("s3://$ARTIFACT_BUCKET/smoke-test/")
CREATED_S3_PREFIXES+=("s3://$ARTIFACT_BUCKET/code/$PROJECT_NAME/")

aws s3 sync data/train/ "s3://$ARTIFACT_BUCKET/smoke-test/iris/train/" --quiet || fail "S3 upload failed"
aws s3 sync data/validation/ "s3://$ARTIFACT_BUCKET/smoke-test/iris/validation/" --quiet || fail "S3 upload failed"

echo "✓ Data uploaded"
echo ""

# Create ml.yaml with passing threshold (with margin)
cat > ml.yaml <<EOF
name: $PROJECT_NAME
team: smoke-test
framework: sklearn
instance_type: ml.m5.large

data:
  train: s3://$ARTIFACT_BUCKET/smoke-test/iris/train/
  validation: s3://$ARTIFACT_BUCKET/smoke-test/iris/validation/

hyperparameters:
  max_depth: 5
  n_estimators: 100
  random_state: 42

quality_gate:
  metric: accuracy
  threshold: 0.70
  direction: maximize
EOF

CREATED_PIPELINES+=("${PROJECT_NAME}-pipeline")
CREATED_MODEL_GROUPS+=("${PROJECT_NAME}-models")

echo "=== Test 2: Validate project ==="
mlctl validate --offline || fail "Validation failed"
echo "✓ Validation passed"
echo ""

echo "=== Test 3: Run locally ==="
mlctl run || fail "Local run failed"
echo "✓ Local run passed"
echo ""

echo "=== Test 4: Submit to SageMaker (success case) ==="
submit_output=$(mlctl submit --skip-validation --output json) || fail "Submit failed"
EXECUTION_ARN=$(echo "$submit_output" | jq -r '.execution_arn') || fail "Failed to parse execution ARN"
[ -n "$EXECUTION_ARN" ] || fail "Empty execution ARN"
CREATED_EXECUTIONS+=("$EXECUTION_ARN")

# Record execution-specific pipeline output prefix
EXEC_ID="${EXECUTION_ARN##*/}"
CREATED_S3_PREFIXES+=("s3://$ARTIFACT_BUCKET/pipelines/$EXEC_ID/")

echo "✓ Pipeline submitted: $EXECUTION_ARN"
echo ""

echo "=== Test 5: Wait for pipeline completion ==="
MAX_WAIT=1200
ELAPSED=0
SLEEP_INTERVAL=30
CONSECUTIVE_ERRORS=0
MAX_CONSECUTIVE_ERRORS=5

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! STATUS=$(aws sagemaker describe-pipeline-execution \
        --pipeline-execution-arn "$EXECUTION_ARN" \
        --query 'PipelineExecutionStatus' \
        --output text 2>/dev/null); then
        CONSECUTIVE_ERRORS=$((CONSECUTIVE_ERRORS + 1))
        echo "  Transient describe error ($CONSECUTIVE_ERRORS/$MAX_CONSECUTIVE_ERRORS), retrying..."
        if [ $CONSECUTIVE_ERRORS -ge $MAX_CONSECUTIVE_ERRORS ]; then
            fail "Too many consecutive describe errors"
        fi
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        continue
    fi
    
    CONSECUTIVE_ERRORS=0
    echo "  Status: $STATUS (waited ${ELAPSED}s)"
    
    if [ "$STATUS" = "Succeeded" ]; then
        echo "✓ Pipeline succeeded"
        break
    elif [ "$STATUS" = "Failed" ] || [ "$STATUS" = "Stopped" ]; then
        fail "Pipeline $STATUS unexpectedly"
    fi
    
    sleep $SLEEP_INTERVAL
    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))
done

[ $ELAPSED -lt $MAX_WAIT ] || fail "Pipeline did not complete within ${MAX_WAIT}s"
echo ""

echo "=== Test 6: Verify model registration ==="
model_package=$(aws sagemaker list-model-packages \
    --model-package-group-name "${PROJECT_NAME}-models" \
    --max-results 1 \
    --query 'ModelPackageSummaryList[0]' \
    --output json) || fail "Failed to list model packages"

[ "$model_package" != "null" ] || fail "No model package found"

# Check ModelApprovalStatus
model_status=$(echo "$model_package" | jq -r '.ModelApprovalStatus')
[ "$model_status" = "PendingManualApproval" ] || fail "Model status is $model_status, expected PendingManualApproval"
echo "✓ Model approval status correct: $model_status"

# Get full model package details
model_arn=$(echo "$model_package" | jq -r '.ModelPackageArn')
model_details=$(aws sagemaker describe-model-package --model-package-name "$model_arn" --output json) || fail "Failed to describe model package"

# Check ModelMetrics exists and head-object the S3 URI
model_metrics=$(echo "$model_details" | jq '.ModelMetrics')
[ "$model_metrics" != "null" ] || fail "ModelMetrics not found"
metrics_s3_uri=$(echo "$model_details" | jq -r '.ModelMetrics.ModelQuality.Statistics.S3Uri // empty')
[ -n "$metrics_s3_uri" ] || fail "ModelMetrics.ModelQuality.Statistics.S3Uri is empty"
    
    aws s3api head-object --bucket "$ARTIFACT_BUCKET" --key "${metrics_s3_uri#s3://$ARTIFACT_BUCKET/}" >/dev/null 2>&1 || \
        fail "metrics.json not found at $metrics_s3_uri"

echo "✓ ModelMetrics attached and metrics.json exists"

# Check MlYamlS3Uri in CustomerMetadataProperties
ml_yaml_uri=$(echo "$model_details" | jq -r '.CustomerMetadataProperties.MlYamlS3Uri // empty')
[ -n "$ml_yaml_uri" ] || fail "CustomerMetadataProperties.MlYamlS3Uri is empty"
aws s3api head-object --bucket "$ARTIFACT_BUCKET" --key "${ml_yaml_uri#s3://$ARTIFACT_BUCKET/}" >/dev/null 2>&1 || \
    fail "ml.yaml file not found at $ml_yaml_uri"
echo "✓ MlYamlS3Uri in CustomerMetadataProperties and file exists"

# Check CustomerMetadataProperties
customer_metadata=$(echo "$model_details" | jq '.CustomerMetadataProperties')
[ "$customer_metadata" != "null" ] || fail "CustomerMetadataProperties not found"

git_commit=$(echo "$customer_metadata" | jq -r '.GitCommit')
project_name=$(echo "$customer_metadata" | jq -r '.ProjectName')
[ "$git_commit" != "null" ] || fail "GitCommit not in CustomerMetadataProperties"
[ "$project_name" = "$PROJECT_NAME" ] || fail "ProjectName incorrect in CustomerMetadataProperties"
echo "✓ CustomerMetadataProperties correct (GitCommit, ProjectName, etc.)"
echo ""

echo "=== Test 7: Submit with failing quality gate ==="
cat > ml.yaml <<EOF
name: $PROJECT_NAME
team: smoke-test
framework: sklearn
instance_type: ml.m5.large

data:
  train: s3://$ARTIFACT_BUCKET/smoke-test/iris/train/
  validation: s3://$ARTIFACT_BUCKET/smoke-test/iris/validation/

hyperparameters:
  max_depth: 5
  n_estimators: 100
  random_state: 42

quality_gate:
  metric: accuracy
  threshold: 1.01
  direction: maximize
EOF

submit_output_fail=$(mlctl submit --skip-validation --output json) || fail "Submit failed"
EXECUTION_ARN_FAIL=$(echo "$submit_output_fail" | jq -r '.execution_arn') || fail "Failed to parse execution ARN"
[ -n "$EXECUTION_ARN_FAIL" ] || fail "Empty execution ARN for failing case"
CREATED_EXECUTIONS+=("$EXECUTION_ARN_FAIL")

# Record execution-specific pipeline output prefix for failed case
EXEC_ID_FAIL="${EXECUTION_ARN_FAIL##*/}"
CREATED_S3_PREFIXES+=("s3://$ARTIFACT_BUCKET/pipelines/$EXEC_ID_FAIL/")

echo "✓ Pipeline submitted (expected to fail): $EXECUTION_ARN_FAIL"
echo ""

echo "=== Test 8: Wait for quality gate failure ==="
ELAPSED=0
CONSECUTIVE_ERRORS=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! STATUS=$(aws sagemaker describe-pipeline-execution \
        --pipeline-execution-arn "$EXECUTION_ARN_FAIL" \
        --query 'PipelineExecutionStatus' \
        --output text 2>/dev/null); then
        CONSECUTIVE_ERRORS=$((CONSECUTIVE_ERRORS + 1))
        echo "  Transient describe error ($CONSECUTIVE_ERRORS/$MAX_CONSECUTIVE_ERRORS), retrying..."
        if [ $CONSECUTIVE_ERRORS -ge $MAX_CONSECUTIVE_ERRORS ]; then
            fail "Too many consecutive describe errors"
        fi
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        continue
    fi
    
    CONSECUTIVE_ERRORS=0
    echo "  Status: $STATUS (waited ${ELAPSED}s)"
    
    if [ "$STATUS" = "Failed" ]; then
        # Check that QualityGateFailed step is what failed
        steps=$(aws sagemaker list-pipeline-execution-steps \
            --pipeline-execution-arn "$EXECUTION_ARN_FAIL" \
            --query 'PipelineExecutionSteps[?StepName==`QualityGateFailed`]' \
            --output json) || fail "Failed to list steps"
        
        [ "$steps" != "[]" ] || fail "QualityGateFailed step not found"
        
        step_status=$(echo "$steps" | jq -r '.[0].StepStatus')
        [ "$step_status" = "Failed" ] || fail "QualityGateFailed step status is $step_status, expected Failed"
        
        # Try FailureReason first, fall back to Metadata.Fail.ErrorMessage
        failure_reason=$(echo "$steps" | jq -r '.[0].FailureReason // empty')
        if [ -z "$failure_reason" ]; then
            failure_reason=$(echo "$steps" | jq -r '.[0].Metadata.Fail.ErrorMessage // empty')
        fi
        
        [ -n "$failure_reason" ] || fail "No FailureReason or Metadata.Fail.ErrorMessage found"
        [[ "$failure_reason" =~ accuracy ]] || fail "FailureReason doesn't contain 'accuracy'"
        [[ "$failure_reason" =~ 1\.01 ]] || fail "FailureReason doesn't contain threshold '1.01'"
        [[ "$failure_reason" =~ "Actual value: "[0-9] ]] || fail "FailureReason doesn't contain actual numeric value"
        
        echo "  Failure reason: $failure_reason"
        
        echo "✓ Quality gate failed as expected with correct error message"
        break
    elif [ "$STATUS" = "Succeeded" ]; then
        fail "Pipeline succeeded but should have failed quality gate"
    fi
    
    sleep $SLEEP_INTERVAL
    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))
done

[ $ELAPSED -lt $MAX_WAIT ] || fail "Pipeline did not complete within ${MAX_WAIT}s"
echo ""

echo "=== All smoke tests passed! ==="
