#!/usr/bin/env bash
set -euo pipefail

echo "=== SageMaker Self-Service Training Smoke Test ==="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Track resources for cleanup
declare -a CREATED_PIPELINES
declare -a CREATED_EXECUTIONS
declare -a CREATED_MODEL_GROUPS
declare -a CREATED_S3_PREFIXES

CLEANUP_DONE=false

cleanup() {
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi
    
    echo ""
    echo "=== Cleanup ==="
    
    # Stop running executions
    for exec_arn in "${CREATED_EXECUTIONS[@]}"; do
        echo "Stopping execution: $exec_arn"
        aws sagemaker stop-pipeline-execution --pipeline-execution-arn "$exec_arn" 2>/dev/null || true
    done
    
    # Collect job names from pipeline executions before deleting
    declare -a JOB_NAMES
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
    
    # Delete model packages
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
    
    # Delete pipelines
    for pipeline in "${CREATED_PIPELINES[@]}"; do
        echo "Deleting pipeline: $pipeline"
        aws sagemaker delete-pipeline --pipeline-name "$pipeline" 2>/dev/null || true
    done
    
    # Delete log streams
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
    
    # Delete S3 prefixes
    for prefix in "${CREATED_S3_PREFIXES[@]}"; do
        echo "Deleting S3 prefix: $prefix"
        aws s3 rm "$prefix" --recursive 2>/dev/null || true
    done
    
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
pip install -q scikit-learn==1.2.2 numpy==1.24.3 xgboost==1.7.6 || fail "Failed to install example deps"

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
trap "rm -rf $WORK_DIR" EXIT

cp -r "$ROOT_DIR/examples/sklearn-iris"/* "$WORK_DIR/"
cd "$WORK_DIR"

# Create org-config.yaml in temp dir
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

teams: {}
EOF

export HOME="$WORK_DIR"  # Make mlctl look for config here

echo "=== Test 1: Generate and upload data ==="
python3 generate_data.py || fail "Data generation failed"

CREATED_S3_PREFIXES+=("s3://$ARTIFACT_BUCKET/smoke-test/")

aws s3 sync data/train/ "s3://$ARTIFACT_BUCKET/smoke-test/iris/train/" --quiet || fail "S3 upload failed"
aws s3 sync data/validation/ "s3://$ARTIFACT_BUCKET/smoke-test/iris/validation/" --quiet || fail "S3 upload failed"

echo "✓ Data uploaded"
echo ""

# Create ml.yaml with passing threshold (with margin)
cat > ml.yaml <<EOF
name: smoke-test-iris
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

CREATED_PIPELINES+=("smoke-test-iris-pipeline")
CREATED_MODEL_GROUPS+=("smoke-test-iris-models")

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

echo "✓ Pipeline submitted: $EXECUTION_ARN"
echo ""

echo "=== Test 5: Wait for pipeline completion ==="
MAX_WAIT=1200
ELAPSED=0
SLEEP_INTERVAL=30

while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS=$(aws sagemaker describe-pipeline-execution \
        --pipeline-execution-arn "$EXECUTION_ARN" \
        --query 'PipelineExecutionStatus' \
        --output text 2>&1) || {
        echo "  Transient describe error, retrying..."
        sleep 5
        continue
    }
    
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
    --model-package-group-name "smoke-test-iris-models" \
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

# Check ModelMetrics exists
model_metrics=$(echo "$model_details" | jq '.ModelMetrics')
[ "$model_metrics" != "null" ] || fail "ModelMetrics not found"
echo "✓ ModelMetrics attached"

# Check CustomerMetadataProperties
customer_metadata=$(echo "$model_details" | jq '.CustomerMetadataProperties')
[ "$customer_metadata" != "null" ] || fail "CustomerMetadataProperties not found"

git_commit=$(echo "$customer_metadata" | jq -r '.GitCommit')
project_name=$(echo "$customer_metadata" | jq -r '.ProjectName')
[ "$git_commit" != "null" ] || fail "GitCommit not in CustomerMetadataProperties"
[ "$project_name" = "smoke-test-iris" ] || fail "ProjectName incorrect in CustomerMetadataProperties"
echo "✓ CustomerMetadataProperties correct (GitCommit, ProjectName, etc.)"
echo ""

echo "=== Test 7: Submit with failing quality gate ==="
cat > ml.yaml <<EOF
name: smoke-test-iris
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

echo "✓ Pipeline submitted (expected to fail): $EXECUTION_ARN_FAIL"
echo ""

echo "=== Test 8: Wait for quality gate failure ==="
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS=$(aws sagemaker describe-pipeline-execution \
        --pipeline-execution-arn "$EXECUTION_ARN_FAIL" \
        --query 'PipelineExecutionStatus' \
        --output text 2>&1) || {
        echo "  Transient describe error, retrying..."
        sleep 5
        continue
    }
    
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
        
        failure_reason=$(echo "$steps" | jq -r '.[0].FailureReason // empty')
        [[ "$failure_reason" =~ "accuracy" ]] || fail "FailureReason doesn't contain 'accuracy'"
        [[ "$failure_reason" =~ "1.01" ]] || fail "FailureReason doesn't contain threshold '1.01'"
        
        # Extract actual value from message (it should be present due to Std:Join with Std:JsonGet)
        # The message format is: "Quality gate failed. Metric 'accuracy' did not meet threshold 1.01 (direction: maximize). Actual value: X.XX"
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
