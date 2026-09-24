#!/usr/bin/env bash
set -e

echo "=== SageMaker Self-Service Training Smoke Test ==="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

source "$ROOT_DIR/venv/bin/activate"

if [ ! -f "$ROOT_DIR/venv/bin/mlctl" ]; then
    echo "Installing mlctl..."
    cd "$ROOT_DIR"
    pip install -q -e .
fi

cd "$ROOT_DIR/terraform"

if [ ! -f "terraform.tfstate" ]; then
    echo "✗ Terraform not deployed. Run 'make apply' first."
    exit 1
fi

EXECUTION_ROLE=$(terraform output -raw execution_role_arn)
ARTIFACT_BUCKET=$(terraform output -raw artifact_bucket_name)
REGION=$(terraform output -raw region)
ACCOUNT_ID=$(terraform output -raw account_id)

export MLCTL_ARTIFACT_BUCKET="$ARTIFACT_BUCKET"
export AWS_DEFAULT_REGION="$REGION"

echo "Configuration:"
echo "  Execution Role: $EXECUTION_ROLE"
echo "  Artifact Bucket: $ARTIFACT_BUCKET"
echo "  Region: $REGION"
echo "  Account ID: $ACCOUNT_ID"
echo ""

cat > "$ROOT_DIR/org-config.yaml" <<EOF
execution_role: $EXECUTION_ROLE
artifact_bucket: $ARTIFACT_BUCKET

frameworks:
  sklearn:
    container_uri_template: "{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    default_instance_type: "ml.m5.large"
    version: "1.2-1"
  xgboost:
    container_uri_template: "{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.7-1"
    default_instance_type: "ml.m5.large"
    version: "1.7-1"
  pytorch:
    container_uri_template: "{account}.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.1.0-cpu-py310"
    default_instance_type: "ml.m5.large"
    version: "2.1.0"

default_instance_type: "ml.m5.large"

allowed_instance_types:
  - ml.m5.large
  - ml.m5.xlarge
  - ml.m5.2xlarge

required_tags:
  Project: sagemaker-self-service-training

teams: {}
EOF

echo "✓ Created org-config.yaml"
echo ""

PROJECT_DIR="$ROOT_DIR/examples/sklearn-iris"
cd "$PROJECT_DIR"

echo "=== Test 1: Generate and upload data ==="
python3 generate_data.py

aws s3 sync data/train/ "s3://$ARTIFACT_BUCKET/smoke-test/iris/train/" --quiet
aws s3 sync data/validation/ "s3://$ARTIFACT_BUCKET/smoke-test/iris/validation/" --quiet

echo "✓ Data uploaded to S3"
echo ""

cat > ml.yaml <<EOF
name: smoke-test-iris
team: smoke-test
framework: sklearn
instance_type: ml.m5.large

data:
  train: s3://$ARTIFACT_BUCKET/smoke-test/iris/train/
  validation: s3://$ARTIFACT_BUCKET/smoke-test/iris/validation/

hyperparameters:
  max_depth: "5"
  n_estimators: "100"
  random_state: "42"

quality_gate:
  metric: accuracy
  threshold: 0.85
  direction: maximize
EOF

echo "=== Test 2: Validate project ==="
mlctl validate --offline
echo "✓ Validation passed"
echo ""

echo "=== Test 3: Run locally ==="
mlctl run --local
echo "✓ Local run passed"
echo ""

echo "=== Test 4: Submit to SageMaker (success case) ==="
mlctl submit --skip-validation > /tmp/smoke-submit.log 2>&1

EXECUTION_ARN=$(grep "Execution ARN" /tmp/smoke-submit.log | tail -1 | awk '{print $NF}')

if [ -z "$EXECUTION_ARN" ]; then
    echo "✗ Failed to extract execution ARN"
    cat /tmp/smoke-submit.log
    exit 1
fi

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
        --output text)
    
    echo "  Status: $STATUS (waited ${ELAPSED}s)"
    
    if [ "$STATUS" = "Succeeded" ]; then
        echo "✓ Pipeline succeeded"
        break
    elif [ "$STATUS" = "Failed" ] || [ "$STATUS" = "Stopped" ]; then
        echo "✗ Pipeline $STATUS"
        aws sagemaker describe-pipeline-execution --pipeline-execution-arn "$EXECUTION_ARN"
        exit 1
    fi
    
    sleep $SLEEP_INTERVAL
    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "✗ Pipeline did not complete within ${MAX_WAIT}s"
    exit 1
fi

echo ""

echo "=== Test 6: Verify model registration ==="
MODEL_PACKAGES=$(aws sagemaker list-model-packages \
    --model-package-group-name "smoke-test-iris-models" \
    --max-results 1 \
    --query 'ModelPackageSummaryList[0]' \
    --output json)

if [ "$MODEL_PACKAGES" = "null" ] || [ -z "$MODEL_PACKAGES" ]; then
    echo "✗ No model package found"
    exit 1
fi

MODEL_STATUS=$(echo "$MODEL_PACKAGES" | jq -r '.ModelApprovalStatus')

if [ "$MODEL_STATUS" != "PendingManualApproval" ]; then
    echo "✗ Model status is $MODEL_STATUS, expected PendingManualApproval"
    exit 1
fi

echo "✓ Model registered with status: $MODEL_STATUS"
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
  max_depth: "5"
  n_estimators: "100"
  random_state: "42"

quality_gate:
  metric: accuracy
  threshold: 0.99
  direction: maximize
EOF

mlctl submit --skip-validation > /tmp/smoke-submit-fail.log 2>&1

EXECUTION_ARN_FAIL=$(grep "Execution ARN" /tmp/smoke-submit-fail.log | tail -1 | awk '{print $NF}')

echo "✓ Pipeline submitted (expected to fail): $EXECUTION_ARN_FAIL"
echo ""

echo "=== Test 8: Wait for quality gate failure ==="
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS=$(aws sagemaker describe-pipeline-execution \
        --pipeline-execution-arn "$EXECUTION_ARN_FAIL" \
        --query 'PipelineExecutionStatus' \
        --output text)
    
    echo "  Status: $STATUS (waited ${ELAPSED}s)"
    
    if [ "$STATUS" = "Failed" ]; then
        STEPS=$(aws sagemaker list-pipeline-execution-steps \
            --pipeline-execution-arn "$EXECUTION_ARN_FAIL" \
            --query 'PipelineExecutionSteps[?StepName==`QualityGateFailed`]' \
            --output json)
        
        if [ "$STEPS" != "[]" ]; then
            echo "✓ Quality gate failed as expected"
            break
        else
            echo "✗ Failed but not at quality gate step"
            exit 1
        fi
    elif [ "$STATUS" = "Succeeded" ]; then
        echo "✗ Pipeline succeeded but should have failed quality gate"
        exit 1
    fi
    
    sleep $SLEEP_INTERVAL
    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "✗ Pipeline did not complete within ${MAX_WAIT}s"
    exit 1
fi

echo ""
echo "=== All smoke tests passed! ==="
