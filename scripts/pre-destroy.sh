#!/usr/bin/env bash
set -e

echo "=== Pre-Destroy Cleanup ==="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR/terraform"

if [ ! -f "terraform.tfstate" ]; then
    echo "No terraform state found, nothing to clean up"
    exit 0
fi

REGION=$(terraform output -raw region 2>/dev/null || echo "us-east-1")
ARTIFACT_BUCKET=$(terraform output -raw artifact_bucket_name 2>/dev/null || echo "")

export AWS_DEFAULT_REGION="$REGION"

echo "Region: $REGION"
echo ""

echo "=== Cleaning up SageMaker pipelines ==="
PIPELINES=$(aws sagemaker list-pipelines --query 'PipelineSummaries[].PipelineName' --output text 2>/dev/null || echo "")

if [ -n "$PIPELINES" ]; then
    for PIPELINE in $PIPELINES; do
        if [[ "$PIPELINE" == *"-pipeline" ]]; then
            echo "  Deleting pipeline: $PIPELINE"
            aws sagemaker delete-pipeline --pipeline-name "$PIPELINE" 2>/dev/null || true
        fi
    done
    echo "✓ Pipelines deleted"
else
    echo "  No pipelines found"
fi

echo ""

echo "=== Cleaning up model packages and groups ==="
MODEL_GROUPS=$(aws sagemaker list-model-package-groups --query 'ModelPackageGroupSummaryList[].ModelPackageGroupName' --output text 2>/dev/null || echo "")

if [ -n "$MODEL_GROUPS" ]; then
    for GROUP in $MODEL_GROUPS; do
        if [[ "$GROUP" == *"-models" ]] || [[ "$GROUP" == smoke-test* ]]; then
            echo "  Deleting model packages in group: $GROUP"
            
            MODEL_ARNS=$(aws sagemaker list-model-packages \
                --model-package-group-name "$GROUP" \
                --query 'ModelPackageSummaryList[].ModelPackageArn' \
                --output text 2>/dev/null || echo "")
            
            for ARN in $MODEL_ARNS; do
                echo "    Deleting model package: $ARN"
                aws sagemaker delete-model-package --model-package-name "$ARN" 2>/dev/null || true
            done
            
            echo "  Deleting model package group: $GROUP"
            aws sagemaker delete-model-package-group --model-package-group-name "$GROUP" 2>/dev/null || true
        fi
    done
    echo "✓ Model packages and groups deleted"
else
    echo "  No model package groups found"
fi

echo ""

echo "=== Cleaning up S3 objects ==="
if [ -n "$ARTIFACT_BUCKET" ]; then
    echo "  Emptying bucket: $ARTIFACT_BUCKET"
    aws s3 rm "s3://$ARTIFACT_BUCKET/" --recursive 2>/dev/null || true
    echo "✓ S3 bucket emptied"
else
    echo "  No artifact bucket configured"
fi

echo ""

echo "=== Cleaning up CloudWatch log streams ==="
for LOG_GROUP in "/aws/sagemaker/TrainingJobs" "/aws/sagemaker/ProcessingJobs"; do
    echo "  Cleaning log group: $LOG_GROUP"
    
    LOG_STREAMS=$(aws logs describe-log-streams \
        --log-group-name "$LOG_GROUP" \
        --query 'logStreams[?starts_with(logStreamName, `smoke-test`)].logStreamName' \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$LOG_STREAMS" ]; then
        for STREAM in $LOG_STREAMS; do
            echo "    Deleting log stream: $STREAM"
            aws logs delete-log-stream --log-group-name "$LOG_GROUP" --log-stream-name "$STREAM" 2>/dev/null || true
        done
    fi
done

echo "✓ Log streams cleaned"
echo ""

echo "=== Pre-destroy cleanup complete ==="
echo ""
echo "IMPORTANT: SageMaker training/processing job records cannot be deleted."
echo "They will remain visible in the SageMaker console but do not incur charges."
echo ""
