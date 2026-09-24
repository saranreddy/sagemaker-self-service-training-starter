#!/usr/bin/env bash
set -euo pipefail

echo "=== Pre-Destroy Cleanup ==="
echo "This script deletes SageMaker resources that Terraform doesn't manage."
echo "It only affects resources tagged with Project=sagemaker-self-service-training"
echo "or explicitly created by this project."
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Get region and project tag from terraform
cd "$ROOT_DIR/terraform"

if [ ! -f "terraform.tfstate" ]; then
    echo "No terraform.tfstate found. Skipping cleanup."
    exit 0
fi

REGION=$(terraform output -raw region 2>/dev/null || echo "us-east-1")
PROJECT_TAG="sagemaker-self-service-training"

export AWS_DEFAULT_REGION="$REGION"

echo "Region: $REGION"
echo "Project Tag: $PROJECT_TAG"
echo ""

ERRORS=0

# Function to report operation status
report() {
    local status="$1"
    local message="$2"
    
    if [ "$status" = "success" ]; then
        echo "✓ $message"
    elif [ "$status" = "error" ]; then
        echo "✗ $message"
        ERRORS=$((ERRORS + 1))
    else
        echo "  $message"
    fi
}

# 1. Stop all running pipeline executions for project pipelines
echo "Step 1: Stopping running pipeline executions..."

pipelines=$(aws sagemaker list-pipelines \
    --query "PipelineSummaries[].PipelineName" \
    --output text 2>/dev/null || echo "")

for pipeline in $pipelines; do
    # Check if this pipeline has our project tag
    tags=$(aws sagemaker list-tags \
        --resource-arn "arn:aws:sagemaker:${REGION}:$(aws sts get-caller-identity --query Account --output text):pipeline/${pipeline}" \
        --query "Tags[?Key=='Project' && Value=='${PROJECT_TAG}']" \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$tags" ]; then
        report "info" "Checking pipeline: $pipeline"
        
        executions=$(aws sagemaker list-pipeline-executions \
            --pipeline-name "$pipeline" \
            --query "PipelineExecutionSummaries[?PipelineExecutionStatus=='Executing'].PipelineExecutionArn" \
            --output text 2>/dev/null || echo "")
        
        for exec_arn in $executions; do
            if aws sagemaker stop-pipeline-execution --pipeline-execution-arn "$exec_arn" 2>/dev/null; then
                report "success" "Stopped execution: $exec_arn"
            else
                report "error" "Failed to stop execution: $exec_arn"
            fi
        done
    fi
done

# Wait a moment for executions to stop
sleep 5

# 2. Delete model packages in tagged model package groups
echo ""
echo "Step 2: Deleting model packages..."

model_groups=$(aws sagemaker list-model-package-groups \
    --query "ModelPackageGroupSummaryList[].ModelPackageGroupName" \
    --output text 2>/dev/null || echo "")

for group in $model_groups; do
    # Check if this group has our project tag
    tags=$(aws sagemaker list-tags \
        --resource-arn "arn:aws:sagemaker:${REGION}:$(aws sts get-caller-identity --query Account --output text):model-package-group/${group}" \
        --query "Tags[?Key=='Project' && Value=='${PROJECT_TAG}']" \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$tags" ]; then
        report "info" "Deleting packages in group: $group"
        
        packages=$(aws sagemaker list-model-packages \
            --model-package-group-name "$group" \
            --query "ModelPackageSummaryList[].ModelPackageArn" \
            --output text 2>/dev/null || echo "")
        
        for pkg_arn in $packages; do
            if aws sagemaker delete-model-package --model-package-name "$pkg_arn" 2>/dev/null; then
                report "success" "Deleted package: $(basename $pkg_arn)"
            else
                report "error" "Failed to delete package: $(basename $pkg_arn)"
            fi
        done
        
        # Delete the group itself
        if aws sagemaker delete-model-package-group --model-package-group-name "$group" 2>/dev/null; then
            report "success" "Deleted model package group: $group"
        else
            report "error" "Failed to delete model package group: $group"
        fi
    fi
done

# 3. Delete pipelines with project tag
echo ""
echo "Step 3: Deleting pipelines..."

pipelines=$(aws sagemaker list-pipelines \
    --query "PipelineSummaries[].PipelineName" \
    --output text 2>/dev/null || echo "")

for pipeline in $pipelines; do
    # Check if this pipeline has our project tag
    tags=$(aws sagemaker list-tags \
        --resource-arn "arn:aws:sagemaker:${REGION}:$(aws sts get-caller-identity --query Account --output text):pipeline/${pipeline}" \
        --query "Tags[?Key=='Project' && Value=='${PROJECT_TAG}']" \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$tags" ]; then
        if aws sagemaker delete-pipeline --pipeline-name "$pipeline" 2>/dev/null; then
            report "success" "Deleted pipeline: $pipeline"
        else
            report "error" "Failed to delete pipeline: $pipeline"
        fi
    fi
done

# 4. Note about log streams and training/processing jobs
echo ""
echo "Step 4: Training and processing job artifacts..."
report "info" "SageMaker training and processing job records cannot be deleted via API."
report "info" "Log streams in /aws/sagemaker/TrainingJobs and /aws/sagemaker/ProcessingJobs"
report "info" "are shared across the account and managed by CloudWatch log retention policies."
report "info" "The smoke test deletes its own log streams; manual cleanup is possible by prefix."

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "✓ Pre-destroy cleanup completed successfully"
    exit 0
else
    echo "✗ Pre-destroy cleanup completed with $ERRORS error(s)"
    exit 1
fi
