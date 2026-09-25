#!/usr/bin/env bash
set -euo pipefail

echo "=== Pre-Destroy Cleanup ==="
echo "This script deletes SageMaker resources that Terraform doesn't manage."
echo "It only affects resources tagged with the deployment-scoped tag from org-config."
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Get region and deployment tag from terraform
cd "$ROOT_DIR/terraform"

if [ ! -f "terraform.tfstate" ]; then
    echo "No terraform.tfstate found. Skipping cleanup."
    exit 0
fi

REGION=$(terraform output -raw region 2>/dev/null || echo "us-east-1")
if ! DEPLOYMENT_TAG=$(terraform output -raw deployment_tag 2>/dev/null) || [ -z "$DEPLOYMENT_TAG" ]; then
    echo "Warning: terraform output deployment_tag failed, using default" >&2
    DEPLOYMENT_TAG="mlctl:deployment=sagemaker-self-service-training"
fi

export AWS_DEFAULT_REGION="$REGION"

# Parse tag key and value
TAG_KEY=$(echo "$DEPLOYMENT_TAG" | cut -d= -f1)
TAG_VALUE=$(echo "$DEPLOYMENT_TAG" | cut -d= -f2-)

# Cache account ID
if ! ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || [ -z "$ACCOUNT_ID" ]; then
    echo "Error: cannot get AWS account ID. Check credentials." >&2
    exit 1
fi

echo "Region: $REGION"
echo "Deployment Tag: $TAG_KEY=$TAG_VALUE"
echo "Account ID: $ACCOUNT_ID"
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

# Function to check if resource has deployment tag
has_deployment_tag() {
    local resource_arn="$1"
    local tags
    tags=$(aws sagemaker list-tags \
        --resource-arn "$resource_arn" \
        --query "Tags[?Key=='$TAG_KEY' && Value=='$TAG_VALUE']" \
        --output text 2>/dev/null || echo "")
    [ -n "$tags" ]
}

# 1. Collect pipeline names and their execution job names BEFORE stopping/deleting
echo "Step 1: Collecting tagged pipelines and their job names..."

TAGGED_PIPELINES=()
ALL_JOB_NAMES=()

pipelines=$(aws sagemaker list-pipelines \
    --query "PipelineSummaries[].PipelineName" \
    --output text 2>/dev/null || echo "")

for pipeline in $pipelines; do
    pipeline_arn="arn:aws:sagemaker:${REGION}:${ACCOUNT_ID}:pipeline/${pipeline}"
    
    if has_deployment_tag "$pipeline_arn"; then
        report "info" "Found tagged pipeline: $pipeline"
        TAGGED_PIPELINES+=("$pipeline")
        
        # Collect executions for this pipeline
        executions=$(aws sagemaker list-pipeline-executions \
            --pipeline-name "$pipeline" \
            --query "PipelineExecutionSummaries[].PipelineExecutionArn" \
            --output text 2>/dev/null || echo "")
        
        for exec_arn in $executions; do
            # Collect job names from this execution
            steps=$(aws sagemaker list-pipeline-execution-steps \
                --pipeline-execution-arn "$exec_arn" \
                --query 'PipelineExecutionSteps[].Metadata' \
                --output json 2>/dev/null || echo "[]")
            
            # Extract training job names
            training_jobs=$(echo "$steps" | jq -r '.[] | select(.TrainingJob != null) | .TrainingJob.Arn' | sed 's|.*/||' || true)
            for job in $training_jobs; do
                [ -n "$job" ] && ALL_JOB_NAMES+=("$job")
            done
            
            # Extract processing job names
            processing_jobs=$(echo "$steps" | jq -r '.[] | select(.ProcessingJob != null) | .ProcessingJob.Arn' | sed 's|.*/||' || true)
            for job in $processing_jobs; do
                [ -n "$job" ] && ALL_JOB_NAMES+=("$job")
            done
        done
    fi
done

echo "Found ${#TAGGED_PIPELINES[@]} tagged pipelines, ${#ALL_JOB_NAMES[@]} jobs"
echo ""

# 2. Stop running executions for tagged pipelines and wait for terminal state
if [ ${#TAGGED_PIPELINES[@]} -gt 0 ]; then
    echo "Step 2: Stopping running executions..."
    
    for pipeline in "${TAGGED_PIPELINES[@]}"; do
        executions=$(aws sagemaker list-pipeline-executions \
            --pipeline-name "$pipeline" \
            --query "PipelineExecutionSummaries[?PipelineExecutionStatus=='Executing'].PipelineExecutionArn" \
            --output text 2>/dev/null || echo "")
        
        for exec_arn in $executions; do
            if aws sagemaker stop-pipeline-execution --pipeline-execution-arn "$exec_arn" 2>/dev/null; then
                report "success" "Stopped execution: $(basename "$exec_arn")"
            else
                report "error" "Failed to stop execution: $(basename "$exec_arn")"
            fi
        done
    done
    
    # Wait for stopped executions to reach terminal state
    echo "Waiting for stopped executions to reach terminal state..."
    for pipeline in "${TAGGED_PIPELINES[@]}"; do
        executions=$(aws sagemaker list-pipeline-executions \
            --pipeline-name "$pipeline" \
            --query "PipelineExecutionSummaries[?PipelineExecutionStatus=='Stopping'].PipelineExecutionArn" \
            --output text 2>/dev/null || echo "")
        
        for exec_arn in $executions; do
            max_wait=60
            waited=0
            while [ $waited -lt $max_wait ]; do
                status=$(aws sagemaker describe-pipeline-execution \
                    --pipeline-execution-arn "$exec_arn" \
                    --query 'PipelineExecutionStatus' \
                    --output text 2>/dev/null || echo "UNKNOWN")
                
                if [[ "$status" =~ ^(Succeeded|Failed|Stopped)$ ]]; then
                    break
                fi
                
                sleep 2
                waited=$((waited + 2))
            done
            
            # Report if execution didn't reach terminal state
            if [ $waited -ge $max_wait ]; then
                echo "Error: Execution $exec_arn did not reach terminal state after ${max_wait}s" >&2
            fi
        done
    done
    
    echo ""
fi

# 3. Delete model packages in tagged model package groups
echo "Step 3: Deleting model packages in tagged groups..."

model_groups=$(aws sagemaker list-model-package-groups \
    --query "ModelPackageGroupSummaryList[].ModelPackageGroupName" \
    --output text 2>/dev/null || echo "")

for group in $model_groups; do
    group_arn="arn:aws:sagemaker:${REGION}:${ACCOUNT_ID}:model-package-group/${group}"
    
    if has_deployment_tag "$group_arn"; then
        report "info" "Deleting packages in group: $group"
        
        packages=$(aws sagemaker list-model-packages \
            --model-package-group-name "$group" \
            --query "ModelPackageSummaryList[].ModelPackageArn" \
            --output text 2>/dev/null || echo "")
        
        for pkg_arn in $packages; do
            if aws sagemaker delete-model-package --model-package-name "$pkg_arn" 2>/dev/null; then
                report "success" "Deleted package: $(basename "$pkg_arn")"
            else
                report "error" "Failed to delete package: $(basename "$pkg_arn")"
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

echo ""

# 4. Delete tagged pipelines
if [ ${#TAGGED_PIPELINES[@]} -gt 0 ]; then
    echo "Step 4: Deleting pipelines..."
    
    for pipeline in "${TAGGED_PIPELINES[@]}"; do
        if aws sagemaker delete-pipeline --pipeline-name "$pipeline" 2>/dev/null; then
            report "success" "Deleted pipeline: $pipeline"
        else
            report "error" "Failed to delete pipeline: $pipeline"
        fi
    done
    
    echo ""
fi

# 5. Delete CloudWatch log streams for collected jobs
if [ ${#ALL_JOB_NAMES[@]} -gt 0 ]; then
    echo "Step 5: Deleting CloudWatch log streams for jobs..."
    
    for job_name in "${ALL_JOB_NAMES[@]}"; do
        for log_group in "/aws/sagemaker/TrainingJobs" "/aws/sagemaker/ProcessingJobs"; do
            streams=$(aws logs describe-log-streams \
                --log-group-name "$log_group" \
                --log-stream-name-prefix "$job_name" \
                --query 'logStreams[].logStreamName' \
                --output text 2>/dev/null || true)
            
            for stream in $streams; do
                if [ -n "$stream" ]; then
                    if aws logs delete-log-stream \
                        --log-group-name "$log_group" \
                        --log-stream-name "$stream" 2>/dev/null; then
                        report "success" "Deleted log stream: $log_group/$stream"
                    else
                        report "error" "Failed to delete log stream: $log_group/$stream"
                    fi
                fi
            done
        done
    done
    
    echo ""
fi

# 6. Note about training/processing job records
echo "Step 6: Training and processing job artifacts..."
report "info" "SageMaker training and processing job records cannot be deleted via API."
report "info" "They remain visible in the console history but do not incur charges."

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "✓ Pre-destroy cleanup completed successfully"
    exit 0
else
    echo "✗ Pre-destroy cleanup completed with $ERRORS error(s)"
    exit 1
fi
