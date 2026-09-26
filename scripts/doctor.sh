#!/usr/bin/env bash
set -e

echo "=== Environment Doctor ==="
echo ""

ERRORS=0

check_command() {
    local cmd=$1
    local name=$2
    local version_flag=${3:---version}
    local version
    
    if command -v "$cmd" &> /dev/null; then
        version=$($cmd $version_flag 2>&1 | head -n 1)
        echo "✓ $name: $version"
    else
        echo "✗ $name: NOT FOUND"
        ERRORS=$((ERRORS + 1))
    fi
}

check_command "terraform" "Terraform"
check_command "aws" "AWS CLI"
check_command "python3" "Python" "--version"
check_command "jq" "jq"
check_command "make" "GNU Make" "--version"
check_command "git" "Git" "--version"

echo ""
echo "=== AWS Configuration ==="
if aws sts get-caller-identity &> /dev/null; then
    aws sts get-caller-identity | jq -r '"✓ AWS credentials valid (\(.UserId))"'
else
    echo "✗ AWS credentials not configured or invalid"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "=== Python Virtual Environment ==="
if [ -d "venv" ]; then
    echo "✓ Virtual environment exists"
else
    echo "! Virtual environment not found (will create one)"
    python3 -m venv venv
    echo "✓ Created virtual environment"
fi

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Activated virtual environment"
else
    echo "✗ Failed to activate virtual environment"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "=== Python Dependencies ==="
if python3 -m pip show boto3 &> /dev/null; then
    echo "✓ boto3 installed"
else
    echo "! boto3 not installed (installing...)"
    python3 -m pip install -q -r requirements.txt
    echo "✓ Installed dependencies"
fi

echo ""

# Check SageMaker service quotas (warning only, doesn't fail)
echo "=== SageMaker Service Quotas ==="

QUOTA_WARNINGS=0

# Get region
REGION=$(aws configure get region 2>/dev/null || echo "us-east-1")

# Check if service-quotas is available
if aws service-quotas list-service-quotas --service-code sagemaker --region "$REGION" --max-items 1 >/dev/null 2>&1; then
    # Instance types and job types to check
    # Format: "instance_type:job_type"
    declare -a QUOTAS_TO_CHECK=(
        "ml.m5.large:training"
        "ml.m5.large:processing"
    )
    
    for quota_spec in "${QUOTAS_TO_CHECK[@]}"; do
        IFS=':' read -r instance_type job_type <<< "$quota_spec"
        
        quota_name="${instance_type} for ${job_type} job usage"
        
        # Look up quota by exact name
        quota_value=$(aws service-quotas list-service-quotas \
            --service-code sagemaker \
            --region "$REGION" \
            --query "Quotas[?QuotaName=='${quota_name}'].Value | [0]" \
            --output text 2>/dev/null || echo "")
        
        if [ -n "$quota_value" ] && [ "$quota_value" != "None" ]; then
            # Compare as integers (bash 3.2 compatible)
            quota_int=$(printf "%.0f" "$quota_value" 2>/dev/null)
            if [ $? -eq 0 ] && [ "$quota_int" -eq 0 ]; then
                echo "⚠️  Warning: SageMaker quota for '${quota_name}' is 0 in region $REGION"
                echo "   Pipelines using this instance type will fail until you request a quota increase."
                echo "   Request via: https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas"
                QUOTA_WARNINGS=$((QUOTA_WARNINGS + 1))
            fi
        fi
    done
    
    if [ $QUOTA_WARNINGS -eq 0 ]; then
        echo "✓ Default instance type quotas are non-zero"
    fi
else
    echo "⚠️  Note: Unable to check service quotas (access may be restricted or AWS CLI too old)"
fi

echo ""

if [ $ERRORS -eq 0 ]; then
    if [ $QUOTA_WARNINGS -gt 0 ]; then
        echo "✓ All checks passed (with $QUOTA_WARNINGS quota warning(s))"
    else
        echo "✓ All checks passed"
    fi
    exit 0
else
    echo "✗ $ERRORS check(s) failed"
    exit 1
fi
