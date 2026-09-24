#!/usr/bin/env bash
set -e

echo "=== Environment Doctor ==="
echo ""

ERRORS=0

check_command() {
    local cmd=$1
    local name=$2
    local version_flag=${3:---version}
    
    if command -v "$cmd" &> /dev/null; then
        local version=$($cmd $version_flag 2>&1 | head -n 1)
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
if [ $ERRORS -eq 0 ]; then
    echo "✓ All checks passed"
    exit 0
else
    echo "✗ $ERRORS check(s) failed"
    exit 1
fi
