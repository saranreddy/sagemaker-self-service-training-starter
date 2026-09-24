.PHONY: help doctor init apply smoke destroy clean test lint venv

help:
	@echo "Makefile targets:"
	@echo "  doctor    - Check prerequisites and environment"
	@echo "  venv      - Create Python virtual environment with dependencies"
	@echo "  init      - Initialize Terraform"
	@echo "  apply     - Apply Terraform infrastructure"
	@echo "  smoke     - Run smoke tests (requires deployed infrastructure)"
	@echo "  destroy   - Destroy Terraform infrastructure (runs pre-destroy cleanup)"
	@echo "  clean     - Clean build artifacts (preserves tfstate)"
	@echo "  test      - Run Python unit tests"
	@echo "  lint      - Run linters"

doctor:
	@scripts/doctor.sh

venv:
	@echo "Creating Python virtual environment..."
	@python3 -m venv venv
	@./venv/bin/pip install -q --upgrade pip
	@./venv/bin/pip install -q -e .
	@./venv/bin/pip install -q -r requirements-dev.txt
	@echo "✓ Virtual environment created at ./venv"
	@echo "  Activate with: source venv/bin/activate"

init:
	@cd terraform && terraform init

apply:
	@cd terraform && terraform apply

smoke: doctor
	@scripts/smoke-test.sh

destroy:
	@scripts/pre-destroy.sh
	@cd terraform && terraform destroy

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	@rm -rf src/mlctl.egg-info/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf examples/*/data examples/*/local_output
	@echo "Clean complete (tfstate preserved)"

test: venv
	@echo "Running unit tests..."
	@./venv/bin/pytest tests/ -v

lint: venv
	@echo "Running flake8..."
	@./venv/bin/flake8 src/mlctl/ tests/ --max-line-length=100 --extend-ignore=E203,W503 || true
	@echo "Running black check..."
	@./venv/bin/black --check src/mlctl/ tests/
	@echo "Linting complete"
