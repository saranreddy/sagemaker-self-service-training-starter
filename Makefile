.PHONY: help doctor init apply smoke destroy clean test lint venv

help:
	@echo "Makefile targets:"
	@echo "  doctor    - Check prerequisites and environment"
	@echo "  venv      - Create Python virtual environment with dependencies"
	@echo "  init      - Initialize Terraform"
	@echo "  apply     - Apply Terraform infrastructure (set AUTO_APPROVE=1 to skip confirmation)"
	@echo "  smoke     - Run smoke tests (requires deployed infrastructure)"
	@echo "  destroy   - Destroy Terraform infrastructure (set AUTO_APPROVE=1 to skip confirmation)"
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
	@if [ ! -f terraform/terraform.tfvars ]; then \
		echo "Creating terraform/terraform.tfvars from terraform.tfvars.example..."; \
		cp terraform/terraform.tfvars.example terraform/terraform.tfvars; \
		echo "✓ Created terraform.tfvars. Edit it if needed, then run 'make apply' again."; \
		exit 1; \
	fi
	@if [ "$(AUTO_APPROVE)" = "1" ]; then \
		cd terraform && terraform apply -auto-approve; \
	else \
		cd terraform && terraform apply; \
	fi

smoke: doctor
	@scripts/smoke-test.sh

destroy:
	@echo "Running pre-destroy cleanup..."
	@if ! scripts/pre-destroy.sh; then \
		echo ""; \
		echo "WARNING: pre-destroy.sh failed. This is usually safe to ignore if resources are already deleted."; \
		echo "Continuing with terraform destroy..."; \
		echo ""; \
	fi
	@if [ "$(AUTO_APPROVE)" = "1" ]; then \
		cd terraform && terraform destroy -auto-approve; \
	else \
		cd terraform && terraform destroy; \
	fi

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	@rm -rf src/mlctl.egg-info/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf examples/*/data examples/*/local_output
	@echo "Clean complete (tfstate preserved)"

test: venv
	@echo "Running unit tests..."
	@./venv/bin/pytest tests/ -v

lint:
	@echo "Running flake8..."
	@if [ -d venv ] && [ -f ./venv/bin/flake8 ]; then \
		./venv/bin/flake8 src tests; \
	else \
		python3 -m flake8 src tests; \
	fi
	@echo "Running black check..."
	@if [ -d venv ] && [ -f ./venv/bin/black ]; then \
		./venv/bin/black --check src tests; \
	else \
		python3 -m black --check src tests; \
	fi
	@echo "Linting complete"
