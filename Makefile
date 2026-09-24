.PHONY: help doctor init apply smoke destroy clean test lint

help:
	@echo "Makefile targets:"
	@echo "  doctor    - Check prerequisites and environment"
	@echo "  init      - Initialize Terraform"
	@echo "  apply     - Apply Terraform infrastructure"
	@echo "  smoke     - Run smoke tests (requires deployed infrastructure)"
	@echo "  destroy   - Destroy Terraform infrastructure (runs pre-destroy cleanup)"
	@echo "  clean     - Clean build artifacts (preserves tfstate)"
	@echo "  test      - Run Python unit tests"
	@echo "  lint      - Run linters"

doctor:
	@scripts/doctor.sh

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

test:
	@echo "Running unit tests..."
	@python3 -m pytest tests/ -v

lint:
	@echo "Running flake8..."
	@python3 -m flake8 src/mlctl/ tests/
	@echo "Running black check..."
	@python3 -m black --check src/mlctl/ tests/
	@echo "Linting complete"
