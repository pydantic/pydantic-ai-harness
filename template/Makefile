.DEFAULT_GOAL := help

.PHONY: help install format lint typecheck test testcov

help:
	@echo "Available targets:"
	@echo "  install    Install dependencies"
	@echo "  format     Run ruff format"
	@echo "  lint       Run ruff check"
	@echo "  typecheck  Run pyright"
	@echo "  test       Run pytest"
	@echo "  testcov    Run pytest with coverage"

install:
	uv sync --all-groups

format:
	uv run ruff format
	uv run ruff check --fix

lint:
	uv run ruff format --check
	uv run ruff check

typecheck:
	uv run pyright

test:
	uv run pytest

testcov:
	uv run coverage run -m pytest
	uv run coverage report
