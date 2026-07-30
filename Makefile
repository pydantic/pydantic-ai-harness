.DEFAULT_GOAL := all

.PHONY: .uv .prek install format lint typecheck test testcov integration-belgie integration-localstack integration-mongodb all

.uv:
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.prek:
	@prek --version || echo 'Please install prek: https://github.com/j178/pre-commit-rs'

install: .uv .prek
	uv sync --frozen --all-extras --group lint
	prek install --install-hooks

format:
	uv run ruff format
	uv run ruff check --fix --fix-only

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

integration-belgie:
	uv run --no-sync pytest -m belgie_live tests/belgie_sandbox/test_belgie_live.py

integration-localstack:
	uv run pytest integration_tests/localstack/test_live_localstack.py

# Needs a reachable mongod (`docker run -d -p 27017:27017 mongo:8`); without one
# the tests skip. Set MONGODB_TEST_URL to point at a server elsewhere.
integration-mongodb:
	uv run pytest integration_tests/mongodb/test_live_mongodb.py

all: format lint typecheck testcov
