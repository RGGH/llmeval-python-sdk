.PHONY: install install-dev test format lint clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

format:
	black llmeval examples tests

lint:
	mypy llmeval

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
