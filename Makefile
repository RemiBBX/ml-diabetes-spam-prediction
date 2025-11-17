# Makefile pour projet Python utilisant uv et ruff

.PHONY: install lint format check run test clean

default: format lint
# Installation des dépendances via uv
install:
	uv sync

# Lint avec ruff
lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

# Lint + formatage

test:
	uv run pytest


