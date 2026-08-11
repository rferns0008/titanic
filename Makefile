SHELL := /bin/bash

.PHONY: help venv install train test

help:
	@echo "Available targets:"
	@echo "  make venv     # create virtualenv at .venv.nosync"
	@echo "  make install  # install dependencies into the venv"
	@echo "  make train    # run training (forwards ARGS to the script)"
	@echo "  make test     # run unit tests"

venv:
	python3 -m venv .venv.nosync

install: venv
	set -euo pipefail; \
	source .venv.nosync/bin/activate && python3 -m pip install --upgrade pip; \
	source .venv.nosync/bin/activate && python3 -m pip install -r requirements.txt; \
	source .venv.nosync/bin/activate && python3 -m pip install "dvc[s3]"

train:
	./scripts/run_train.sh $${ARGS:-}

test:
	set -euo pipefail; \
	source .venv.nosync/bin/activate && python3 -m pip install pytest || true; \
	source .venv.nosync/bin/activate && pytest src/tests
