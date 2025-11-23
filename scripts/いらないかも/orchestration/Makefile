PY ?= python3
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
VENV_DIR ?= $(REPO_ROOT)/.venv

RUN_IN_ROOT = cd $(REPO_ROOT) &&

DATASETS_DIR ?= $(REPO_ROOT)/datasets
DERIVED_DIR  ?= $(DATASETS_DIR)/derived_artifacts

ENV_FILE ?= $(REPO_ROOT)/.env
ENV_FILE_FLAG := $(if $(wildcard $(ENV_FILE)),--env-file $(ENV_FILE),)

.PHONY: help setup download-srcmap download-coverage extract-metrics metrics-code-text metrics-patch coverage-aggregate aggregate-daily train rq3 selection-count docker-build docker-bash package

help:
	@echo "Targets:"
	@echo "  setup              - Create venv and install editable package"
	@echo "  download-srcmap    - Download srcmap JSONs (GCS)"
	@echo "  download-coverage  - Download coverage reports (GCS)"
	@echo "  extract-metrics    - Run metrics extraction pipeline"
	@echo "  metrics-code-text  - Merge code/text metrics"
	@echo "  metrics-patch      - Patch coverage pipeline"
	@echo "  coverage-aggregate - Aggregate coverage summaries"
	@echo "  aggregate-daily    - Build daily dataset from metrics+coverage"
	@echo "  train              - Train models (within-project)"
	@echo "  rq3                - Risk prediction (RQ3 prepare)"
	@echo "  selection-count    - Count candidate projects (mapping + cloned repos)"
	@echo "  docker-build       - Build Docker image"
	@echo "  docker-bash        - Run container shell with volumes mounted"
	@echo "  package            - Create zip artifact of repo"

setup:
	$(PY) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && pip install -U pip && pip install -e $(REPO_ROOT)

download-srcmap:
	$(RUN_IN_ROOT) $(PY) -m vuljit download-srcmap

download-coverage:
	$(RUN_IN_ROOT) $(PY) -m vuljit download-coverage

extract-metrics:
	$(RUN_IN_ROOT) $(PY) -m vuljit extract-metrics

metrics-code-text:
	$(RUN_IN_ROOT) $(PY) -m vuljit metrics code-text

metrics-patch:
	$(RUN_IN_ROOT) $(PY) -m vuljit metrics patch-coverage

coverage-aggregate:
	$(RUN_IN_ROOT) $(PY) -m vuljit metrics coverage-aggregate

aggregate-daily:
	$(RUN_IN_ROOT) $(PY) -m vuljit metrics aggregate-daily

train:
	$(RUN_IN_ROOT) $(PY) -m vuljit prediction train

rq3:
	$(RUN_IN_ROOT) $(PY) -m vuljit prediction rq3

selection-count:
	$(RUN_IN_ROOT) $(PY) -m vuljit mapping count-candidates

docker-build:
	docker build -t vuljit:latest -f $(REPO_ROOT)/scripts/orchestration/Dockerfile $(REPO_ROOT)

docker-bash:
	docker run --rm -it \
		$(ENV_FILE_FLAG) \
		-v $(DATASETS_DIR):/app/vuljit/datasets \
		-v $(DERIVED_DIR):/app/vuljit/datasets/derived_artifacts \
		-w /app/vuljit \
		vuljit:latest bash

package:
	@cd $(REPO_ROOT) && mkdir -p dist
	@cd $(REPO_ROOT) && zip -r dist/vuljit-replication.zip . -x "datasets/*" "**/__pycache__/*" "**/.venv/*" "**/.git/*" "dist/*"
	@echo "Created dist/vuljit-replication.zip"
