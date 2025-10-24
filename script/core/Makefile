PY ?= python3
PIP ?= pip3

# Default dirs (overridden by .env or environment)
DATA_DIR ?= $(PWD)/vuljit/data
OUT_DIR  ?= $(PWD)/vuljit/outputs

.PHONY: help setup download-srcmap download-coverage extract-metrics docker-build docker-bash package

help:
	@echo "Targets:"
	@echo "  setup              - Create venv and install requirements"
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
	$(PY) -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r vuljit/requirements.txt

download-srcmap:
	$(PY) -m vuljit download-srcmap

download-coverage:
	$(PY) -m vuljit download-coverage

extract-metrics:
	$(PY) -m vuljit extract-metrics

metrics-code-text:
	$(PY) -m vuljit metrics code-text

metrics-patch:
	$(PY) -m vuljit metrics patch-coverage --clone-repos

coverage-aggregate:
	$(PY) -m vuljit metrics coverage-aggregate

aggregate-daily:
	$(PY) -m vuljit metrics aggregate-daily

train:
	$(PY) -m vuljit prediction train

rq3:
	$(PY) -m vuljit prediction rq3

selection-count:
	$(PY) -m vuljit mapping count-candidates

docker-build:
	docker build -t vuljit:latest -f vuljit/Dockerfile .

docker-bash:
	docker run --rm -it \
		--env-file vuljit/.env \
		-v $(DATA_DIR):/app/vuljit/data \
		-v $(OUT_DIR):/app/vuljit/outputs \
		-w /app/vuljit \
		vuljit:latest bash

package:
	@mkdir -p dist
	@zip -r dist/vuljit-replication.zip vuljit -x "**/__pycache__/*" -x "**/.venv/*" -x "**/.git/*"
	@echo "Created dist/vuljit-replication.zip"
