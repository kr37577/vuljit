# Replication Package for the Risk-Aware Intensive Supplemental Execution Framework

This repository provides the replication package for our research paper. It contains the datasets, scripts, and instructions necessary to reproduce the experimental results reported in the paper.

> **Paper**: *[Raise or Check? Predicting the Necessity of
Additional Fuzzing for Continuous Fuzzing]*  
> **Authors**: [Riku Kato, Tatsuya Shirai, Olivier Nourry, Yutaro Kashiwa, Kenji Fujiwara, Yasutaka Kamei, and Hajimu Iida]  
> **DOI / Preprint**: [Link]

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Repository Structure](#repository-structure)
5. [Reproducing the Experiments](#reproducing-the-experiments)
6. [Expected Results](#expected-results)
7. [License](#license)

---

## Overview

<!-- Brief (2-3 sentences) description of what the paper studies and what this package reproduces. -->

This package allows you to:

- Prepare the dataset used in our study
- Extract and aggregate software metrics
- Train cross-project prediction models
- Generate the results for RQ1, RQ2, and RQ3

---

## Requirements

| Item | Version / Note |
|------|----------------|
| **OS** | Linux (tested on Ubuntu 22.04) |
| **Python** | 3.12.3 |
| **Disk Space** | ~3 TB |
| **Runtime** | Several days to weeks (SLURM cluster recommended) |
| **Chrome / Chromium** | Required for `osv_monorail_selenium.py` |
| **ChromeDriver** | Must match your Chrome version |

All Python dependencies are listed in `requirements.txt`.

### Note on Selenium-based Scripts

The script `scripts/data_acquisition/osv_monorail_selenium.py` requires a GUI browser (Chrome/Chromium) and is **recommended to run on a local PC** rather than a headless server or cluster environment.

If you must run it on a server, ensure:
- Chrome is installed with headless mode support
- ChromeDriver version matches the installed Chrome
- X virtual framebuffer (Xvfb) is available if needed

## Setup

```bash
# 1. Verify Python version
python3 --version   # should be 3.12.x

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### External Data

We use [OSS-Fuzz](https://github.com/google/oss-fuzz). The scripts automatically clone the repository at the following commit:

```
96cf5fd552e90b1879d5d3c9bf6d9cdb95e6f122
```

---

## Docker Replication

This repository includes a Docker-based replication environment with `docker-compose`.
The container runs from `/workspace/RAISE` and is designed to execute the same pipeline as `run_all_process.sh`.

### 1) Prepare environment files

```bash
cp .env.docker.example .env.docker
```

Optionally keep your existing `.env` for project variables; the container entrypoint loads both `.env` and `.env.docker` if present.

### 2) Build image

```bash
docker compose build
```

### 3) Run full pipeline

```bash
docker compose run --rm replication run_all
```

### 4) Run a specific replication step

```bash
docker compose run --rm replication run_step RQ3
docker compose run --rm replication run_step RQ3 --prepare-only
docker compose run --rm replication run_step RQ3 --simulate-only
docker compose run --rm replication run_step data_acquisition --no-vulcsv --no-coverage --no-srcmap
```

### 5) Open shell inside container

```bash
docker compose run --rm replication shell
```

### Notes

- `datasets/` is expected to be mounted from host storage (default compose volume mapping uses the project directory).
- Selenium is enabled in-container (headless Chromium). `shm_size: "2gb"` is set in compose for stability.
- For very large runs, tune `VULJIT_WORKERS`, `VULJIT_START_DATE`, and `VULJIT_END_DATE` in `.env.docker`.
- Data acquisition (`VULJIT_START_DATE`, `VULJIT_END_DATE`) may download a wider range of data, but daily aggregation uses a narrower default window in `scripts/modeling/aggregate_metrics_pipeline.py`: `--start-date=2018-08-22`, `--end-date=2025-06-01`.
- `run_step RQ3` executes its child scripts via `bash`, so `chmod +x analysis/research_question3/*.sh` is normally unnecessary.
- If your local environment still reports a permission error, use `chmod +x analysis/research_question3/run_prepare_RQ3.sh analysis/research_question3/rq3.sh` as troubleshooting.

---

## Repository Structure

```
.
├── README                # This file
├── requirements.txt      # Python dependencies
├── run_all_process.sh    # One-click replication script
├── datasets/             # Raw and derived datasets
├── replication/          # Step-by-step scripts
│   ├── data_acquisition.sh
│   ├── ossfuzz_clone_and_analyze.sh
│   ├── metrics_extraction.sh
│   ├── aggregate_metrics_pipeline.sh
│   ├── cross_project_prediction.sh
│   ├── RQ1_2.sh
│   └── RQ3.sh
├── scripts/              # scripts
└── analysis/             # Analysis outputs
```

---

## Reproducing the Experiments

### Quick Start (Recommended)

Run all steps sequentially from the project root:

```bash
bash run_all_process.sh
```

### Step-by-Step Execution

```bash
cd replication

# Step 1: Data acquisition
bash data_acquisition.sh

# Step 2: Clone OSS-Fuzz and analyze builds
bash ossfuzz_clone_and_analyze.sh

# Step 3: Extract metrics
bash metrics_extraction.sh

# Step 4: Aggregate metrics
bash aggregate_metrics_pipeline.sh

# Step 5: Train cross-project prediction models
bash cross_project_prediction.sh

# Step 6: Generate RQ1/RQ2 results
bash RQ1_2.sh

# Step 7: Run RQ3 simulation
bash RQ3.sh
```

> **Note**: Some scripts require substantial time and resources. We strongly recommend running them on a SLURM-managed cluster. Due to the large data volume, we provide the aggregated dataset (After Step4) instead of the raw data.
---

## Expected Results

<!-- Describe where the outputs are stored and what files the reader should expect. -->

After successful execution, results will be located in:
| Research Question | Output path                          |
|-------------------|--------------------------------------|
| RQ1 / RQ2         | `datasets/derived_artifacts/rq1_rq2` |
| RQ3               | `datasets/derived_artifacts/rq3`     |

---

## License
This project is licensed under the [MIT License](LICENSE).
