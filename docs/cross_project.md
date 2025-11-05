# Cross-Project Modeling Guide

This guide explains how to run the new cross-project prediction pipeline.

## Preparing Data
1. Run `scripts/modeling/aggregate_metrics_pipeline.py` with `--emit-cross-project` (or set `VULJIT_EMIT_CROSS_PROJECT=1`).
2. The script adds `project_name` into each aggregated CSV and keeps an up-to-date combined dataset at `datasets/model_inputs/cross_project/cross_project_metrics.parquet` (override with `VULJIT_CROSS_PROJECT_DATA_PATH`).

## Training & Evaluation
```bash
python scripts/modeling/main_cross_project.py \
  --input datasets/model_inputs/cross_project/cross_project_metrics.parquet \
  --evaluation-method group_k_fold \
  --holdout-projects apache-httpd,openssl
```

Key options:
- `--train-projects` / `--holdout-projects`: restrict the dataset.
- `--evaluation-method`: choose `group_k_fold`, `stratified_group_k_fold`, or `leave_one_project_out`.
- `--output-dir`: custom results directory (defaults to `datasets/model_outputs/<model>/cross_project`).

Outputs include:
- `<exp>_metrics.json`: averaged metrics over repetitions.
- `<exp>_per_fold_metrics.csv`: per fold/per repetition diagnostics.
- `<exp>_per_project_summary.csv`: aggregated metrics for each target project.
- `cross_project_predictions.csv`: merged dataset with probability/label columns per experiment.

## Batch Execution on Slurm
Use the helper script:
```bash
sbatch scripts/modeling/predict_cross_project.sh "apache-httpd" "openssl,nginx"
```
Arguments map to holdout projects and explicit training projects. Override dataset/results paths via `CROSS_INPUT_PATH` or `OUTPUT_DIR` environment variables.
