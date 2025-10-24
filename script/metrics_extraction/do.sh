#!/bin/bash

#SBATCH --job-name=metrics_extract
#SBATCH --output=logs/metrics_extract_%A_%a.out
#SBATCH --error=errors/metrics_extract_%A_%a.err
#SBATCH --array=1-3
#SBATCH --time=120:00:00
#SBATCH --partition=cluster_low
#SBATCH --ntasks=1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=52
#SBATCH --mail-user=kato.riku.ks5@naist.ac.jp
#SBATCH --mail-type=END,FAIL 

mkdir -p errors
mkdir -p logs


bash ex1/ex1.sh $SLURM_ARRAY_TASK_ID
bash /work/riku-ka/metrics_culculator/ex1/ex2.sh
bash /work/riku-ka/metrics_culculator/collect_metrics_from_github_project.sh