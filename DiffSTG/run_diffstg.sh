#!/bin/bash

#SBATCH --account=a100free
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=4 --gres=gpu:ampere:1
#SBATCH --time=48:00:00
#SBATCH --job-name="Run_DiffSTG"
#SBATCH --mail-user=chkkar002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=64G
#SBATCH --array=0-2

source ~/envs/diffstg/bin/activate

SEEDS=(2024 2025 2026)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python train.py --n_trials 1 --study_name diffstg_run_seed${SEED} --seed ${SEED}
