#!/bin/sh

#SBATCH --account=a100free
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:ampere:1
#SBATCH --time=48:00:00
#SBATCH --job-name="DiffSTGHyperparam"
#SBATCH --mail-user=chkkar002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=64G

source ~/envs/diffstg/bin/activate
python train.py --n_trials 30 --study_name diffstg_saws_tuning_v2
