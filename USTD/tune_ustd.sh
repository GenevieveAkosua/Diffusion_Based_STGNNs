#!/bin/sh

#SBATCH --account=a100free
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=4 --gres=gpu:ampere:1
#SBATCH --time=48:00:00
#SBATCH --job-name="ustd_tuning"
#SBATCH --mail-user=chkkar002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=64G

#source ~/envs/diffstg/bin/activate
#./train.sh gwavenet SAWS NA 48 NA config1 128 0 2030
#./train.sh stdiffusionfore SAWS NA 48 gwavenet_NA_20260819T022245 config_SAWS 64 0 2030

source ~/envs/diffstg/bin/activate
cd /scratch/chkkar002/Diffusion_Based_STGNNs/USTD
python tune_ustd_optuna.py --n_trials 30 --study_name ustd_stdiffusionfore_tuning --batch_size 64 --seed 2030 --gpu_ids 0
