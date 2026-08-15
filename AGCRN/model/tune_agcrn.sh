#!/bin/sh

#SBATCH --account=a100free
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=4 --gres=gpu:ampere:1
#SBATCH --time=2:00:00
#SBATCH --job-name="AGCRN_tune_TeachForce"
#SBATCH --mail-user=chkkar002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=64G

source ~/envs/diffstg/bin/activate
python Run.py
