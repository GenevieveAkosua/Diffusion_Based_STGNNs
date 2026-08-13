#!/bin/sh

#SBATCH --account=acsl
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="DiffSTGHyperparam"
#SBATCH --mail-user=chkkar002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/envs/diffstg/bin/activate
pip install nni
nnictl create --config config.yaml --port 8080

echo "NNI experiment started!"
