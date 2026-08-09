#!/bin/sh

#SBATCH --account=a100free
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=4 --gres=gpu:ampere:1
#SBATCH --time=48:00:00
#SBATCH --job-name="DiffSTGHyperparam"
#SBATCH --mail-user=chkkar002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=64G

source ~/envs/diffstg/bin/activate
echo "=== nvidia-smi check ==="
which nvidia-smi
nvidia-smi
echo "========================"
pip install nni
echo "=== proxy env on compute node ==="
env | grep -i proxy
echo "================================="
export no_proxy="localhost,127.0.0.1,${no_proxy:-}"
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
nnictl create --config config.yaml --port 8080 --foreground

echo "NNI experiment started!"
