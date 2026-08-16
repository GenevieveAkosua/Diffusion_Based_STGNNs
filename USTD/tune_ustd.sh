#!/bin/sh

#SBATCH --account=a100free
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=4 --gres=gpu:ampere:1
#SBATCH --time=2:00:00
#SBATCH --job-name="Test_ustd_pretrain_test"
#SBATCH --mail-user=chkkar002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=64G

source ~/envs/diffstg/bin/activate
./train.sh gwavenet SAWS NA 24 NA config1 128 0 2030
#./train.sh stdiffusionfore SAWS NA 24 gwavenet_NA_20260815T200107 config_SAWS 64 0 2030
