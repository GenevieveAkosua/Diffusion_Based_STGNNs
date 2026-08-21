#!/bin/sh

#SBATCH --account=a100free
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=4 --gres=gpu:ampere:1
#SBATCH --time=14:00:00
#SBATCH --job-name="AGCRN_RV_WindSir"
#SBATCH --mail-user=chkkar002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=64G

source ~/envs/diffstg/bin/activate
# Parameters extracted from "Best Trial (number=23).png"
BATCH_SIZE=32
LR_INIT=0.0002  # Rounded off from 0.00024218018597619645
LAG=24
NUM_LAYERS=2
RNN_UNITS=64
EMBED_DIM=30
# Loop through 3 different seeds
for seed in 1 2 3; do
    echo "================================================================="
    echo "Running Run.py | Seed: $seed"
    echo "================================================================="
        
    python Run.py \
        --seed $seed \
        --batch_size $BATCH_SIZE \
        --lr_init $LR_INIT \
        --lag $LAG \
        --num_layers $NUM_LAYERS \
        --rnn_units $RNN_UNITS \
        --embed_dim $EMBED_DIM
            
done
