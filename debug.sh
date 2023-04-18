#!/bin/bash
#SBATCH --partition=SCSEGPU_M1
#SBATCH --qos=q_amsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --job-name=fzm_6103proj_debugging
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

export CUBLAS_WORKSPACE_CONFIG=:16:8
module load anaconda/anaconda3
eval "$(conda shell.bash hook)" 
conda activate 6103proj
python train.py \
--img_size 64 \
--batch_size 64 \
--T 1000 \
--epochs 300 \