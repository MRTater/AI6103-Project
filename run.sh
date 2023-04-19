#!/bin/bash
#SBATCH --partition=SCSEGPU_M2
#SBATCH --qos=q_dmsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --job-name=fzm_6103proj
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

export CUBLAS_WORKSPACE_CONFIG=:16:8
module load anaconda/anaconda3
eval "$(conda shell.bash hook)" 
conda activate 6103proj
python train.py \
--img_size 64 \
--batch_size 128 \
--epochs 500 \
--T 300 \
--dataset_folder "/home/msai/zfu009/dataset/stanford_cars" \
--activation silu \
--lr_scheduler cosine \
--use_skip_connection \
--use_self_attention \
# --beta_schedule cosine \
# --resume_from "/home/msai/zfu009/workspace/AI6103-Project/models/1.pth"