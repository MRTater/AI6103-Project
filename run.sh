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
--img_size 256 \
--batch_size 64 \
--epochs 500 \
--T 500 \
--dataset_folder "/home/msai/zfu009/dataset/Human_faces" \
--activation silu \
--lr_scheduler cosine \
--use_skip_connection \
--resume_from "/home/msai/zfu009/workspace/AI6103-Project/models/50.pth"
# --beta_schedule cosine \
# --use_self_attention \