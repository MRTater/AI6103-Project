#!/bin/bash
#SBATCH --partition=SCSEGPU_M1
#SBATCH --qos=q_amsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --job-name=mobilenetv2
#SBATCH --output=diffusion_test.out
#SBATCH --error=test_error.err
export CUBLAS_WORKSPACE_CONFIG=:16:8
module load anaconda
eval "$(conda shell.bash hook)" 
conda activate diffusionsdf
python test.py \
--img_size 64 \
--model_path "/home/msai/luoy0043/workspace/6103/project/models/450.pth" 
