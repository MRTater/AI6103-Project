#!/bin/bash
#SBATCH --partition=SCSEGPU_M1
#SBATCH --qos=q_amsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --job-name=diffusion_pokemon
#SBATCH --output=diffusion_pokemon.out
#SBATCH --error=error.err
export CUBLAS_WORKSPACE_CONFIG=:16:8
module load anaconda
eval "$(conda shell.bash hook)" 
conda activate diffusionsdf
python train.py \
--epochs 100
