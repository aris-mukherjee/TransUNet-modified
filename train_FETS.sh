#!/bin/bash
#SBATCH  --output=train_log_2022/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
#SBATCH  --constraint=geforce_gtx_titan_x

source /itet-stor/arismu/net_scratch/conda/bin/activate pytorch_env
conda activate pytorch_env

CUDA_VISIBLE_DEVICES=0 python -u train_FETS.py