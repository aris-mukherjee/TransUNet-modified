#!/bin/bash
#SBATCH  --output=log_NCI/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
#SBATCH  --constraint=geforce_gtx_titan_x

source /itet-stor/arismu/net_scratch/conda/bin/activate pytorch_env
conda activate pytorch_env

python test_training_set.py --dataset RUNMC --vit_name R50-ViT-B_16
