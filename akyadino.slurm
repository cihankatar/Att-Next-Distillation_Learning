#!/bin/bash
#SBATCH -A ckatar
#SBATCH -p barbun-cuda
#SBATCH -J qe_test
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

### Load modules
module load miniconda3
conda activate segmentation
cd /arf/home/ckatar/Desktop/Att-Next-Distillation_Learning
python train_dino.py
