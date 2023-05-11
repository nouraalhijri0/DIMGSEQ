#!/bin/bash --login
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=512G
#SBATCH --constraint=[v100]
#SBATCH --job-name=VAE
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j-result.out
#SBATCH --error=%x-%j-result.err

conda activate pytorch

nvidia-smi

python train_vae.py --checkpoint_path='vae-checkpoint'