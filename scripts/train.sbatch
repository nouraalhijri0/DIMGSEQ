#!/bin/bash --login
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=700G
#SBATCH --constraint=[v100]
#SBATCH --job-name=Diffusion
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j-result.out
#SBATCH --error=%x-%j-result.err

conda activate pytorch

nvidia-smi

python -m torch.distributed.launch --nproc_per_node=8 --master_port=12233 --use_env run_train.py \
--diff_steps 1000 \
--lr 0.0001 \
--learning_steps 232184 \
--save_interval 64 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 768 \
--bsz 512 \
--dataset artelingo \
--data_dir '../../wiki_art_paintings/english/train/artemis_preprocessed.csv' \
--vocab bert \
--seq_len 64 \
--schedule_sampler lossaware \
--notes qqp
