#!/bin/bash
##SBATCH --time=2  # wall-clock time limit in minutes
#SBATCH -p special
#SBATCH --gres=gpu:4,gpu_mem:4000M  # number of GPUs (keep it at 3) and memory limit
#SBATCH --cpus-per-task=2            # number of CPU cores
#SBATCH --output=logging/bs32_catch_DQN.txt       # output file
##SBATCH --error=error.txt # error file
CUDA_VISIBLE_DEVICES=2
python3 main.py -cf configs/atari_categorical.yaml --exp-name epsilon_decay_greedy