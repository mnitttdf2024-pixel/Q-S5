#!/bin/bash
#SBATCH -p gpu
#SBATCH --account=gts-smukhopadhyay6-ece
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH -t 5:00:00

# setup python (use PACE module system instead of pyenv)
module load python/3.10

# activate venv
source ./venv/bin/activate

python test_jax_gpu.py