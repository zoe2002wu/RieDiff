#!/bin/bash
#SBATCH -c 1                     # Number of cores requested
#SBATCH -t 600                  # Runtime in minutes
#SBATCH -p serial_requeue        # Partition to submit to
#SBATCH --open-mode=append       # Append when writing files
#SBATCH -o output_%j.out       # Standard out goes to this file
#SBATCH -e output_%j.err       # Standard err goes to this file
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --mem=20000

# Activate your virtual environment (if you have one)
source ~/.bashrc
mamba activate venv         # Ensure the path to your virtual environment is c$
# Run your Python script
module purge
module load gcc/10.2.0-fasrc01 
module load cuda/11.8.0-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01

echo $CUDA_HOME
rm -rf work_dir
python main.py -cc configs/default_cifar10.txt -sc configs/specific_cifar10.txt --root ./ --mode train --workdir work_dir/cifar10 --n_gpus_per_node 1 --training_batch_size 64 --testing_batch_size 64 --sampling_batch_size 64