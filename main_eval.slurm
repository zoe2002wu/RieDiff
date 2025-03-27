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
module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01

echo $CUDA_HOME
!rm -rf /content/CLD-SGM/work_dir/cifar10_seed_0/eval
!/content/py38env/bin/python /content/CLD-SGM/main.py \
  -cc /content/CLD-SGM/configs/default_cifar10.txt \
  -sc /content/CLD-SGM/configs/specific_cifar10.txt \
  --root /content/CLD-SGM \
  --mode eval \
  --eval_folder eval/cifar10 \
  --workdir work_dir/cifar10 \
  --n_gpus_per_node 1 \
  --training_batch_size 64 \
  --testing_batch_size 64 \
  --sampling_batch_size 64 \
  --ckpt_file checkpoint.pth
