#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
 
#Loading modules
module load 2021 Python/3.9.5-GCCcore-10.3.0
module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1


python train.py
