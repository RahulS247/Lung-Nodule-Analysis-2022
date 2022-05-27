#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
 
module load 2021
module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1

python train.py
