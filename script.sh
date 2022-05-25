#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
<<<<<<< HEAD
 
module load 2021
module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1

#source ~/venv/bin/activate
=======

source ~/venv/bin/activate
>>>>>>> c73e3615662eb0740f2bc956c91d42535ef0c8ec
python train.py
