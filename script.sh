#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
 
#Loading modules
module purge 
module load 2021 Python/3.9.5-GCCcore-10.3.0 
pip uninstall tensorflow
python train.py
