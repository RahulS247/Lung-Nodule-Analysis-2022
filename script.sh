#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
 
#Loading modules
source /home/dbalsameda/new_rah/bodyct-luna22-ismi-training-baseline/env/bin/activate

#module purge
module load 2021
module load cuDNN/8.2.1.32-CUDA-11.3.1


python train.py --problem malignancy
python train.py --problem noduletype
