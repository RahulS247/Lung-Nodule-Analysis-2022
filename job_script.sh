#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
 
#Loading modules
module load 2021
module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1

#Copy input file to scratch
USER_DIR_NAME="LNA22_t"
DATA_DIR_NAME="Data"
mkdir -p "$TMPDIR"/"$USER_DIR_NAME"/in_dir
mkdir -p "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir
cp -R "$HOME"/"$DATA_DIR_NAME"/LUNA22_prequel/ "$TMPDIR"/"$USER_DIR_NAME"/in_dir
cp -R "$HOME"/"$DATA_DIR_NAME"/gen_data_dir/ "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir

#Create output directory on scratch
mkdir "$TMPDIR"/"$USER_DIR_NAME"/output_dir
 
#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --problem malignancy --epochs 1 --base_model "resnet" --problem "noduletype"
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --problem malignancy --epochs 1 --base_model "resnet" --problem "malignancy"

#Copy output directory from scratch to home
cp -r "$TMPDIR"/"$USER_DIR_NAME"/output_dir "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/
cp -r "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/
