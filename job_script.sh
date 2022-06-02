#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 5:00
 
#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
pip install --user SimpleITK
 
#Copy input file to scratch
mkdir "$TMPDIR"/LIDC_IDRI
cp -R "$HOME"/LUNG-NODULE-ANALYSIS-2022/LIDC_IDRI/ "$TMPDIR"/LIDC_IDRI

#Create output directory on scratch
mkdir "$TMPDIR"/gen_data_dir
mkdir "$TMPDIR"/output_dir
 
#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
python "$HOME"/LUNG-NODULE-ANALYSIS-2022//train.py --raw_data_dir "$TMPDIR"/LIDC_IDRI --out_dir "$TMPDIR"/output_dir --gen_data_dir "$TMPDIR"/LIDC_IDRI_GEN --problem malignancy  --epochs 1

mkdir -p "$HOME"/LUNG-NODULE-ANALYSIS-2022/gen_data_dir
mkdir -p "$HOME"/LUNG-NODULE-ANALYSIS-2022/output_dir
#Copy output directory from scratch to home
cp -r "$TMPDIR"/output_dir "$HOME"/LUNG-NODULE-ANALYSIS-2022/output_dir
cp -r "$TMPDIR"/gen_data_dir "$HOME"/LUNG-NODULE-ANALYSIS-2022/gen_data_dir