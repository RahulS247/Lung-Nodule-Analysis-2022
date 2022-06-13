#!/bin/bash
#Set job requirementsclass_weight./
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
 
#Loading modules
module load 2021
module load scikit-build/0.11.1-GCCcore-10.3.0
module load OpenCV/4.5.3-foss-2021a-CUDA-11.3.1-contrib
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
mkdir "$HOME"/"$USER_DIR_NAME"/terminal_logs

# Activate venv
python -m venv ./venv
source ./venv/bin/activate

pip install scikit-image
pip install matplotlib
pip install tensorflow-addons
pip install click
pip install simpleitk
pip install sklearn

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "seresnet18" --run_name "seresnet18_normal" --problem "noduletype" --val_fraciton 0.075 --batch_size 15 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/noduletype_ser18n_log.txt
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "seresnet18" --run_name "seresnet18_normal" --problem "malignancy" --val_fraciton 0.075 --batch_size 16 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/malignancy_ser18n_log.txt

python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "seresnet50" --run_name "seresnet50_normal" --problem "noduletype" --val_fraciton 0.075 --batch_size 15 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/noduletype_ser50n_log.txt
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "seresnet50" --run_name "seresnet50_normal" --problem "malignancy" --val_fraciton 0.075 --batch_size 16 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/malignancy_ser50n_log.txt

python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb7" --run_name "eb7_normal" --problem "noduletype" --val_fraciton 0.075 --batch_size 15 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/noduletype_e7n_log.txt
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb7" --run_name "eb7_normal" --problem "malignancy" --val_fraciton 0.075 --batch_size 16 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/malignancy_e7n_log.txt

python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb6" --run_name "eb6_normal" --problem "noduletype" --val_fraciton 0.075 --batch_size 15 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/noduletype_e6n_log.txt
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb6" --run_name "eb6_normal" --problem "malignancy" --val_fraciton 0.075 --batch_size 16 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/malignancy_e6n_log.txt

python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb5" --run_name "eb5_normal" --problem "noduletype" --val_fraciton 0.075 --batch_size 15 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/noduletype_e5n_log.txt
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb5" --run_name "eb5_normal" --problem "malignancy" --val_fraciton 0.075 --batch_size 16 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/malignancy_e5n_log.txt

python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb4" --run_name "eb4_normal" --problem "noduletype" --val_fraciton 0.075 --batch_size 15 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/noduletype_e4n_log.txt
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb4" --run_name "eb4_normal" --problem "malignancy" --val_fraciton 0.075 --batch_size 16 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/malignancy_e4n_log.txt

python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb3" --run_name "eb3_normal" --problem "noduletype" --val_fraciton 0.075 --batch_size 15 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/noduletype_e3n_log.txt
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb3" --run_name "eb3_normal" --problem "malignancy" --val_fraciton 0.075 --batch_size 16 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/malignancy_e3n_log.txt

python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb2" --run_name "eb2_normal" --problem "noduletype" --val_fraciton 0.075 --batch_size 15 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/noduletype_e2n_log.txt
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb2" --run_name "eb2_normal" --problem "malignancy" --val_fraciton 0.075 --batch_size 16 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/malignancy_e2n_log.txt

python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb1" --run_name "eb1_normal" --problem "noduletype" --val_fraciton 0.075 --batch_size 15 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/noduletype_e1n_log.txt
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb1" --run_name "eb1_normal" --problem "malignancy" --val_fraciton 0.075 --batch_size 16 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/malignancy_e1n_log.txt

python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb3v2" --run_name "eb3v2_normal" --problem "noduletype" --val_fraciton 0.075 --batch_size 15 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/noduletype_e3v2n_log.txt
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb3v2" --run_name "eb3v2_normal" --problem "malignancy" --val_fraciton 0.075 --batch_size 16 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/malignancy_e3v2n_log.txt

python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb2v2" --run_name "eb2v2_normal" --problem "noduletype" --val_fraciton 0.075 --batch_size 15 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/noduletype_e2v2n_log.txt
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb2v2" --run_name "eb2v2_normal" --problem "malignancy" --val_fraciton 0.075 --batch_size 16 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/malignancy_e2v2n_log.txt

python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb1v2" --run_name "eb1v2_normal" --problem "noduletype" --val_fraciton 0.075 --batch_size 15 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/noduletype_e1v2n_log.txt
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb1v2" --run_name "eb1v2_normal" --problem "malignancy" --val_fraciton 0.075 --batch_size 16 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/malignancy_e1v2n_log.txt

python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb0v2" --run_name "eb0v2_normal" --problem "noduletype" --val_fraciton 0.075 --batch_size 15 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/noduletype_e0v2n_log.txt
python "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/train.py --raw_data_dir "$TMPDIR"/"$USER_DIR_NAME"/in_dir/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --epochs 250 --base_model "efficientnetb0v2" --run_name "eb0v2_normal" --problem "malignancy" --val_fraciton 0.075 --batch_size 16 --sample_strat normal --preprocessing_type normal > "$HOME"/"$USER_DIR_NAME"/terminal_logs/malignancy_e0v2n_log.txt

#Copy output directory from scratch to home
cp -r "$TMPDIR"/"$USER_DIR_NAME"/output_dir "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/
cp -r "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir "$HOME"/"$USER_DIR_NAME"/Lung-Nodule-Analysis-2022/
