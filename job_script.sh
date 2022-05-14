#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=03:00:00


#runFolderID=$( date '+%s')
#echo "${runFolderID}"
echo_status () {
    duration=$SECONDS
    textVar="${section}  ($(($duration / 60))m and $(($duration % 60))s)"
    echo $textVar
    SECONDS=0
}

SECONDS=0
echo "start"


#Loading modules
section="Loading python"
#https://servicedesk.surf.nl/wiki/display/WIKI/Loading+modules
module load 2021 # Why do we need this?
module load TensorFlow #Python/3.9.5-GCCcore-10.3.0 # load python
#makes sure the requiret packages are installed
pip install --user SimpleITK --quiet
pip install --user numpy --quiet
pip install --user matplotlib --quiet
pip install --user tensorflow --quiet
pip install --user pathlib --quiet
##TODO load the requirement.txt file instead
echo_status



#Copy input file to scratch
section="copying the Dataset"
USER_DIR_NAME="LNA22_t"
in_dir="$TMPDIR"/LunaDataFolder/in_dir
mkdir -p $in_dir
# We don't have to change the data. Therfore we dont need to have in the git nore in every project folder. One Static location is entought I Think
cp -R "$HOME"/Data/LUNA22_prequel/ "$in_dir"
#Create output directory on scratch
mkdir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir
mkdir "$TMPDIR"/"$USER_DIR_NAME"/output_dir
echo_status
echo $'\n\n------------------------\n'


section="Run code\n"
#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
python "$PWD"/train.py --raw_data_dir "$in_dir"/LUNA22_prequel --out_dir "$TMPDIR"/"$USER_DIR_NAME"/output_dir --gen_data_dir "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir --problem malignancy --epochs 1

echo_status

section="save output to $PWD" 
#Copy output directory from scratch to home
mkdir -p "$PWD"/result/output_dir
mkdir -p "$PWD"/result/gen_data_dir

cp -r "$TMPDIR"/"$USER_DIR_NAME"/output_dir "$PWD"/result/output_dir
cp -r "$TMPDIR"/"$USER_DIR_NAME"/gen_data_dir "$PWD"/result/gen_data_dir
echo_status
