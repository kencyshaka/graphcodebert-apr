#!/bin/sh
#SBATCH -p crtai3
#SBATCH -N 1
#SBATCH --gpus-per-node=1

source activate /home/mshaka/anaconda3/envs/graphcodebert

scale=small
output_dir=data/$scale/input
train_file=data/$scale/train.buggy-fixed.buggy,data/$scale/train.buggy-fixed.fixed
val_file=data/$scale/valid.buggy-fixed.buggy,data/$scale/valid.buggy-fixed.fixed
test_file=data/$scale/test.buggy-fixed.buggy,data/$scale/test.buggy-fixed.fixed
pretrained_model=graphcodebert-base/
source_length=320
target_length=256

mkdir -p $output_dir

python run.py --do_data_processing --model_type roberta --model_name_or_path graphcodebert-base/ --tokenizer_name $pretrained_model --config_name $pretrained_model --train_filename $train_file --val_filename $val_file --test_filename $test_file --max_source_length $source_length --max_target_length $target_length --output_dir $output_dir 2>&1| tee $output_dir/process.log

