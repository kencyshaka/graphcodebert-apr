#!/bin/sh
#SBATCH -p crtai3
#SBATCH -N 1
#SBATCH --gpus-per-node=1

source activate /home/mshaka/anaconda3/envs/graphcodebert 


scale=small
lr=1e-4
batch_size=64 
beam_size=10  #10
source_length=320
target_length=256
output_dir=saved_models/final/
train_file=data/$scale/input/train.pkl
val_file=data/$scale/input/valid.pkl
dev_file=data/$scale/input/dev.pkl
test_file=data/$scale/input/test.pkl
epochs=50   #should be changed to 50
pretrained_model=graphcodebert-base/


mkdir -p $output_dir

python run.py --do_train --do_early_checkpoint --do_best_bleu --model_type roberta --model_name_or_path graphcodebert-base/ --tokenizer_name $pretrained_model --config_name $pretrained_model --train_filename $train_file --val_filename $val_file --test_filename $test_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs 2>&1| tee $output_dir/train.log
