scale=original/input
lr=1e-4
batch_size=2 #8
beam_size=3  #10
source_length=320
target_length=256
output_dir=saved_models/$scale/
train_file=data/$scale/train.pkl
val_file=data/$scale/valid.pkl
dev_file=data/$scale/dev.pkl
test_file=data/$scale/test.pkl
epochs=1   #should be changed to 50
pretrained_model=graphcodebert-base/

source activate /Users/apple/opt/anaconda3/envs/phd

mkdir -p $output_dir

python run.py --do_train --do_test --do_best_loss --model_type roberta --model_name_or_path graphcodebert-base/ --tokenizer_name $pretrained_model --config_name $pretrained_model --train_filename $train_file --val_filename $val_file --test_filename $test_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs 2>&1| tee $output_dir/train.log
