#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00

#time=$(date "+%Y-%m-%d %H:%M:%S")
time=$(date "+%Y-%m-%d-%H-%M-%S")
out=results/mini_test_${time}.txt

nvidia-smi
export CUDA_VISIBLE_DEVICES=0
source activate EP

cd ..

# testing tiered base
time=$(date "+%Y-%m-%d-%H-%M-%S")
out=results_noise/test_${time}.txt
#sb=./logs_semi_base_norm/finetune_tiered_lr0001
#sb=./logs_semi_base_norm/finetune_cub_1gpu
#sb=./logs_semi_base_norm_proto/finetune_cub_1gpu
sb=./logs_noise_semco_v10/finetune_no_trans_encoder
sb=./logs_noise_semco_v12/finetune_trans_encoder_resort3
sb=./logs_noise_v12/cw_hp
#sb=./logs_noise_semco_v12/compare_test10000
#sb=./logs_noise_semco_v12/finetune_trans_encoder_no_ep
#sb=./logs_noise_v12_base/finetune_noise
#sb='./logs_tmp/test2'
echo $sb
python3 testing.py -sb $sb #>>$out
#python3 testing.py -sb $sb -p True >>$out
#python3 save_final_table.py -sb $sb  -m Base >>$out