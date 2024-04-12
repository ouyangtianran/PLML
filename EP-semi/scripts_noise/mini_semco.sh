#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=iiai
#SBATCH --job-name=mini_semco


#time=$(date "+%Y-%m-%d %H:%M:%S")
time=$(date "+%Y-%m-%d-%H-%M-%S")
ver=semco_v12
out=results_noise/mini_${ver}_noise_${time}.txt

nvidia-smi
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export PATH="~/miniconda3/bin:$PATH"
source activate EP

cd ..

sb=./logs_noise_${ver}/finetune_trans_encoder_resort3
#python3 trainval_semi_test.py -e finetune_noise -sb ./logs_noise_semco_v0/finetune_cw01 -d ./data >>$out
#python3 trainval_semi_test.py -e finetune_noise -sb ./logs_noise_semco_v01/finetune_cw01 -d ./data >>$out
#python3 trainval_semi_test.py -e finetune_noise -sb ./logs_noise_${ver}/finetune_trans_encoder_rbf -d ./data >>$out
#python3 trainval_semi_test.py -e finetune_noise -sb ./logs_noise_${ver}/finetune_no_trans_encoder -d ./data >>$out
python3 trainval_semi_test.py -e finetune_noise -sb $sb -d ./data

python3 testing.py -sb $sb
#python3 trainval_emb_semi.py -e ssl_large -sb ./logs_emb_semi/ssl2/ -d ./data >>$out

#python3 trainval_emb_semi_test.py -e finetune -sb ./logs_semi_base_semco/finetune -d ./data >>$out
#python3 trainval_semi_test.py -e finetune_transformer -sb ./logs_trans/finetune -d ./data