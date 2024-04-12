#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=iiai
#SBATCH --job-name=m_s_flexmatch

nvidia-smi
export CUDA_VISIBLE_DEVICES=0
export PATH="~/miniconda3/bin:$PATH"
source activate EP

cd ..

time=$(date "+%Y-%m-%d-%H-%M-%S")
e=flexmatch_ssl
out=results_noise_v12/${e}_${time}.txt

sb=./logs_noise_v12/${e}
echo $sb
#python3 trainval_semi_test.py -e $e -sb $sb -d ./data >>$out
python3 save_final_table_ssl.py -sb $sb  -m ssl >>$out
