#!/bin/bash
#SBATCH --ntasks=20
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --partition=iiai
#SBATCH --job-name=mini_semco

nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1
export PATH="~/miniconda3/bin:$PATH"
source activate EP

cd ..

time=$(date "+%Y-%m-%d-%H-%M-%S")
e=finetune
out=results_noise_v12/${e}_${time}.txt

sb=./logs_noise_v12/${e}
echo $sb
python3 trainval_semi_test.py -e ${e} -sb $sb -d ./data >>$out

python3 testing.py -sb $sb >>$out
python3 testing.py -sb $sb -p True >>$out
wait
#
python3 save_final_table.py -sb $sb  -m base >>$out
