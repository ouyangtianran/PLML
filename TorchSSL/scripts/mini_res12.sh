#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=iiai
#SBATCH --job-name=mini_res12

#time=$(date "+%Y-%m-%d %H:%M:%S")
#time=$(date "+%Y-%m-%d")
#out=results/flexmatch_${time}.txt

nvidia-smi
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="~/miniconda3/bin:$PATH"

source activate ssl
cd ..
d=mini-imagenet
nc=64
npc=020
net=resnet12
for npc in 020 050 100
do
python flexmatch.py --num_per_class ${npc} --dataset ${d} --num_classes ${nc} \
    --net ${net} --save_name flexmatch_${d}_${net}_npc${npc}
done #&
#wait