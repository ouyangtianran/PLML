#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=iiai
#SBATCH --job-name=mini_conv4

#time=$(date "+%Y-%m-%d %H:%M:%S")
#time=$(date "+%Y-%m-%d")
#out=results/flexmatch_${time}.txt
source activate ssl
cd ..

#python flexmatch.py --c ./config/flexmatch/flexmatch_mini_conv4_npc020.yaml
d=mini-imagenet
nc=64
npc=020
net=conv4
#020 050 100
for npc in 50
do
python flexmatch.py --num_per_class ${npc} --dataset ${d} --num_classes ${nc} \
    --net ${net} --save_name flexmatch_${d}_${net}_npc${npc} --num_train_iter 400000
done #&