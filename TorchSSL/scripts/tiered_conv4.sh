#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=iiai
#SBATCH --job-name=tiered_conv4

#time=$(date "+%Y-%m-%d %H:%M:%S")
#time=$(date "+%Y-%m-%d")
#out=results/flexmatch_${time}.txt
export PATH="~/miniconda3/bin:$PATH"
source activate ssl
cd ..

#python flexmatch.py --c ./config/flexmatch/flexmatch_mini_conv4_npc020.yaml
d=tiered-imagenet
nc=351
npc=020
net=conv4
# 020 100 200
for npc in 200
do
echo $npc
python flexmatch.py --num_per_class ${npc} --dataset ${d} --num_classes ${nc} \
    --net ${net} --save_name flexmatch_${d}_${net}_npc${npc}  --gpu 0
done #&
