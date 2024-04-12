#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00

#time=$(date "+%Y-%m-%d %H:%M:%S")
#time=$(date "+%Y-%m-%d")
#out=results/flexmatch_${time}.txt
source activate ssl
cd ..

#python flexmatch.py --c ./config/flexmatch/flexmatch_mini_conv4_npc020.yaml
d=CUB_200_2011
nc=100
npc=020
net=conv4
save_name={flexmatch_${d}_${net}_npc${npc}
#002 006 010
for npc in 006 010
do
python flexmatch.py --num_per_class ${npc} --dataset ${d} --num_classes ${nc} \
    --net ${net} --save_name ${save_name,,} --num_train_iter 250000
done #&