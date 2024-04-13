#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=iiai
#SBATCH --job-name=mini_conv4

#time=$(date "+%Y-%m-%d %H:%M:%S")
#time=$(date "+%Y-%m-%d")
#out=results/flexmatch_${time}.txt
export CUDA_VISIBLE_DEVICES=0
source activate ssl
cd ..

python flexmatch.py --c ./config/flexmatch/flexmatch_mini_conv4_npc020.yaml 
