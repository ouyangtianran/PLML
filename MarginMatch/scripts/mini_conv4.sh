#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
source activate marginmatch
cd ..

python marginmatch.py --c "./config/marginmatch/marginmatch_mini_050.yaml"
