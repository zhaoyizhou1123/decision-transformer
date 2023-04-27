#!/bin/bash

# source /opt/conda/etc/profile.d/conda.sh
# conda activate dta

# for ctx in 1 2 3
# do
#     for goal in 4 5 10
#     do
#         echo python run_toy.py --context_length $ctx --epochs 3 --model_type 'reward_conditioned' --batch_size 1 --horizon 5 --goal $goal
#         python run_toy.py --context_length $ctx --epochs 3 --model_type 'reward_conditioned' --batch_size 1 --horizon 5 --goal $goal
#     done
# done

horizon=20
file=./dataset/toy4.csv

# for batch in 1 2 3
# do
#     for goal in 10 20 30
#     do 
#         echo python run_toy.py --context_length 2 --epochs 3 --model_type 'reward_conditioned' --batch_size $batch --horizon $horizon --goal $goal --data_file $file
#         python run_toy.py --context_length 2 --epochs 3 --model_type 'reward_conditioned' --batch_size $batch --horizon $horizon --goal $goal --data_file $file
#     done
# done

echo python run_toy.py --context_length 2 --epochs 50 --model_type reward_conditioned --batch_size 4 --horizon 20 --goal 20 --data_file ./dataset/toy5.csv --ckpt_prefix ./model/m5
python run_toy.py --context_length 2 --epochs 50 --model_type reward_conditioned --batch_size 4 --horizon 20 --goal 20 --data_file ./dataset/toy5.csv --ckpt_prefix ./model/m5