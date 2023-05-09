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

# for batch in 1 2 3
# do
#     for goal in 10 20 30
#     do 
#         echo python run_toy.py --context_length 2 --epochs 3 --model_type 'reward_conditioned' --batch_size $batch --horizon $horizon --goal $goal --data_file $file
#         python run_toy.py --context_length 2 --epochs 3 --model_type 'reward_conditioned' --batch_size $batch --horizon $horizon --goal $goal --data_file $file
#     done
# done

# echo python run_toy.py --context_length 20 --epochs 100 --model_type reward_conditioned --batch_size 4 --horizon 20 --goal 0 --data_file ./dataset/toy5.csv --rate 6e-4 --ckpt_prefix ./model/m5_ctx20_g0
# python run_toy.py --context_length 20 --epochs 100 --model_type reward_conditioned --batch_size 4 --horizon 20 --goal 0 --data_file ./dataset/toy5.csv  --rate 6e-4 --ckpt_prefix ./model/m5_ctx20_g0

# echo python run_toy.py --context_length 20 --epochs 100 --model_type reward_conditioned --batch_size 1 --horizon 20 --goal 20 --data_file ./dataset/toy5_rev.csv --rate 6e-4
python run_toy.py --context_length 2 --epochs 75 --model_type reward_conditioned --batch_size 4 --horizon 20 --goal 20 --data_file ./dataset/toy_mix.csv --rate 6e-4
python run_toy.py --context_length 2 --epochs 75 --model_type reward_conditioned --batch_size 4 --horizon 20 --goal 20 --data_file ./dataset/toy_mix.csv --rate 6e-4 --hash True
# python run_toy.py --context_length 20 --epochs 50 --model_type reward_conditioned --batch_size 4 --horizon 20 --goal 20 --data_file ./dataset/toy_mix.csv --rate 6e-4 --ckpt_prefix ./model/m_mix
# python run_toy.py --context_length 2 --epochs 50 --model_type reward_conditioned --batch_size 4 --horizon 20 --goal 20 --data_file ./dataset/toy5.csv --rate 6e-4 

# echo python run_toy.py --context_length 20 --epochs 30 --model_type naive --batch_size 4 --horizon 20 --goal 20 --data_file ./dataset/toy5.csv --ckpt_prefix ./model/m5_fullctx 
# python run_toy.py --context_length 20 --epochs 30 --model_type naive --batch_size 4 --horizon 20 --goal 20 --data_file ./dataset/toy5.csv --ckpt_prefix ./model/m5_fullctx