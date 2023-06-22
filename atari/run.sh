#!/bin/bash

# source /opt/conda/etc/profile.d/conda.sh
# conda activate dta

# for seed in 123 231 312
# do
#     echo python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128
#     python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128
# done

# for seed in 123 231 312
# do
#     echo python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128
#     python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128
# done

# for seed in 123 231 312
# do
#     echo python run_dt_atari.py --seed $seed --context_length 50 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512
#     python run_dt_atari.py --seed $seed --context_length 50 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512
# done

# for seed in 123 231 312
# do
#     echo echo python run_dt_atari.py --seed $seed --context_length 50 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 512
#     python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
# done

python run_dt_atari.py --seed 123 --context_length 2 --epochs 2 --model_type 'reward_conditioned' --num_steps 5 --num_buffers 1 --game 'Breakout' --batch_size 7
