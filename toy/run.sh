#!/bin/bash

# source /opt/conda/etc/profile.d/conda.sh
# conda activate dta

# mm-dd-hour-min
suffix=06121834
env=./env/env_bandit.txt
data=./dataset/bandit_exp_hasha.csv
goal=20
repeat=1
batch=4
lr=6e-4

# for ctx in 1 2 4 6 8
# do
# python run_toy.py --context_length 1 --epochs 50 --env_path ./env/env_3s_rand.txt --batch_size 4 --goal 200 --data_file ./dataset/rand_exp.csv --rate 6e-4 --tb_suffix $suffix --model 'mlp'
# python run_toy.py --context_length 2 --epochs 50 --env_path ./env/env_3s_rand.txt --batch_size 4 --goal 200 --data_file ./dataset/rand_exp.csv --rate 6e-4 --tb_suffix $suffix --model 'mlp'
# done
# python test.py --context_length 6 --epochs 200 --model_type 'reward_conditioned' --batch_size 4 --horizon 20 --goal 20 --data_file ./dataset/toy_g10.csv --rate 6e-4 --tb_suffix $suffix --ckpt_prefix ./model/m_rand_exp

# dt
# for ctx in 1 2 10 20
# do
#     python run_toy.py --context_length $ctx --epochs 200 --env_path ./env/env_mid-s.txt --batch_size 20 --goal 71 --data_file ./dataset/mid-s_exp.csv --rate 6e-4 --tb_suffix $suffix --model 'dt'
# done

# mlp

for ctx in 1 2
do
    for n_embd in 1 2 4
    # echo python run_toy.py --context_length $ctx --epochs 3 --model_type 'reward_conditioned' --batch_size 1 --horizon 5 --goal $goal --data_file ./dataset/toy5.csv --rate 6e-4
    # python run_toy.py --context_length $ctx --epochs 100 --model_type 'reward_conditioned' --batch_size $batch --horizon 20 --goal $goal --data_file ./dataset/toy5.csv --rate 6e-4 --tb_suffix $suffix
    # python run_toy.py --context_length $ctx --epochs 100 --model_type 'reward_conditioned' --batch_size $batch --horizon 20 --goal $goal --data_file ./dataset/toy_alt.csv --rate 6e-4 --tb_suffix $suffix

    # python run_toy.py --context_length $ctx --epochs 500 --env_path ./env/env_alt-s.txt --batch_size 4 --goal 20 --data_file ./dataset/alt-s_exp.csv --rate 6e-4 --tb_suffix $suffix --model 'mlp'
    # python run_toy.py --context_length $ctx --epochs 500 --env_path ./env/env_alt-s.txt --batch_size 4 --goal 20 --data_file ./dataset/alt-s_exp.csv --rate 6e-4 --tb_suffix $suffix --model 'mlp' --arch '10'
    # python run_toy.py --context_length $ctx --epochs 500 --env_path ./env/env_alt-s.txt --batch_size 4 --goal 20 --data_file ./dataset/alt-s_exp.csv --rate 6e-4 --tb_suffix $suffix --model 'mlp' --arch '10-5-2'
    do
        python run_toy.py --context_length $ctx --epochs 200 --env_path $env --batch_size $batch --goal $goal --data_file $data --rate 6e-4 --tb_suffix $suffix --model 'mlp' --arch '' --repeat $repeat --n_embd ${n_embd} --rate $lr
        # for arch in '1' '2' '5'
        # do
        #     python run_toy.py --context_length $ctx --epochs 200 --env_path $env --batch_size $batch --goal $goal --data_file $data --rate 6e-4 --tb_suffix $suffix --model 'mlp' --arch $arch --repeat $repeat --n_embd ${n_embd} --rate $lr
        # done
    done
    # python run_toy.py --context_length $ctx --epochs 100 --model_type 'reward_conditioned' --batch_size $batch --horizon 20 --goal $goal --data_file ./dataset/toy_mix10.csv --rate 6e-4 --tb_suffix $suffix
done

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
# python run_toy.py --context_length 4 --epochs 75 --model_type reward_conditioned --batch_size 2 --horizon 20 --goal 20 --data_file ./dataset/toy5.csv --rate 6e-4
# python run_toy.py --context_length 2 --epochs 75 --model_type reward_conditioned --batch_size 4 --horizon 20 --goal 20 --data_file ./dataset/toy_mix.csv --rate 6e-4 --hash True
# python run_toy.py --context_length 20 --epochs 50 --model_type reward_conditioned --batch_size 4 --horizon 20 --goal 20 --data_file ./dataset/toy_mix.csv --rate 6e-4 --ckpt_prefix ./model/m_mix
# python run_toy.py --context_length 2 --epochs 50 --model_type reward_conditioned --batch_size 4 --horizon 20 --goal 20 --data_file ./dataset/toy5.csv --rate 6e-4 

# echo python run_toy.py --context_length 20 --epochs 30 --model_type naive --batch_size 4 --horizon 20 --goal 20 --data_file ./dataset/toy5.csv --ckpt_prefix ./model/m5_fullctx 
# python run_toy.py --context_length 20 --epochs 30 --model_type naive --batch_size 4 --horizon 20 --goal 20 --data_file ./dataset/toy5.csv --ckpt_prefix ./model/m5_fullctx