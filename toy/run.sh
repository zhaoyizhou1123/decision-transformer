#!/bin/bash

suffix=06292138 # directory name for tensorboard record
env=./env/env_linearq.txt # path to environment descriptio/gscratch/simondu/zhaoyi/decision-transformer/toy/logs/cqln file
data=./dataset/linearq_exp.csv # path to dataset file
epoch=4000 # training epochs
# batch_rvs=4 # batch size for supervised learning/dt
# batch_cql=8000 # batch size for Q-learning
batch=8000
lr=6e-3 # learning rate
ctx=1 # context length for supervised learning
goal=38 # learning objective for supervised learning
embd=-1
env_type=linearq
# period=100

# embedding dimension, -1 stands for no embedding
for arch in '1' '5' '10' '20' '40' '80'
do
    # run rvs method
    # python run_toy.py --epochs $epoch --batch_size ${batch_rvs} --context_length $ctx --data_file $data --env_path $env --rate $lr --arch $arch --n_embd $embd --model 'mlp' --tb_suffix $suffix --goal $goal
    # for ctx in 1 2 4
    # do
    python run_toy.py --epochs $epoch --batch_size ${batch} --context_length $ctx --data_file $data --env_path $env --rate $lr --arch $arch --n_embd $embd --model 'mlp' --tb_suffix $suffix --goal $goal --env_type ${env_type} --sample
    # done
    for period in 200
    do
        # run Q-learning method
        python run_cql.py --epochs $epoch --batch_size ${batch} --data_file $data --rate $lr --arch $arch --env_path $env --tradeoff_coef 0 --tb_suffix $suffix --env_type ${env_type} --train_mode 'dqn' --dqn_upd_period $period
    done
done