#!/bin/bash

suffix=06281712 # directory name for tensorboard record
env=./env/env_hard.txt # path to environment descriptio/gscratch/simondu/zhaoyi/decision-transformer/toy/logs/cqln file
data=./dataset/timevarbandit_exp.csv # path to dataset file
epoch=20000 # training epochs
# batch_rvs=4 # batch size for supervised learning/dt
# batch_cql=8000 # batch size for Q-learning
batch=8000
lr=6e-3 # learning rate
ctx=1 # context length for supervised learning
goal=20 # learning objective for supervised learning
embd=-1
env_type=timevar_bandit
# period=100

# embedding dimension, -1 stands for no embedding
for period in 800
do
    # network architecture, e.g., '10-1' means 2 hidden layers, first 10 neurons, second 1 neuron. '/' means no hidden layer
    for arch in '10' '20' '40' '80' '160' '200'
    do
        # run rvs method
        # python run_toy.py --epochs $epoch --batch_size ${batch_rvs} --context_length $ctx --data_file $data --env_path $env --rate $lr --arch $arch --n_embd $embd --model 'mlp' --tb_suffix $suffix --goal $goal
        # for ctx in 1 2 4
        # do
        # python run_toy.py --epochs $epoch --batch_size ${batch} --context_length $ctx --data_file $data --env_path $env --rate $lr --arch $arch --n_embd $embd --model 'mlp' --tb_suffix $suffix --goal $goal --env_type ${env_type}
        # done

        # run Q-learning method
        python run_cql.py --epochs $epoch --batch_size ${batch} --data_file $data --rate $lr --arch $arch --env_path $env --tradeoff_coef 0 --tb_suffix $suffix --env_type ${env_type} --train_mode 'dqn' --dqn_upd_period $period
    done
done