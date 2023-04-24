#!/bin/bash
#SBATCH --job-name=atari
#SBATCH --partition=gpu-a40
#SBATCH --account=simondu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/gscratch/simondu/zhaoyi/decision-transformer/atari/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaoyi20@cs.washington.edu

source ~/.bashrc
conda activate test
rm result.log
for seed in 123 231 312
do
	echo game=Breakout, seed=$seed
	python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128 >> result.log
done
