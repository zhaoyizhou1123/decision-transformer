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
output_file=atari_20230424.txt
./run.sh > result/${output_file}
