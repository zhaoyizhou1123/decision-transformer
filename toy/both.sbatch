#!/bin/bash
#SBATCH --job-name=both
#SBATCH --partition=gpu-a40
#SBATCH --account=simondu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --chdir=/gscratch/simondu/zhaoyi/decision-transformer/toy/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaoyi20@cs.washington.edu

source ~/.bashrc
conda activate test
# singularity shell --nv --bind /gscratch dta.sif
# source /opt/conda/etc/profile.d/conda.sh
# conda activate dta
# singularity exec --nv --bind /gscratch dta.sif ./run.sh > result/Breakout2.txt

output_file=bandit_exp_both.txt
./cql.sh
./run.sh
# ./run.sh > result/${output_file}
# ./cql.sh >> result/${output_file}


