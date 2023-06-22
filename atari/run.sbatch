#!/bin/bash
#SBATCH --job-name=atari
#SBATCH --partition=gpu-a40
#SBATCH --account=simondu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/gscratch/simondu/zhaoyi/decision-transformer/atari/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaoyi20@cs.washington.edu

source ~/.bashrc
conda activate test
# singularity shell --nv --bind /gscratch dta.sif
# source /opt/conda/etc/profile.d/conda.sh
# conda activate dta
# singularity exec --nv --bind /gscratch dta.sif ./run.sh > result/Breakout2.txt

output_file=debug.txt
./run.sh >> result/${output_file}
