#!/bin/bash
#SBATCH --job-name=toy
#SBATCH --partition=gpu-a40
#SBATCH --account=simondu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --chdir=/gscratch/simondu/zhaoyi/decision-transformer/toy/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaoyi20@cs.washington.edu

source ~/.bashrc
conda activate test
# singularity shell --nv --bind /gscratch dta.sif
# source /opt/conda/etc/profile.d/conda.sh
# conda activate dta
# singularity exec --nv --bind /gscratch dta.sif ./run.sh > result/Breakout2.txt

output_file=toy5_2.txt
./run.sh > result/${output_file}
