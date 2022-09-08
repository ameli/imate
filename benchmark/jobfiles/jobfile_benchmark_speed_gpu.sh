#!/bin/bash

#SBATCH --job-name=benchmark_gpu
#SBATCH --mail-type=ALL                         # (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sameli@berkeley.edu
#SBATCH --partition=savio2_1080ti
#SBATCH --account=fc_biome
#SBATCH --qos=savio_normal
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
###SBATCH --mem=64gb
#SBATCH --output=output_gpu.log

PYTHON_DIR=$HOME/programs/miniconda3
SCRIPTS_DIR=$(dirname $PWD)/scripts
LOG_DIR=$PWD

module load cuda/11.2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PYTHON_DIR/bin/python ${SCRIPTS_DIR}/benchmark_speed.py -g > ${LOG_DIR}/stream_output_gpu.txt
