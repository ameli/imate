#!/bin/bash

#SBATCH --job-name=lanc_deg
#SBATCH --mail-type=ALL                         # (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sameli@berkeley.edu
#SBATCH --partition=savio2
#SBATCH --account=fc_biome
#SBATCH --qos=savio_normal
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
###SBATCH --mem=64gb
#SBATCH --output=output_vary_lanczos_degree.log

PYTHON_DIR=$HOME/programs/miniconda3
PACKAGE_DIR=$(dirname $PWD)
BENCHMARK_DIR=$PACKAGE_DIR/benchmark
LOG_DIR=$BENCHMARK_DIR

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PYTHON_DIR/bin/python ${BENCHMARK_DIR}/vary_lanczos_degree.py > ${LOG_DIR}/stream_output_vary_lanczos_degree.txt
