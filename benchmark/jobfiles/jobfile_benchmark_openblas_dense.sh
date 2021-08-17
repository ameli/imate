#!/bin/bash

#SBATCH --job-name=openblas-dense
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
###SBATCH --output=output_without_openblas_dense.log
#SBATCH --output=output_with_openblas_dense.log

PYTHON_DIR=$HOME/programs/miniconda3
PACKAGE_DIR=$(dirname $PWD)
BENCHMARK_DIR=$PACKAGE_DIR/benchmark
LOG_DIR=$BENCHMARK_DIR

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PYTHON_DIR/bin/python ${BENCHMARK_DIR}/benchmark_openblas_dense.py -o True > ${LOG_DIR}/stream_output_with_openblas_dense.txt
#$PYTHON_DIR/bin/python ${BENCHMARK_DIR}/benchmark_openblas_dense.py -o False > ${LOG_DIR}/stream_output_without_openblas_dense.txt
