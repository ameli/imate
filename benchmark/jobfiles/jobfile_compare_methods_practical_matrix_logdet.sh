#!/bin/bash

#SBATCH --job-name=comp_pract_logdet
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
#SBATCH --output=output_compare_method_logdet.log

PYTHON_DIR=$HOME/programs/miniconda3
SCRIPTS_DIR=$(dirname $PWD)/scripts
LOG_DIR=$PWD

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# $PYTHON_DIR/bin/python ${SCRIPTS_DIR}/compare_methods_practical_matrix.py -a -f traceinv > ${LOG_DIR}/stream_output_compare_methods_practical_matrix_traceinv.txt
$PYTHON_DIR/bin/python ${SCRIPTS_DIR}/compare_methods_practical_matrix.py -a -f logdet > ${LOG_DIR}/stream_output_compare_methods_practical_matrix_logdet.txt
