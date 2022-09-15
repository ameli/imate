#!/bin/bash

#SBATCH --job-name=affine_matrix
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
#SBATCH --output=output_affine_matrix_%J.log

PYTHON_DIR=$HOME/programs/miniconda3
SCRIPTS_DIR=$(dirname $PWD)/scripts
LOG_DIR=$PWD

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ${SLURM_ARRAY_TASK_ID}

# Job 0: logdet without gram
if [ ${SLURM_ARRAY_TASK_ID} -eq 0 ];
then
    $PYTHON_DIR/bin/python ${SCRIPTS_DIR}/affine_matrix_function.py -f logdet > ${LOG_DIR}/affine_matrix_function_logdet.txt
fi

# Job 1: logdet with gram
if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ];
then
    $PYTHON_DIR/bin/python ${SCRIPTS_DIR}/affine_matrix_function.py -f logdet -g > ${LOG_DIR}/affine_matrix_function_logdet_gram.txt
fi

# Job 2: traceinv without gram
if [ ${SLURM_ARRAY_TASK_ID} -eq 2 ];
then
    $PYTHON_DIR/bin/python ${SCRIPTS_DIR}/affine_matrix_function.py -f traceinv > ${LOG_DIR}/affine_matrix_function_traceinv.txt
fi

# Job 3: traceinv with gram
if [ ${SLURM_ARRAY_TASK_ID} -eq 3 ];
then
    $PYTHON_DIR/bin/python ${SCRIPTS_DIR}/affine_matrix_function.py -f traceinv -g > ${LOG_DIR}/affine_matrix_function_traceinv_gram.txt
fi

