!/bin/bash

#SBATCH --job-name=vary_practical
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
#SBATCH --output=output_vary_method_practical-%J.log

### Running 1 jobs on 2 nodes.
#SBATCH --array=0-1

PYTHON_DIR=$HOME/programs/miniconda3
SCRIPTS_DIR=$(dirname $PWD)/scripts
LOG_DIR=$PWD

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ${SLURM_ARRAY_TASK_ID}

# Job 0: using ortho
if [ ${SLURM_ARRAY_TASK_ID} -eq 0 ];
then
    $PYTHON_DIR/bin/python ${SCRIPTS_DIR}/vary_lanczos_degree_practical_matrix.py -o > ${LOG_DIR}/stream_output_vary_lanczos_degree_practical_matrix_ortho.txt
fi

# Job 1: not using ortho
if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ];
then
    $PYTHON_DIR/bin/python ${SCRIPTS_DIR}/vary_lanczos_degree_practical_matrix.py -n > ${LOG_DIR}/stream_output_vary_lanczos_degree_practical_matrix_not_ortho.txt
fi
