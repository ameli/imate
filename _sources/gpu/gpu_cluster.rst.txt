.. _gpu-cluster:

Deploy |project| on GPU Clusters
================================

On GPU clusters, the NVIDIA graphic driver and CUDA libraries are pre-installed and they only need to be loaded.

Load Modules
------------

Check which modules are available on the machine

.. prompt:: bash
    
    module avail

Load python and a compatible CUDA version by

.. prompt:: bash

    module load python/3.9
    module load cuda/11.7

Check which modules are loaded

.. prompt:: bash

    module list

Interactive Session with SLURM
------------------------------

There are two ways to work with GPU on a cluster. The first method is to ``ssh`` to a GPU node and for hands-on interaction with the GPU device. If the GPU cluster uses `SLURM manager <https://slurm.schedmd.com/documentation.html>`_, use ``srun`` to initiate a session as follows

.. prompt:: bash

    srun -A fc_biome -p savio2_gpu --gres=gpu:1 --ntasks 2 -t 2:00:00 --pty bash -i

In the above example:

* ``-A fc_biome`` sets the group account associated with the user.
* ``-p savio2_gpu`` sets the name of the GPU node.
* ``--gres=gpu:1`` requests one GPU device on the node.
* ``--ntasks 2`` requests two parallel CPU threads on the node.
* ``-t 2:00:00`` requests a two-hour session.
* ``--pty bash`` starts a Bash shell.
* ``-i`` redirects std input to the user's terminal for interactive use.

See the list of `options of srun <https://slurm.schedmd.com/srun.html>`_ for details. As another example, to request a GPU node named ``savio2_1080ti`` with 4 GPU devices and 8 CPU threads for 10 hours, run

.. prompt:: bash

    srun -A fc_biome -p savio2_1080ti --gres=gpu:4 --ntasks 8 -t 10:00:00 --pty bash -i

.. note::

    Replace the name of nodes and accounts in the above example with yours. The name of GPU nodes and accounts in the above examples are obtained from `SAVIO Cluster <https://docs-research-it.berkeley.edu/services/high-performance-computing/overview/>`_ (an institutional Cluster at UC Berkeley).

Submit Jobs to GPU with SLURM
-----------------------------

To submit a parallel job to GPU nodes on a cluster with `SLURM manager`, use ``sbatch`` command, such as

.. prompt:: bash

    sbatch jobfile.sh

See the list of `options of sbatch <https://slurm.schedmd.com/sbatch.html>`_ for details. A sample job file, ``jobfile.sh`` is shown below. The highlighted line in the file instructs `SLURM` to request the number of GPU devices with ``--gres`` option.

.. code-block:: Slurm
   :emphasize-lines: 11

    #!/bin/bash

    #SBATCH --job-name=your_project
    #SBATCH --mail-type=your_email
    #SBATCH --mail-user=your_email
    #SBATCH --partition=savio2_1080ti
    #SBATCH --account=fc_biome
    #SBATCH --qos=savio_normal
    #SBATCH --time=72:00:00
    #SBATCH --nodes=1
    #SBATCH --gres=gpu:4
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=64gb
    #SBATCH --output=output.log

    # Point to where Python is installed
    PYTHON_DIR=$HOME/programs/miniconda3

    # Point to where a script should run
    SCRIPTS_DIR=$(dirname $PWD)/scripts

    # Directory of log files
    LOG_DIR=$PWD

    # Load modules
    module load cuda/11.2

    # Export OpenMP variables
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

    # Run the script
    $PYTHON_DIR/bin/python ${SCRIPTS_DIR}/script.py > ${LOG_DIR}/output.txt

In the above job file, modify ``--partition``, ``--account``, and ``--qos`` according to your user account allowance on the cluster.
