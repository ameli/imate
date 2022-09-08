.. _perf-gpu:

Performance on GPU Farm
***********************

The performance of |project| is tested on multi-GPU devices and the results are compared with the performance on a CPU cluster.

Task
====

In these tests, the trace of inverse of matrix :math:`\mathbf{A}`, *i.e.*,

.. math::
    :label: traceinv
    
    \mathrm{trace}(\mathbf{A}^{-1}),

is computed on several benchmark matrices that are symmetric and positive-definite. To compute :math:numref:`traceinv`, the stochastic Lanczos quadrature (SLQ) algorithm is employed. SLQ is a randomized algorithm to compute the trace of matrix functions, such as the inverse of matrix.

Benchmark Matrices
==================

The following table shows the name of benchmark matrices used in the test, which are chosen from `SparseSuite Matrix Collection <https://sparse.tamu.edu>`_. The matrices are generated for numerical simulation in practical applications. The selected matrices below are all symmetric positive-definite, which is a requirement for the SLQ method. The `NNZ` in the third column of the table indicates the number of non-zero elements of the sparse matrix.

.. table::
   :class: right2 right3

   =================  =========  ===========  ============================
   Matrix Name             Size  NNZ          Application
   =================  =========  ===========  ============================
   |nos5|_                  468        5,172  Structural Problem
   |mhd4800b|_            4,800       27,520  Electromagnetics
   |bodyy6|_             19,366      134,208  Structural Problem
   |G2_circuit|_        150,102      726,674  Circuit Simulation
   |parabolic_fem|_     525,825    3,674,625  Computational Fluid Dynamics
   |StocF-1465|_      1,465,137   21,005,389  Computational Fluid Dynamics 
   |Bump_2911|_       2,911,419  127,729,899  Structural Problem
   |Queen_4147|_      4,147,110  329,499,284  Structural Problem
   =================  =========  ===========  ============================

.. note::

    The matrix |Queen_4147|_ is exceptionally large, beyound the matrix sizes that most numerical software could handle. Particularly, a 32-bit ``int`` type integers cannot hold the column and row indexing of this matrix beyound :math:`2^{31}-1`. Rather, a 32-bit ``unsigned int`` should be used. |project| can handle such massive data, however, it has to be recompiled with ``UNSIGNED_LONG_INT=1`` flag to use larger integer space for matrix indices to :math:`2^{32}-1` limit. To do so, change ``UNSIGNED_LONG_INT=1`` in |def-use-cblas-2|_ file, or in terminal set

    .. tab-set::

        .. tab-item:: UNIX
            :sync: unix

            .. prompt:: bash

                export UNSIGNED_LONG_INT=1

        .. tab-item:: Windows (Powershell)
            :sync: win

            .. prompt:: powershell

                $env:export UNSIGNED_LONG_INT = "1"

    Then, recompile |project|. See :ref:`Compile from Source <compile-source>`.

.. |def-use-cblas-2|  replace:: ``/imate/imate/_definitions/definition.h``
.. _def-use-cblas-2: https://github.com/ameli/imate/blob/main/imate/_definitions/definitions.h#L57
.. |nos5| replace:: ``nos5``
.. _nos5: https://sparse.tamu.edu/HB/nos5
.. |mhd4800b| replace:: ``mhd4800b``
.. _mhd4800b: https://sparse.tamu.edu/Bai/mhd4800b
.. |bodyy6| replace:: ``bodyy6``
.. _bodyy6: https://sparse.tamu.edu/Pothen/bodyy6
.. |G2_circuit| replace:: ``G2_circuit``
.. _G2_circuit: https://sparse.tamu.edu/AMD/G2_circuit
.. |parabolic_fem| replace:: ``parabolic_fem``
.. _parabolic_fem: https://sparse.tamu.edu/Wissgott/parabolic_fem
.. |StocF-1465| replace:: ``StocF-1465``
.. _StocF-1465: https://sparse.tamu.edu/Janna/StocF-1465
.. |Bump_2911| replace:: ``Bump_2911``
.. _Bump_2911: https://sparse.tamu.edu/Janna/Bump_2911
.. |Queen_4147| replace:: ``Queen_4147``
.. _Queen_4147: https://sparse.tamu.edu/Janna/Queen_4147


Results
=======

.. image:: ../_static/images/performance/benchmark_speed_time.png
   :align: center
   :height: 375
   :class: custom-dark

.. image:: ../_static/images/performance/benchmark_speed_accuracy.png
   :align: center
   :height: 375
   :class: custom-dark

.. image:: ../_static/images/performance/benchmark_speed_cores.png
   :align: center
   :height: 375
   :class: custom-dark

Considerations
==============

**Parameters:**

* In SLQ and Hutchinson methods, `min_num_samples` and `max_num_samples` are fixed to 200.
* In SLQ method, `lanczos_degree` is 80.
* All 24 cores of Intel Xeon E5-2670 v3 processor are used for all algorithms.
* The GPU results are obtained by GTX-3090 GPUs.

**Notes:**

* CPU computations have all 32, 64, and 128 but data types.
* GPU computations have ony 32, and 64 data types.


How to Reproduce Results
========================

Run the code both on CPU and GPU as follows.

Run on Local CPU
----------------

Run ``/benchmark/scripts/benchmark_speed.py`` by

::

    cd /benchmark/scripts
    python ./benchmark_speed.py -c

where ``-c`` runs the code on CPU on all 32-bit, 64-bit, and 128-bit data types.

The output is stored in `/benchnmark/pickle_results/benchmark_results_cpu.pickle`. Rename the results on CPU to

    /benchmark/pickle_results/benchmark_results_cpu_2670v3.pickle

Run on CPU Cluster
------------------

To run this script on a cluster with SLURM:

    cd jobfiles
    sbatch jobfile_benchmark_speed_cpu.sh

When submitting the jobs, make sure that the cpu is the same as the previous runs. For isnatnce, nodes on savio2 cluster between `n027` and `n150` are *Intel Xeon E5-2670 v3*.

Run on Local GPU
----------------

Run ``/benchmark/scripts/benchmark_speed.py`` by

::

    cd /benchmark/scripts
    module load cuda/11.2
    python ./benchmark_speed.py -g

where ``-g`` runs the code on CPU on all 32-bit, 64-bit, and 128-bit data types.

The output is stored in ``/benchnmark/pickle_results/benchmark_results_cpu.pickle``. Rename the results to

    /benchmark/pickle_results/benchmark_results_gpu_3090.pickle

Run on GPU Cluster
------------------

To run this script on a cluster with SLURM:

::

    cd jobfiles
    sbatch jobfile_benchmark_speed_gpu.sh

When submitting the jobs, make sure that the cpu is the same as the previous runs. For isnatnce, nodes on savio2 cluster between `n027` and `n150` are *Intel Xeon E5-2670 v3*.

Run on Cluster GPU Using Docker Image
--------------------------------------

On a virtual machine, it is better to install |project| using the it's `docker image <https://hub.docker.com/repository/docker/sameli/imate>`_.

* `Install docker <https://docs.docker.com/engine/install/ubuntu/>`_
* Set docker `without sudo <https://docs.docker.com/engine/install/linux-postinstall/>`_ password.
* Install `imate`:

  ::

      docker pull sameli/imate

* Follow instruction for `using docker's host's GPU <https://hub.docker.com/repository/docker/sameli/imate>`_.
