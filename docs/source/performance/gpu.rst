.. _perf-gpu:

Performance on GPU Farm
***********************

The performance of |project| is tested on multi-GPU devices and the results are compared with the performance on a CPU cluster.

Test Description
================

The performance test computes the quantity

.. math::
    :label: traceinv
    
    \mathrm{trace}(\mathbf{A}^{-1}),

where :math:`\mathbf{A}` is symmetric and positive-definite. The above quantity is a computationally expensive expression that frequently appears in the Jacobian and Hessian of likelihood functions in machine learning.

Algorithm
---------

To compute :math:numref:`traceinv`, the stochastic Lanczos quadrature (SLQ) algorithm is employed. The complexity of this algorithm is

.. math::
   :label: complexity1

    \mathcal{O} \left( (\mathrm{nnz}(\mathbf{A}) l + n l^2) s \right),

where :math:`n` is the matrix size, :math:`\mathrm{nnz}(\mathbf{A})` is the number of nonzero elements of the sparse matrix :math:`\mathbf{A}`, :math:`l` is the number of Lanczos iterations, and :math:`s` is the number of Monte-Carlo iterations (see details in :ref:`imate.traceinv.slq`).  The numerical experiment is performed with :math:`l=80` and :math:`s=200`.

Hardware
--------

The computations were carried out on the following hardware:

* For **CPU** test: Intel(R) Xeon(R) CPU E5-2670 v3  with 24 threads.
* For **GPU** test: a cluster of eight `NVIDIA® GeForce RTX 3090 <https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/>`_ GPUs and Intel Xeon Processor (Skylake, IBRS) with 32 threads.

Benchmark Matrices
------------------

The table below shows the matrices used in the test, which are chosen from `SuiteSparse Matrix Collection <https://sparse.tamu.edu>`_ that are generated for real applications. The matrices in the table below are all symmetric positive-definite and the number of nonzero elements (nnz) of these matrices increase approximately by the factor of 5 on average.

.. table::
   :class: right2 right3

   =================  =========  ===========  ============================
   Matrix Name             Size  nnz          Application
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

Arithmetic Types
----------------

The benchmark test also examines the performance and accuracy of |project| on various arithmetic types of the matrix data. To this end, each of the above matrices were re-cast into 32-bit, 64-bit, and 128-bit floating point arithmetics. Depending on the hardware, the followings data types were tested:

* For **CPU** test: 32-bit, 64-bit and 128-bit floating point data.
* For **GPU** test: 32-bit, 64-bit floating point data.

.. note::

    Supporting 128-bit data types is one of the features if |project|, which is often not available in numerical libraries.

.. note::

    NVIDIA CUDA libraries do not support 128-data types.

Results
=======

Scalability with Data Size
--------------------------

The figure below shows the scalability by the relation between the elapsed (wall) time versus the data size.

The data size is indicated by the matrix nnz (rather than the matrix size). However, the matrix size can be found by the hollow circle marks in the figure.

* **GPU** test: 8 GPU devices were used.
* **CPU** test: 16 CPU cores were used.

.. figure:: ../_static/images/performance/benchmark_speed_time.png
   :align: center
   :height: 375
   :class: custom-dark

.. sidebar:: Scalability Exponent
   :class: custom-sidebar

    .. table::
       :class: custom-table

       +--------+---------+----------------+
       | Device |  Data   | :math:`\alpha` |
       +========+=========+================+
       | CPU    | 32-bit  |  1.08          |
       +        +---------+----------------+
       |        | 64-bit  |  0.89          |
       +        +---------+----------------+
       |        | 128-bit |  0.93          |
       +--------+---------+----------------+
       | GPU    | 32-bit  |  0.86          |
       +        +---------+----------------+
       |        | 64-bit  |  0.92          |
       +--------+---------+----------------+

The computation on GPU is advantageous over CPU at nnz larger than roughly :math:`10^{5}`. The elapsed time :math:`t` is related to the number of nonzero elements :math:`\mathrm{nnz}` by

.. math::

    t \propto \mathcal{O}((\mathrm{nnz}(\mathbf{A}))^{\alpha}),

where the exponent :math:`\alpha` for each experiment asymptotically approaches to the values shown in the table below. It can be seen that the performance is close to the theoretical complexity :math:numref:`complexity1`.


Also, the figure shows that processing 32-bit data is at most twice faster than 64-bit data on both CPU and GPU, and 64-bit data is at least twice faster than 128-bit on CPU.

Extreme Array Sizes
...................

There above results indicate |project| is highly scalable on both CPU and GPU on massive data. However, there are a number of factors that can limit the data size. For instance, hardware memory limit is one such factor. Another limiting factor is the maximum array length in bits to store the content of a sparse matrix. Interestingly, this factor is not a hardware limitation, rather, is related to the maximum integer (often 32-bit ``int`` type) to index the array (in bits) on the memory. The 128-bit format of |Queen_4147|_ matrix is indeed close to such limit. The above results show that |project| is scalable to large scales before reaching such an array size limit.

Beyond Extreme Array Sizes
..........................

|project| can be configured to handle even larger data (if one can indeed store such array of data). To do so, increase the integer space for matrix indices by changing ``UNSIGNED_LONG_INT=1`` in |def-use-cblas-2|_ file, or in terminal set

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

Floating Point Arithmetic Accuracy
----------------------------------

The error of floating point arithmetic of iterative algorithms is sensible on large data. In this test, the result of 32-bit and 64-bit data are compared with the result of 128-bit as a benchmark and shown in the figure below. The results show that both 32-bit and 64-bit data have less than :math:`0.1 \%` error relative to 128-bit data. However, for data larger than :math:`10^{7}`, the error of 32-bit data is :math:`30 \%` relative to 128-bit data whereas 64-bit data maintain :math:`0.1 \sim 1 \%` error. Because of this, 64-bit data is often a good balance between accuracy and speed.

.. image:: ../_static/images/performance/benchmark_speed_accuracy.png
   :align: center
   :height: 375
   :class: custom-dark

Scalability with Increase of GPU Devices
----------------------------------------

The scalability of |project| is examined by the increase of the number of CPU threads or GPU devices as shown in the figure below.

.. image:: ../_static/images/performance/benchmark_speed_cores.png
   :align: center
   :height: 375
   :class: custom-dark

.. raw:: html

    <br/>

.. sidebar:: Scalability Exponent
   :class: custom-sidebar

    .. table::
       :class: custom-table

       +--------+---------+---------------+
       | Device |  Data   | :math:`\beta` |
       +========+=========+===============+
       | CPU    | 32-bit  |  0.83         |
       +        +---------+---------------+
       |        | 64-bit  |  0.80         |
       +        +---------+---------------+
       |        | 128-bit |  0.76         |
       +--------+---------+---------------+
       | GPU    | 32-bit  |  0.98         |
       +        +---------+---------------+
       |        | 64-bit  |  0.96         |
       +--------+---------+---------------+

The above results correspond to only |Queen_4147|_, which is the largest matrix on the list. The performance on GPUs are over thirty-fold faster than the CPU for the same number of threads and GPU devices, although, this may not be a fair comparison. However, the performance of one GPU device is yet five times faster than 8 CPU threads.

The elapsed (wall) time, :math:`t`, can be related to the number of CPU threads or GPU devices, :math:`m`, as

.. math::

    t \propto \mathcal{O}(m^{-\beta}).

The estimated values of :math:`\beta` from the curves on the figure are shown in the table below. The speed (inverse of elapsed time) per CPU thread tend to saturate by the increase off the number of CPU threads. In contrast, the GPU results show better scalability as it maintains the linear behaviour by the increase of the number of GPU devices.

How to Reproduce Results
========================

Scripts to reproduce the above results is available 


Run Locally
-----------

Run the script |benchmark_speed_py|_ as follows.

* To test CPU:

  .. prompt:: bash
  
      cd /imate/benchmark/scripts
      python ./benchmark_speed.py -c

* To test GPU:

  .. prompt:: bash
  
      cd /imate/benchmark/scripts
      python ./benchmark_speed.py -g

Submit Job to Cluster with SLURM
--------------------------------

* The SLURM job file to submit the CPU test is available at |jobfile_speed_cpu_sh|_. Submit the job by

  .. prompt:: bash
  
      cd /imate/benchmark/jobfiles
      sbatch jobfile_benchmark_speed_cpu.sh


* The SLURM job file to submit the GPU test is available at |jobfile_speed_gpu_sh|_. Submit the job by

  .. prompt:: bash
  
      cd /imate/benchmark/jobfiles
      sbatch jobfile_benchmark_speed_gpu.sh

.. |benchmark_speed_py| replace:: ``/imate/benchmark/scripts/benchmark_speed.py``
.. _benchmark_speed_py: https://github.com/ameli/imate/blob/main/benchmark/scripts/benchmark_speed.py

.. |jobfile_speed_cpu_sh| replace:: ``/imate/benchmark/jobfiles/jobfile_benchmark_speed_cpu.sh``
.. _jobfile_speed_cpu_sh: https://github.com/ameli/imate/blob/main/benchmark/jobfiles/jobfile_benchmark_speed_cpu.sh

.. |jobfile_speed_gpu_sh| replace:: ``/imate/benchmark/jobfiles/jobfile_benchmark_speed_gpu.sh``
.. _jobfile_speed_gpu_sh: https://github.com/ameli/imate/blob/main/benchmark/jobfiles/jobfile_benchmark_speed_gpu.sh

