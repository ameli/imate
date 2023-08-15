.. _interpolation:

Interpolation of Affine Matrix Functions
****************************************

One of the unique and novel features of |project| is the ability to interpolate the trace of the arbitrary functions of the affine matrix function :math:`t \mapsto \mathbf{A} + t \mathbf{B}`. Such an affine matrix function appears in variety of optimization formulations in machine learning. Often in these applications, the hyperparameter :math:`t` has to be tuned. To this end, the optimization scheme should compute

.. math::

    t \mapsto \mathrm{trace} \left(f(\mathbf{A} + t \mathbf{B}) \right),

for a large number of input hyperparameter :math:`t \in \mathbb{R}`. See common examples of the function :math:`f` in :ref:`Overview <overview>`.

Instead of directly computing the above function for every :math:`t`, |project| can interpolate the above function for a wide range of :math:`t` with a high accuracy with only a handful number of evaluation of the above function. This solution can enhance the processing time of an optimization scheme by several orders of magnitude with only less than :math:`1 \%` error.

Test Description
================

The goal of the following numerical experiments is to interpolate the functions

.. math::
    :label: afm-logdet
    
    t \mapsto \log \det (\mathbf{M} + t \mathbf{I}),

and

.. math::
    :label: afm-traceinv
    
    t \mapsto \mathrm{trace}(\mathbf{A} + t \mathbf{I})^{-1},

for a wide range of :math:`t \in \mathbb{R}_{> 0}` where :math:`\mathbf{A}` and :math:`\mathbf{B}` are symmetric and positive-definite.

Algorithms
----------

The following Algorithms were tested on Intel® Xeon CPU E5-2670 v3  with 24 threads.

.. glossary::

    1. Cholesky Decomposition

        This method is implemented by the following functions:

        * :ref:`imate.logdet.cholesky` to compute :math:numref:`logdet3`.

        * :ref:`imate.traceinv.cholesky` to compute :math:numref:`traceinv3`.

        The complexity of computing :math:numref:`logdet3` for matrices obtained from 1D, 2D, and 3D grids are respectively :math:`\mathcal{O}(n)`, :math:`\mathcal{O}(n^{\frac{3}{2}})`, and :math:`\mathcal{O}(n^2)` where :math:`n` is the matrix size. The complexity of computing :math:numref:`traceinv3` for sparse matrices is :math:`\mathcal{O}(\rho n^2)` where :math:`\rho` is the sparse matrix density.

    2. Hutchinson Algorithm

        This method is only applied to :math:numref:`traceinv3` and implemented by :ref:`imate.traceinv.hutchinson` function. The complexity of this method is:

        .. math::
            :label: comp-hutch

            \mathcal{O}(\mathrm{nnz}(\mathbf{A})s),

        where :math:`s` is the number of Monte-Carlo iterations in the algorithm and :math:`\rho` is the sparse matrix density. In this experiment, :math:`s = 80`.

    3. Stochastic Lanczos Quadrature Algorithm

        This method is implemented by:

        * :ref:`imate.logdet.cholesky` to compute :math:numref:`logdet3`.
        * :ref:`imate.traceinv.cholesky` to compute :math:numref:`traceinv3`.

        The complexity of this method is:

        .. math::
            :label: comp-slq

            \mathcal{O} \left( (\mathrm{nnz}(\mathbf{A}) l + n l^2) s \right),

        where :math:`l` is the number of Lanczos iterations, and :math:`s` is the number of Monte-Carlo iterations.  The numerical experiment is performed with :math:`l=80` and :math:`s=200`. 




Interpolating Log-Determinant
=============================

.. image:: ../_static/images/performance/affine_matrix_function_logdet.png
   :align: center
   :class: custom-dark

Interpolating Trace of Inverse
==============================

.. image:: ../_static/images/performance/affine_matrix_function_traceinv.png
   :align: center
   :class: custom-dark

How to Reproduce Results
========================

Run Locally
-----------

Run |affine_matrix_py|_ as follows:

   .. prompt:: bash
  
       cd /imate/benchmark/scripts
       python ./affine_matrix_function.py -f logdet       # for log-determinant
       python ./affine_matrix_function.py -f logdet -g    # for log-determinanrt on Gram matrix
       python ./affine_matrix_function.py -f traceinv     # for trace of inverse
       python ./affine_matrix_function.py -f traceinv -g  # for trace of inverse of Gram matrix

Submit to Cluster with SLURM
----------------------------

Submit |jobfile_affine_matrix|_ by

   .. prompt:: bash
  
       cd /imate/benchmark/jobfiles
       sbatch ./jobfile_affine_matrix_function.sh

Plot Results
------------

Run |notebook_affine_matrix|_ to generate plots. This notebook stores the plots as `svg` files in |svg_plots|_.
    
.. |affine_matrix_py| replace:: ``/imate/benchmark/scripts/affine_matrix_function.py``
.. _affine_matrix_py: https://github.com/ameli/imate/blob/main/benchmark/scripts/affine_matrix_function.py

.. |jobfile_affine_matrix| replace:: ``/imate/benchmark/scripts/jobfile_affine_matrix_function.sh``
.. _jobfile_affine_matrix: https://github.com/ameli/imate/blob/main/benchmark/jobfiles/jobfile_affine_matrix_function.sh

.. |notebook_affine_matrix| replace:: ``/imate/benchmark/notebooks/plot_affine_matrix_function.ipynb``
.. _notebook_affine_matrix: https://github.com/ameli/imate/blob/main/benchmark/notebooks/plot_affine_matrix_function.ipynb

.. |svg_plots| replace:: ``/imate/benchmark/svg_plots/``
.. _svg_plots: https://github.com/ameli/imate/blob/main/benchmark/svg_plots
