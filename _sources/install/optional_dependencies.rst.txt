.. _optional-dependencies:

Optional Runtime Dependencies
=============================

Runtime libraries are not required to be present during the installation of |project|. However, they may be required to be installed during running |project|.

CUDA Toolkit and NVIDIA Graphic Driver (`Optional`)
---------------------------------------------------

To use GPU devices, install NVIDIA Graphic Driver and CUDA Toolkit. See the instructions below.

* :ref:`Install NVIDIA Graphic Driver <install-graphic-driver>`.
* :ref:`Install CUDA Toolkit <install-cuda-toolkit>`.

SuiteSparse (`Optional`)
------------------------

`SuiteSarse <https://people.engr.tamu.edu/davis/suitesparse.html>`_ is a library for efficient calculations on sparse matrices. |project| does not require this library as it has its own library for sparse matrices. However, if this library is available, |project| uses it.

.. note::

    The SuiteSparse library is only used for those functions in |project| that uses the Cholesky decomposition method by passing ``method=cholesky`` argument to the functions. See :ref:`API reference for Functions <Functions>` for details. 

1. Install SuiteSparse development library by

   .. tab-set::

       .. tab-item:: Ubuntu/Debian
          :sync: ubuntu

          .. prompt:: bash

              sudo apt install libsuitesparse-dev

       .. tab-item:: CentOS 7
          :sync: centos

          .. prompt:: bash

              sudo yum install libsuitesparse-devel

       .. tab-item:: RHEL 9
          :sync: rhel

          .. prompt:: bash

              sudo dnf install libsuitesparse-devel

       .. tab-item:: macOS
          :sync: osx

          .. prompt:: bash

              sudo brew install suite-sparse

   Alternatively, if you are using *Anaconda* python distribution (on either of the operating systems), install Suite Sparse by:

   .. prompt:: bash

       sudo conda install -c conda-forge suitesparse

2. Install ``scikit-sparse`` python package:

   .. prompt:: bash
       
       python -m pip install scikit-sparse

OpenBLAS (`Optional`)
---------------------

`OpenBLAS <https://www.openblas.net/>`__ is a library for efficient dense matrix operations. |project| does not require this library as it has its own library for dense matrices. However, if you compiled |project| to use OpenBLAS (see :ref:`Compile from Source <compile-source>`), OpenBLAS library should be available at runtime.

.. note::

    A default installation of |project| through ``pip`` or ``conda`` does not use OpenBLAS, and you may skip this section.

Install OpenBLAS library by

.. tab-set::

   .. tab-item:: Ubuntu/Debian
      :sync: ubuntu

      .. prompt:: bash

            sudo apt install libopenblas-dev

   .. tab-item:: CentOS 7
      :sync: centos

      .. prompt:: bash

          sudo yum install openblas-devel

   .. tab-item:: RHEL 9
      :sync: rhel

      .. prompt:: bash

          sudo dnf install openblas-devel

   .. tab-item:: macOS
      :sync: osx

      .. prompt:: bash

          sudo brew install openblas

Alternatively, you can install OpenBLAS using ``conda``:

.. prompt:: bash

    conda install -c anaconda openblas
