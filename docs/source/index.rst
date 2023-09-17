.. module:: imate

|project| Documentation
***********************

|deploy-docs|

|project|, short for **I**\ mplicit **Ma**\ trix **T**\ race **E**\ stimator, is a modular and high-performance C++/CUDA library distributed as a Python package that provides scalable randomized algorithms for the computationally expensive matrix functions in machine learning.

.. .. toctree::
    :maxdepth: 1

    old/ComputeLogDeterminant.rst
    old/ComputeTraceOfInverse.rst
    old/examples.rst
    old/generate_matrix.rst
    old/InterpolateTraceOfInverse.rst
    old/introduction.rst

.. grid:: 4

    .. grid-item-card:: GitHub
        :link: https://github.com/ameli/imate
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: PyPI
        :link: https://pypi.org/project/imate/
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: Anaconda Cloud
        :link: https://anaconda.org/s-ameli/imate
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: Docker Hub
        :link: https://hub.docker.com/r/sameli/imate
        :text-align: center
        :class-card: custom-card-link

.. grid:: 4

    .. grid-item-card:: Install
        :link: install
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: Tutorials
        :link: index_tutorials
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: API reference
        :link: api
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: Performance
        :link: index_performance
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

Overview
========

To learn more about |project| functionality, see:

.. toctree::

    overview

Supported Platforms
===================

Successful installation and tests performed on the following operating systems, architectures, and Python and `PyPy <https://www.pypy.org/>`_ versions:

.. |y| unicode:: U+2714
.. |n| unicode:: U+2716

+----------+--------+--------+-------+-------+-------+-------+-------+-------+-----------------+
| Platform | Arch   | Device | Python Version        | PyPy Version          | Continuous      |
+          |        +        +-------+-------+-------+-------+-------+-------+ Integration     +
|          |        |        |  3.9  |  3.10 |  3.11 |  3.8  |  3.9  |  3.10 |                 |
+==========+========+========+=======+=======+=======+=======+=======+=======+=================+
| Linux    | X86-64 | CPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  | |build-linux|   |
+          +        +--------+-------+-------+-------+-------+-------+-------+                 +
|          |        | GPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |                 |
+----------+--------+--------+-------+-------+-------+-------+-------+-------+-----------------+
| macOS    | X86-64 | CPU    |  |y|  |  |y|  |  |y|  |  |n|  |  |n|  |  |n|  | |build-macos|   |
+          +        +--------+-------+-------+-------+-------+-------+-------+                 +
|          |        | GPU    |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |                 |
+----------+--------+--------+-------+-------+-------+-------+-------+-------+-----------------+
| Windows  | X86-64 | CPU    |  |y|  |  |y|  |  |y|  |  |n|  |  |n|  |  |n|  | |build-windows| |
+          +        +--------+-------+-------+-------+-------+-------+-------+                 +
|          |        | GPU    |  |y|  |  |y|  |  |y|  |  |n|  |  |n|  |  |n|  |                 |
+----------+--------+--------+-------+-------+-------+-------+-------+-------+-----------------+

.. |build-linux| image:: https://img.shields.io/github/actions/workflow/status/ameli/imate/build-linux.yml
   :target: https://github.com/ameli/imate/actions?query=workflow%3Abuild-linux 
.. |build-macos| image:: https://img.shields.io/github/actions/workflow/status/ameli/imate/build-macos.yml
   :target: https://github.com/ameli/imate/actions?query=workflow%3Abuild-macos
.. |build-windows| image:: https://img.shields.io/github/actions/workflow/status/ameli/imate/build-windows.yml
   :target: https://github.com/ameli/imate/actions?query=workflow%3Abuild-windows

Python wheels for |project| for all supported platforms and versions in the above are available through `PyPI <https://pypi.org/project/imate/>`_ and `Anaconda Cloud <https://anaconda.org/s-ameli/imate>`_. If you need |project| on other platforms, architectures, and Python or PyPy versions, `raise an issue <https://github.com/ameli/imate/issues>`_ on GitHub and we build its Python Wheel for you.

Install
=======

|conda-downloads|

.. grid:: 2

    .. grid-item-card:: 

        Install with ``pip`` from `PyPI <https://pypi.org/project/imate/>`_:

        .. prompt:: bash
            
            pip install imate

    .. grid-item-card::

        Install with ``conda`` from `Anaconda Cloud <https://anaconda.org/s-ameli/imate>`_:

        .. prompt:: bash
            
            conda install -c s-ameli imate

For complete installation guide, see:

.. toctree::
    :maxdepth: 2

    Install <install/install>

Docker
======

|docker-pull| |deploy-docker|

The docker image comes with a pre-installed |project|, an NVIDIA graphic driver, and a compatible version of CUDA Toolkit libraries.

.. grid:: 1

    .. grid-item-card::

        Pull docker image from `Docker Hub <https://hub.docker.com/r/sameli/imate>`_:

        .. prompt:: bash
            
            docker pull sameli/imate

For a complete guide, see:

.. toctree::
    :maxdepth: 2

    Docker <docker/docker>

GPU
===

|project| can run on CUDA-capable **multi**-GPU devices, which can be set up in several ways. Using the **docker container** is the easiest way to run |project| on GPU devices. For a comprehensive guide, see:

.. toctree::
    :maxdepth: 2

    GPU <gpu/gpu>

The supported GPU micro-architectures and CUDA version are as follows:

+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| Version \\ Arch | Fermi   | Kepler  | Maxwell | Pascal  | Volta   | Turing  | Ampere  | Hopper |
+=================+=========+=========+=========+=========+=========+=========+=========+========+
| CUDA 9          |   |n|   |   |n|   |   |n|   |   |n|   |   |n|   |   |n|   |   |n|   |   |n|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| CUDA 10         |   |n|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| CUDA 11         |   |n|   |   |n|   |   |n|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| CUDA 12         |   |n|   |   |n|   |   |n|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+

.. _index_tutorials:

Tutorials
=========

|binder|

.. toctree::
    :maxdepth: 1

    Jupyter Notebook <notebooks/quick_start.ipynb>

Launch `online interactive notebook <https://mybinder.org/v2/gh/ameli/glearn/HEAD?filepath=notebooks%2Fquick_start.ipynb>`_ with Binder.

API Reference
=============

Check the list of functions, classes, and modules of |project| with their usage, options, and examples.

.. toctree::
   :maxdepth: 2
   
   API Reference <api>

.. _index_performance:

Performance
===========

|project| is scalable to **very large matrices**. Its core library for basic linear algebraic operations is **faster than OpenBLAS**, and its **pseudo-random generator** is a hundred-fold faster than the implementation in the standard C++ library.

Read about the performance of |project| in practical applications:

.. toctree::
    :maxdepth: 1
    :hidden:

    performance <performance/performance>

.. Performance on GPU Farm <performance/gpu>
.. Performance on CPU <performance/scalability>
.. Comparison With and Without OpenBLAS <performance/openblas>
.. Interpolation of Affine Matrix Functions <performance/interpolation>


.. grid:: 2

   .. grid-item-card:: :ref:`Performance on GPU Farm <perf-gpu>`
        :link: perf-gpu
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link-2

        .. image:: _static/images/performance/benchmark_speed_time.png
           :align: center
           :width: 320px
           :height: 200px
           :class: custom-dark

   .. grid-item-card:: :ref:`Comparison of Randomized Algorithms <perf-algorithms>`
        :link: perf-algorithms
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link-2

        .. image:: _static/images/performance/compare_methods_practical_matrix_logdet_time.png
           :align: center
           :width: 320px
           :height: 200px
           :class: custom-dark

.. grid:: 2

   .. grid-item-card:: :ref:`Comparison With and Without OpenBLAS <perf-openblas>` 
        :link: perf-openblas
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link-2

        .. image:: _static/images/performance/benchmark_openblas_sparse_time.png
           :align: center
           :width: 320px
           :height: 200px
           :class: custom-dark


   .. grid-item-card:: :ref:`Interpolation of Affine Matrix Functions <interpolation>` 
        :link: interpolation
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link-2

        .. image:: _static/images/performance/affine_matrix_function_logdet.png
           :align: center
           :width: 320px
           :height: 200px
           :class: custom-dark

Features
========

* Matrices can be dense or sparse (`CSR` or `CSC` format), with 32-bit, 64-bit, or 128-bit data types, and stored either by row-ordering (`C` style) or column-ordering (`Fortran` style).
* Matrices can be **linear operators** with parameters (see :class:`imate.Matrix` and :class:`imate.AffineMatrixFunction` classes).
* **Randomized algorithms** using Hutchinson and stochastic Lanczos quadrature algorithms (see :ref:`Overview <overview>`)
* Novel method to **interpolate** matrix functions. See :ref:`Interpolation of Affine Matrix Functions <interpolation>`.
* Parallel processing both on **shared memory** and CUDA Capable **multi-GPU** devices.

Technical Notes
===============

|tokei-2| |languages|

The core of |project|, which is implemented in C++ and NVIDIA CUDA framework, is a standalone modular library for high-performance low-level algebraic operations on linear operators (including matrices and affine matrix functions). This library provides a unified interface for computations on both CPU and GPU, a unified interface for dense and sparse matrices, a unified container for various data types, and fully automatic memory management and data transfer between CPU and GPU devices on demand. This library can be employed independently for projects other than |project|. The Doxygen generated reference of `C++/CUDA Classes and Namespaces <doxygen/html/annotated.html>`_ of |project| is available for developers.

The front-end interface of |project| is implemented in Cython and Python (see Python :ref:`API Reference <api>` for end-users).

Some notable implementation techniques used to develop |project| are:

* Polymorphic and curiously recurring template pattern programming (CRTP) technique.
* OS-independent customized `dynamic loading` of CUDA libraries (as opposed to dynamic linking).
* Static dispatching enables executing |project| with and without CUDA on the user's machine with the same pre-compiled |project| installation.
* Completely `GIL <https://en.wikipedia.org/wiki/Global_interpreter_lock>`_-*free* Cython implementation.
* Providing `manylinux wheels <https://pypi.org/project/imate/#files>`_ build upon customized docker images with CUDA support available on DockerHub:
     
  * `manylinux CUDA 10.2 <https://hub.docker.com/r/sameli/manylinux2014_x86_64_cuda_10.2>`_
  * `manylinux CUDA 11.7 <https://hub.docker.com/r/sameli/manylinux2014_x86_64_cuda_11.7>`_
  * `manylinux CUDA 11.8 <https://hub.docker.com/r/sameli/manylinux2014_x86_64_cuda_11.8>`_
  * `manylinux CUDA 12.0 <https://hub.docker.com/r/sameli/manylinux2014_x86_64_cuda_12.0>`_
  * `manylinux CUDA 12.2 <https://hub.docker.com/r/sameli/manylinux2014_x86_64_cuda_12.2>`_

How to Contribute
=================

We welcome contributions via `GitHub's pull request <https://github.com/ameli/imate/pulls>`_. If you do not feel comfortable modifying the code, we also welcome feature requests and bug reports as `GitHub issues <https://github.com/ameli/imate/issues>`_.

Publications
============

For information on how to cite |project|, publications, and software packages that used |project|, see:

.. toctree::
    :maxdepth: 2

    Publications <cite>

License
=======

|license|

This project uses a `BSD 3-clause license <https://github.com/ameli/imate/blob/main/LICENSE.txt>`_, in hopes that it will be accessible to most projects. If you require a different license, please raise an `issue <https://github.com/ameli/imate/issues>`_ and we will consider a dual license.

Related Projects
================

.. grid:: 3

   .. grid-item-card:: |glearn-light| |glearn-dark|
       :link: https://ameli.github.io/glearn/index.html
       :text-align: center
       :class-card: custom-card-link
   
       A high-performance python package for machine learning using Gaussian process regression.

   .. grid-item-card:: |detkit-light| |detkit-dark|
       :link: https://ameli.github.io/detkit/index.html
       :text-align: center
       :class-card: custom-card-link

       A python package for matrix determinant functions used in machine learning.

   .. grid-item-card:: |ortho-light| |ortho-dark|
      :link: https://ameli.github.io/ortho/index.html
      :text-align: center
      :class-card: custom-card-link

      A python package to generate orthogonal basis functions for matrix functions interpolation.

.. |deploy-docs| image:: https://img.shields.io/github/actions/workflow/status/ameli/imate/deploy-docs.yml?label=docs
   :target: https://github.com/ameli/imate/actions?query=workflow%3Adeploy-docs
.. |deploy-docker| image:: https://img.shields.io/github/actions/workflow/status/ameli/imate/deploy-docker.yml?label=build%20docker
   :target: https://github.com/ameli/imate/actions?query=workflow%3Adeploy-docker
.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/imate
   :target: https://codecov.io/gh/ameli/imate
.. |license| image:: https://img.shields.io/github/license/ameli/imate
   :target: https://opensource.org/licenses/BSD-3-Clause
.. |implementation| image:: https://img.shields.io/pypi/implementation/imate
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/imate
.. |format| image:: https://img.shields.io/pypi/format/imate
.. |pypi| image:: https://img.shields.io/pypi/v/imate
.. |conda| image:: https://anaconda.org/s-ameli/traceinv/badges/installer/conda.svg
   :target: https://anaconda.org/s-ameli/traceinv
.. |platforms| image:: https://img.shields.io/conda/pn/s-ameli/traceinv?color=orange?label=platforms
   :target: https://anaconda.org/s-ameli/traceinv
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/traceinv
   :target: https://anaconda.org/s-ameli/traceinv
.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/ameli/imate/HEAD?filepath=notebooks%2Fquick_start.ipynb
.. |conda-downloads| image:: https://img.shields.io/conda/dn/s-ameli/imate
   :target: https://anaconda.org/s-ameli/imate
.. |tokei| image:: https://tokei.rs/b1/github/ameli/imate?category=lines
   :target: https://github.com/ameli/imate
.. |tokei-2| image:: https://img.shields.io/badge/code%20lines-72.8k-blue
   :target: https://github.com/ameli/imate
.. |languages| image:: https://img.shields.io/github/languages/count/ameli/imate
   :target: https://github.com/ameli/imate
.. |docker-pull| image:: https://img.shields.io/docker/pulls/sameli/imate?color=green&label=downloads
   :target: https://hub.docker.com/r/sameli/imate
.. |glearn-light| image:: _static/images/icons/logo-glearn-light.svg
   :height: 30
   :class: only-light
.. |glearn-dark| image:: _static/images/icons/logo-glearn-dark.svg
   :height: 30
   :class: only-dark
.. |detkit-light| image:: _static/images/icons/logo-detkit-light.svg
   :height: 27
   :class: only-light
.. |detkit-dark| image:: _static/images/icons/logo-detkit-dark.svg
   :height: 27
   :class: only-dark
.. |ortho-light| image:: _static/images/icons/logo-ortho-light.svg
   :height: 24
   :class: only-light
.. |ortho-dark| image:: _static/images/icons/logo-ortho-dark.svg
   :height: 24
   :class: only-dark
.. |special_functions-light| image:: _static/images/icons/logo-special-functions-light.svg
   :height: 35
   :class: only-light
.. |special_functions-dark| image:: _static/images/icons/logo-special-functions-dark.svg
   :height: 35
   :class: only-dark
