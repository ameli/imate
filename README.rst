|travis-devel| |codecov-devel| |licence| |format| |pypi| |implementation| |pyversions|

TraceInv
========

A python package to compute the trace of the inverse of a matrix or a linear matrix function.

For users
    * `PyPi package <https://pypi.org/project/TraceInv/>`_
    * `Source code <https://github.com/ameli/TraceInv>`_
    * `Documentation <https://ameli.github.io/TraceInv/index.html>`_

For developers
    * `API <https://ameli.github.io/TraceInv/_modules/modules.html>`_
    * `Travis-CI <https://travis-ci.com/github/ameli/TraceInv>`_
    * `Codecov <https://codecov.io/gh/ameli/TraceInv>`_

.. Status
.. ------
..
.. +------------+--------------------------+
.. | Platform   | CI Status                |
.. +============+==========================+
.. | Linux      | |travis-devel-linux|     |
.. +------------+--------------------------+
.. | OSX        | |travis-devel-osx|       |
.. +------------+--------------------------+
.. | Windows    | |travis-devel-windows|   |
.. +------------+--------------------------+

Description
-----------

This package computes the trace of inverse of matrices for two purposes:

1. Fixed Matrix
~~~~~~~~~~~~~~~

For a given generic invertible matrix |image01|, this package can compute |image02| by either of these three methods:

1. *Cholesky method*: This results the exact computation of the trace.
2. *Hutchinson's method*: This is a randomized approximation and suitable for large or implicit matrices.
3. *Stochastic Lanczos Quadrature Method*: This is a randomized approximation and suitable for large or implicit matrices.

2. Linear Matrix Function
~~~~~~~~~~~~~~~~~~~~~~~~~

Consider two matrices |image01| and |image03| and a range of real number |image04| such that |image05| is invertible. Then, this package can interpolate the function

.. image:: https://latex.codecogs.com/svg.latex?t\mapsto\mathrm{trace}\left((\mathbf{A}+t\mathbf{B})^{-1}\right)
       :align: center

by the method described in [Ameli-2020]_. The above function is featured in a wide range of applications in statistics and machine learning, particularly, in model selection and optimizing hyperparameters with gradient-based maximum likelihood methods.


Install
-------

To run ``TraceInv``, the prerequisite packages are ``numpy``, ``scipy`` are required. Also, to run the examples provided in ``/examples``, the packages ``matplotlib`` and ``seaborn`` are required. These prerequisite packages will be installed automatically by installing ``TraceInv``. You may install ``TraceInv`` by either of the following ways:

- Method 1: install from the package available at `PyPi <https://pypi.org/project/TraceInv>`_:

  ::

    python -m pip install TraceInv


- Method 2: download the source code and install from the source code by:

  ::

    git clone https://github.com/ameli/TraceInv.git
    cd TraceInv
    python -m pip install -e .


Additional Installations (*Optional*)
-------------------------------------

Installing the additional packages ``ray`` and ``scikit-sparse`` can improve the performance of ``TraceInv``, but they are not required. 

1. Installing ``ray``
~~~~~~~~~~~~~~~~~~~~~

If you want to run the examples provided in ``/examples``, you may install the ``ray`` package to leverage the parallel processing used in generate large sparse matrices. Installing ``ray`` is optional as the examples can still produce results without it. Install ``ray`` package by:

::

    python -m pip install ray

2. Installing ``scikit-sparse``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``TraceInv`` package, one of the methods to compute the trace of a matrix is by the *Cholesky decomposition*. If the input matrix is *sparse*, the Cholesky decomposition is computed with either of these packages below, whichever is available:

* ``scikit-sparse``: If this is available, ``TraceInv`` uses this package by *default*. Note that ``scikit-sparse`` is not available by installing ``TraceInv`` only, rather, it should be installed separately (see instructions below).
* ``scipy``: If ``scikit-sparse`` is not available, the ``scipy`` package is used instead. The ``scipy`` package is readily available by installing ``TraceInv``.

To install ``scikit-sparse``, follow the two steps below.

1. Install `Suite Sarse <https://people.engr.tamu.edu/davis/suitesparse.html>`_ as described below depending on the operating system.

   + In *Linux*, install ``libsuitesparse-dev`` package, such as for a few Linux distros below:

     + Install with ``apt`` (in Debian, Ubuntu, Mint)

       ::

         sudo apt install libsuitesparse-dev  

     + Or, install with ``yum`` (in Redhat, Fedora, CentOS)

       ::

         sudo yum install libsuitesparse-dev  

     + or, install with ``pacman`` (in Arch Linux)

       ::

         sudo pacman -S install libsuitesparse-dev  

   + In *macOS*, install with ``brew``:

       ::

         sudo brew install libsuitesparse-dev


   + Alternatively, if you are using *Anaconda* for python distribution (on either of the operating systems), install Suite Sparse by:

       ::

         sudo conda install -c conda-forge suitesparse

2. Install ``scikit-sparse`` package:

   ::

       python -m pip install scikit-sparse


Basic Usage
-----------

For a complete set of options, see the documentation. A minimalistic examples for both fixed matrix and matrix function are as follows.

1. For a Fixed Matrix
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from TraceInv import GenerateMatrix
    from TraceInv import ComputeTraceOfInverse
    
    # Generate a symmetric positive-definite matrix
    A = GenerateMatrix(NumPoints=20)
    
    # Compute trace of inverse
    trace = ComputeTraceOfInverse(A,method='cholesky')

1. For a Linear Matrix Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from TraceInv import GenerateMatrix
    from TraceInv import InterpolateTraceOfInverse
    
    # Generate a symmetric positive-definite matrix
    A = GenerateMatrix(NumPoints=20)
    
    # Create an interpolating TraceInv object
    TI = InterpolateTraceOfInverse(A,InterpolatingMethod='RMBF')
    
    # Interpolate A+tI
    t = 4e-1
    trace = TI.Interpolate(t)

Citation
--------

.. [Ameli-2020] Ameli, S., and Shadden. S. C. (2020). Interpolating the Trace of the Inverse of Matrix **A** + t **B**. `arXiv:2009.07385 <https://arxiv.org/abs/2009.07385>`__ [math.NA]

::

    @misc{AMELI-2020,
        title={Interpolating the Trace of the Inverse of Matrix $\mathbf{A} + t \mathbf{B}$},
        author={Siavash Ameli and Shawn C. Shadden},
        year={2020},
        month = sep,
        eid = {arXiv:2009.07385},
        eprint={2009.07385},
        archivePrefix={arXiv},
        primaryClass={math.NA},
        howpublished={\emph{arXiv}: 2009.07385 [math.NA]},
    }

.. |travis-devel| image:: https://img.shields.io/travis/com/ameli/TraceInv
   :target: https://travis-ci.com/github/ameli/TraceInv
.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/TraceInv
   :target: https://codecov.io/gh/ameli/TraceInv
.. |licence| image:: https://img.shields.io/github/license/ameli/TraceInv
   :target: https://opensource.org/licenses/MIT
.. |travis-devel-linux| image:: https://img.shields.io/travis/com/ameli/TraceInv?env=BADGE=linux&label=build&branch=master
   :target: https://travis-ci.com/github/ameli/TraceInv
.. |travis-devel-osx| image:: https://img.shields.io/travis/com/ameli/TraceInv?env=BADGE=osx&label=build&branch=master
   :target: https://travis-ci.com/github/ameli/TraceInv
.. |travis-devel-windows| image:: https://img.shields.io/travis/com/ameli/TraceInv?env=BADGE=windows&label=build&branch=master
   :target: https://travis-ci.com/github/ameli/TraceInv
.. |implementation| image:: https://img.shields.io/pypi/implementation/TraceInv
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/TraceInv
.. |format| image:: https://img.shields.io/pypi/format/TraceInv
.. |pypi| image:: https://img.shields.io/pypi/v/TraceInv

.. |image01| image:: https://latex.codecogs.com/svg.latex?\mathbf{A}
.. |image02| image:: https://latex.codecogs.com/svg.latex?\mathrm{trace}(\mathbf{A}^{-1})
.. |image03| image:: https://latex.codecogs.com/svg.latex?\mathbf{B}
.. |image04| image:: https://latex.codecogs.com/svg.latex?t\in&space;[t_0,t_1]
.. |image05| image:: https://latex.codecogs.com/svg.latex?\mathbf{A}+t\mathbf{B}
