********
TraceInv
********

|travis-devel| |codecov-devel| |licence| |format| |pypi| |implementation| |pyversions|

A python package to compute the trace of the inverse of a matrix or a linear matrix function.

.. For users
..     * `Documentation <https://ameli.github.io/TraceInv/index.html>`_
..     * `PyPi package <https://pypi.org/project/TraceInv/>`_
..     * `Source code <https://github.com/ameli/TraceInv>`_
..
.. For developers
..     * `API <https://ameli.github.io/TraceInv/_modules/modules.html>`_
..     * `Travis-CI <https://travis-ci.com/github/ameli/TraceInv>`_
..     * `Codecov <https://codecov.io/gh/ameli/TraceInv>`_

+------------------------------------------------------------------+-------------------------------------------------------------------+
|    For users                                                     | For developers                                                    |
+==================================================================+===================================================================+
| * `Documentation <https://ameli.github.io/TraceInv/index.html>`_ | * `API <https://ameli.github.io/TraceInv/_modules/modules.html>`_ |
| * `PyPi package <https://pypi.org/project/TraceInv/>`_           | * `Travis-CI <https://travis-ci.com/github/ameli/TraceInv>`_      |
| * `Source code <https://github.com/ameli/TraceInv>`_             | * `Codecov <https://codecov.io/gh/ameli/TraceInv>`_               |
+------------------------------------------------------------------+-------------------------------------------------------------------+

************
Introduction
************

This package computes the trace of inverse of two forms of matrices:

1. **Fixed Matrix:** For an invertible matrix |image01|, this package computes |image02| for both sparse and dense matrices.
2. **One-Parameter Affine Matrix Function:** |image05|, where |image01| and |image03| are symmetric and positive-definite matrices and ``t`` is a real parameter. This package can interpolate the function

|image06|

**Application:**
    The above function is featured in a wide range of applications in statistics and machine learning. Particular applications are in model selection and optimizing hyperparameters with gradient-based maximum likelihood methods. In such applications, computing the above function is often a computational challenge for large matrices. Often, this function is evaluated for a wide range of the parameter |image00| while |image01| and |image03| remain fixed. As such, an interpolation scheme enables fast computation of the function.

These interpolation methods are described in [Ameli-2020]_. 

.. |image00| image:: https://latex.codecogs.com/svg.latex?t
.. |image01| image:: https://latex.codecogs.com/svg.latex?\mathbf{A}
.. |image02| image:: https://latex.codecogs.com/svg.latex?\mathrm{trace}(\mathbf{A}^{-1})
.. |image03| image:: https://latex.codecogs.com/svg.latex?\mathbf{B}
.. |image04| image:: https://latex.codecogs.com/svg.latex?t\in&space;[t_0,t_1]
.. |image05| image:: https://latex.codecogs.com/svg.latex?t\mapsto\mathbf{A}+t\mathbf{B}
.. |image06| image:: https://latex.codecogs.com/svg.latex?t\mapsto\mathrm{trace}\left((\mathbf{A}+t\mathbf{B})^{-1}\right)

*******
Install
*******

=============
Prerequisites
=============

The prerequisite packages are:

* **Required:** ``numpy`` and ``scipy``.
* **Required:** ``matplotlib`` and ``seaborn``, but only required to run the `examples <https://github.com/ameli/TraceInv#examples>`_.
* **Optional:** ``ray`` and ``scikit-sparse`` can improve performance, but not required.

By installing TraceInv `below <https://github.com/ameli/TraceInv#install>`_, the *required* prerequisite packages (but not the *optional* packages) in the above will be installed automatically and no other action is needed. However, if desired, the *optional* packages should be installed `manually <https://github.com/ameli/TraceInv#install-optional-packages>`_.

================
Install TraceInv
================

Install by either of the following ways:

* **Method 1:** The recommended way is to install through the package available at `PyPi <https://pypi.org/project/TraceInv>`_:

  ::

    python -m pip install TraceInv


* **Method 2:** download the source code and install by:

  ::

    git clone https://github.com/ameli/TraceInv.git
    cd TraceInv
    python -m pip install -e .

=========================
Install Optional Packages
=========================

Installing the additional packages can improve the performance, but not required. 

---------------
Install ``ray``
---------------

::

    python -m pip install ray

When ``ray`` is needed:
    To run the `examples <https://github.com/ameli/TraceInv#examples>`_, you may install the ``ray`` package to leverage the parallel processing used in generate large sparse matrices. However, the examples can still produce results without installing ``ray``.

-------------------------
Install ``scikit-sparse``
-------------------------

1. Install `Suite Sarse <https://people.engr.tamu.edu/davis/suitesparse.html>`_, which is distributed by ``libsuitesparse-dev`` package, and can be installed with ``apt`` package manager in Linux (*Debian, Ubuntu, Mint*) by
   
   ::

       sudo apt install libsuitesparse-dev  

   Replace ``apt`` in the above with the native package manager of your operating system, such as ``yum`` for  *Redhat, Fedora, and CentOS Linux*, ``pacman`` for *Arch Linux*, and ``brew`` for *macOS*.

   Alternatively, if you are using *Anaconda* python distribution (on either of the operating systems), install Suite Sparse by:

   ::

       sudo conda install -c conda-forge suitesparse

2. Install ``scikit-sparse`` package:

   ::
       
       python -m pip install scikit-sparse

When ``scikit-sparse`` is needed:
    In ``TraceInv`` package, one of the methods to compute the trace of a matrix is by the *Cholesky decomposition*. If the input matrix is *sparse*, the Cholesky decomposition is computed using ``scikit-sparse`` if available. But if this package is not available, the ``scipy`` package is used instead.

*****
Usage
*****

The package TraceInv provides three sub-packages:

======================================  =====================================================================
Sub-Package                             Description
======================================  =====================================================================
``TraceInv.GenerateMatrix``             Generates symmetric and positive-definite matrices for test purposes.
``TraceInv.ComputeTraceOfInverse``      Computes trace of inverse for a fixed matrix.
``TraceInv.InterpolateTraceOfInverse``  Interpolates trace of inverse for a linear matrix function.
======================================  =====================================================================

The next two sections presents minimalistic examples respectively for:

1. Fixed matrix using ``ComputeTraceOfInvers`` module.
2. One-parameter affine matrix function using ``InterpolateTraceOfInverse`` module.

=========================
1. Usage for Fixed Matrix
=========================

.. code-block:: python

   >>> from TraceInv import GenerateMatrix
   >>> from TraceInv import ComputeTraceOfInverse
   
   >>> # Generate a symmetric positive-definite matrix of the shape (20**2,20**2)
   >>> A = GenerateMatrix(NumPoints=20)
   
   >>> # Compute trace of inverse
   >>> trace = ComputeTraceOfInverse(A)

In the above, the class ``GenerateMatrix`` produces a sample matrix for test purposes. 

The ``ComputeTraceOfInverse`` class in the above code employs the Cholesky method by default to compute the trace of inverse. However, the user may choose other methods given in the table below.

===================  ====================================  ==============  =============  =============
``ComputeMethod``    Description                           Matrix size     Matrix type    Results       
===================  ====================================  ==============  =============  =============
``'cholesky'``       Cholesky decomposition                small           dense, sparse  exact          
``'hutchinson'``     Hutchinson's randomized method        small or large  dense, sparse  approximation
``'SLQ'``            Stochastic Lanczos Quadrature method  small or large  dense, sparse  approximation
===================  ====================================  ==============  =============  =============  

The desired method of computation can be passed through the ``ComputeMethod`` argument when calling ``ComputeTraceOfInverse``. For instance, in the following example, we apply the *Hutchinson's randomized estimator* method:

.. code-block:: python

   >>> # Using hutchinson method with 20 Monte-Carlo iterations
   >>> trace = ComputeTraceOfInverse(A,ComputeMethod='hutchinson',NumIterations=20)

Each of the methods in the above accept some options. For instance, the Hutchinson's method accepts ``NumIterations`` argument, which sets the number of Monte-Carlo trials. To see the detailed list of all arguments for each method, see the `API <https://ameli.github.io/TraceInv/_modules/modules.html>`__ of the package.

.. _Affine-Matrix:

=================================================
2. Usage for One-Parameter Affine Matrix Function
=================================================

The module ``InterpolateTraceOfInverse`` interpolates the trace of the inverse of ``A + tB``, as shown by the example below.

.. code-block:: python
    
   >>> from TraceInv import GenerateMatrix
   >>> from TraceInv import InterpolateTraceOfInverse
   
   >>> # Generate a symmetric positive-definite matrix of the shape (20**2,20**2)
   >>> A = GenerateMatrix(NumPoints=20)
   
   >>> # Define some interpolant points
   >>> InterpolantPoints = [1e-2,1e-1,1,1e+1]
   
   >>> # Create an interpolating TraceInv object
   >>> TI = InterpolateTraceOfInverse(A,InterpolantPoints=InterpolantPoints)
   
   >>> # Interpolate A+tI at some inquiry point t
   >>> t = 4e-1
   >>> trace = TI.Interpolate(t)

In the above code, we only provided the matrix ``A`` to the module ``InterpolateTraceOfInverse``, which then it assumes ``B`` is identity matrix by default. To compute the trace of the inverse of ``A + tB`` where ``B`` is not identity matrix, pass both ``A`` and ``B`` to ``InterpolateTraceOfInverse`` as follows.

.. code-block:: python

   >>> # Generate two different symmetric positive-definite matrices
   >>> A = GenerateMatrix(NumPoints=20,DecorrelationScale=1e-1)
   >>> B = GenerateMatrix(NumPoints=20,DecorrelationScale=2e-2)
   
   >>> # Create an interpolating TraceInv object
   >>> TI = InterpolateTraceOfInverse(A,B,InterpolantPoints=InterpolantPoints)

The parameter ``DecorrelationScale`` of the class ``GenerateMatrix`` in the above specifies the scale of correlation function used to form a positive-definite matrix. We specified two correlation scales to generate different matrices ``A`` and ``B``. The user may use their own matrix data.

Interpolation for an array of inquiries points can be made by:

.. code-block:: python

   >>> # Create an array of inquiry points
   >>> import numpy
   >>> t_array = numpy.logspace(-3,+3,5)
   >>> traces = TI.Interpolate(t_array,InterpolantPoints=InterpolantPoints)

The module ``InterpolateTraceOfInverse`` can employ various interpolation methods listed in the table below. The method of interpolation can be set by ``InterpolationMethod`` argument when calling ``InterpolateTraceOfInverse``. The default method is ``RMBF``.

=======================  =========================================  ============  =============  ============
``InterpolationMethod``  Description                                Matrix size   Matrix type    Results
=======================  =========================================  ============  =============  ============
``'EXT'``                Computes trace directly, no interpolation  Small         dense, sparse  exact
``'EIG'``                Uses Eigenvalues of matrix                 Small         dense, sparse  exact
``'MBF'``                Monomial Basis Functions                   Small, large  dense, sparse  interpolated
``'RMBF'``               Root monomial basis functions              small, large  dense, sparse  interpolated
``'RBF'``                Radial basis functions                     small, large  dense, sparse  interpolated
``'RPF'``                Rational polynomial functions              small, large  dense, sparse  interpolated
=======================  =========================================  ============  =============  ============

The ``InterpolateTraceOfInverse`` module internally defines an object of ``ComputeTraceOfInverse`` to evaluate the trace of inverse at the given interpolant points ``InterpolantPoints``. You can pass the options for this internal ``ComputeTraceOfInverse`` object by ``ComputeOptions`` argument when initializing  ``InterpolateTraceOfInverse``, such as in the example below.

.. code-block:: python
    
   >>> # Specify options of the internal ComputeTraceOfInverse object in a dictionary
   >>> ComputeOptions = \
   ... {
   ...     'ComputeMethod': 'hutchinson',
   ...     'NumIterations': 20
   ... }
   
   >>> # Pass options by ComputeOptions argument
   >>> TI = InterpolateTraceOfInverse(A,
   ...             InterpolantPoints=InterpolantPoints,
   ...             InterpolatingMethod='RMBF',
   ...             ComputeOptions=ComputeOptions)

.. _ref_Examples:

********
Examples
********

Three examples are provided in |examplesdir|_, which aim to reproduce the figures presented in [Ameli-2020]_. Namely, in that reference,

Before running examples:
   To run the examples, you may not need to install the ``TraceInv`` package. Rather, download the source code and install requirements:

   ::
    
       # Download
       git clone https://github.com/ameli/TraceInv.git

       # Install prerequisite packages
       cd TraceInv
       python -m pip install --upgrade -r requirements.txt
    
   Then, run either of the examples as described below.


=========
Example 1
=========

Run the script |example1|_ by

::

    python examples/Plot_TraceInv_FullRank.py

The script generates the figure below (see Figure 2 of [Ameli-2020]_).

.. image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/Example1.svg
   :align: center

=========
Example 2
=========

Run the script |example2|_ by

::

    python examples/Plot_TraceInv_IllConditioned.py

The script generates the figure below (see also  Figure 3 of [Ameli-2020]_).

.. image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/Example2.svg
   :align: center

=========
Example 3
=========

Run the script |example3|_ by

::

    python examples/Plot_GeneralizedCrossValidation.py

The script generates the figure below and prints the processing times of the computations. See more details in Figure 3 and results of Table 2 of [Ameli-2020]_.

.. image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/GeneralizedCrossValidation.svg
   :width: 550
   :align: center

********
Citation
********

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

****************
Acknowledgements
****************

* National Science Foundation #1520825
* American Heart Association #18EIA33900046

.. |examplesdir| replace:: ``/examples`` 
.. _examplesdir: https://github.com/ameli/TraceInv/blob/master/examples
.. |example1| replace:: ``/examples/Plot_TraceInv_FullRank.py``
.. _example1: https://github.com/ameli/TraceInv/blob/master/examples/Plot_TraceInv_FullRank.py
.. |example2| replace:: ``/examples/Plot_TraceInv_IllConditioned.py``
.. _example2: https://github.com/ameli/TraceInv/blob/master/examples/Plot_TraceInv_IllConditioned.py
.. |example3| replace:: ``/examples/Plot_GeneralizedCorssValidation.py``
.. _example3: https://github.com/ameli/TraceInv/blob/master/examples/Plot_GeneralizedCrossValidation.py

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
