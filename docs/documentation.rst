************
Introduction
************

This package computes the trace of inverse of two forms of matrices:

===============
1. Fixed Matrix
===============

For an invertible matrix |image01|, this package computes |image02| by either of these three methods:

#. **Cholesky Decomposition**:  accurate, suitable for small matrices.
#. **Hutchinson's Randomized Estimator**: approximation, suitable for large matrices.
#. **Stochastic Lanczos Quadrature**: approximation, suitable for large matrices.

Both sparse and dense matrices are supported.

=======================================
2. One-Parameter Affine Matrix Function
=======================================

This package can interpolate the one-parameter affine function

|image06|

where |image01| and |image03| are symmetric and positive-definite matrices and :math:`t` is a parameter.

The above function is featured in a wide range of applications in statistics and machine learning. Particular applications are in model selection and optimizing hyperparameters with gradient-based maximum likelihood methods. In such applications, computing the above function is often a computational challenge for large matrices. Often, this function is evaluated for a wide range of the parameter :math:`t` while :math:`\mathbf{A}` and :math:`\mathbf{B}` remain fixed. As such, an interpolation scheme enables fast computation of the function.

Various interpolation methods of the above function are implemented in this package, namely:

#. **Eigenvalues Method**
#. **Monomial Basis Functions**
#. **Root Monomial Basis Functions**
#. **Rational Polynomial Functions**
#. **Radial Basis Functions**

These interpolation methods are described in [Ameli-2020]_. 


.. include:: math_sphinx.rst

*******
Install
*******

=============
Prerequisites
=============

The prerequisite packages are:

* ``numpy`` and ``scipy``.
* ``matplotlib`` and ``seaborn``, but only needed to run the examples provided in ``/examples`` (see also :ref:`Examples`).

By installing ``TraceInv`` (see :ref:`Install ``TraceInv``` below), the prerequisite packages will be installed automatically and no other action is needed.

====================
Install ``TraceInv``
====================

Install ``TraceInv`` by either of the following ways:

- Method 1: install from the package available at `PyPi <https://pypi.org/project/TraceInv>`_:

  ::

    python -m pip install TraceInv


- Method 2: download the source code and install by:

  ::

    git clone https://github.com/ameli/TraceInv.git
    cd TraceInv
    python -m pip install -e .

=========================
Install Optional Packages
=========================

Installing the additional packages ``ray`` and ``scikit-sparse`` can improve the performance of ``TraceInv``, but they are not required. 

---------------
Install ``ray``
---------------

To run the examples provided in ``/examples`` (see also `Examples <Examples>`_), you may install the ``ray`` package to leverage the parallel processing used in generate large sparse matrices. Installing ``ray`` is optional as the examples can still produce results without it. Install ``ray`` package by:

::

    python -m pip install ray

-------------------------
Install ``scikit-sparse``
-------------------------

In ``TraceInv`` package, one of the methods to compute the trace of a matrix is by the *Cholesky decomposition*. If the input matrix is *sparse*, the Cholesky decomposition is computed with either of these packages below, whichever is available:

* ``scikit-sparse``: If this is available, ``TraceInv`` uses this package by *default*. Note that ``scikit-sparse`` is not available by installing ``TraceInv`` only, rather, it should be installed separately (see instructions below).
* ``scipy``: If ``scikit-sparse`` is not available, the ``scipy`` package is used instead. The ``scipy`` package is readily available by installing ``TraceInv``.

To install ``scikit-sparse``, follow the two steps below.

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

*****
Usage
*****

The package ``TraceInv`` provides three sub-packages:

======================================  =====================================================================
Sub-Package                             Description
======================================  =====================================================================
``TraceInv.GenerateMatrix``             Generates symmetric and positive-definite matrices for test purposes.
``TraceInv.ComputeTraceOfInverse``      Computes trace of inverse for a fixed matrix.
``TraceInv.InterpolateTraceOfInverse``  Interpolates trace of inverse for a linear matrix function.
======================================  =====================================================================

The next two sections presents minimalistic examples respectively for:

1. :ref:`Fixed matrix <Fixed-Matrix>` using ``ComputeTraceOfInverse`` module.
2. :ref:`One-parameter affine matrix function <Affine-Matrix>` using ``InterpolateTraceOfInverse`` module.

.. _Fixed-Matrix:

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

In the above, the class ``GenerateMatrix`` produces a sample matrix for test purposes. Specifically, this class produces a correlation matrix from the mutual Euclidean distance of a set of points. By default, the set of points are generated on an equally spaced rectangular grid in the unit square. The produced matrix is symmetric and positive-definite, hence, invertible. The sample matrix ``A`` in the above code is of the size :math:`20^2 \times 20^2` corresponding to a rectangular grid of :math:`20 \times 20` points.

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

The module ``InterpolateTraceOfInverse`` interpolates the trace of the inverse of :math:`\mathbf{A} + t \mathbf{B}`, as shown by the example below.

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

In the above code, we only provided the matrix ``A`` to the module ``InterpolateTraceOfInverse``, which then it assumes ``B`` is identity matrix by default. To compute the trace of the inverse of :math:`\mathbf{A} + t \mathbf{B}` where :math:`\mathbf{B}` is not identity matrix, pass both ``A`` and ``B`` to ``InterpolateTraceOfInverse`` as follows.

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

=======================  =========================================  ============  =============  =============
``InterpolationMethod``  Description                                Matrix size   Matrix type    Results
=======================  =========================================  ============  =============  =============
``'EXT'``                Computes trace directly, no interpolation  Small         dense, sparse  exact
``'EIG'``                Uses Eigenvalues of matrix                 Small         dense, sparse  exact
``'MBF'``                Monomial Basis Functions                   Small, large  dense, sparse  interpolation
``'RMBF'``               Root monomial basis functions              small, large  dense, sparse  interpolation
``'RBF'``                Radial basis functions                     small, large  dense, sparse  interpolation
``'RPF'``                Rational polynomial functions              small, large  dense, sparse  interpolation
=======================  =========================================  ============  =============  =============

The ``InterpolateTraceOfInverse`` module internally defines a ``ComputeTraceOfInverse`` object (see :ref:`Fixed Matrix <Fixed-Matrix>`) to evaluate the trace of inverse at the given interpolant points ``InterpolantPoints``. You can pass the options for this internal ``ComputeTraceOfInverse`` object by ``ComputeOptions`` argument when initializing  ``InterpolateTraceOfInverse``, such as in the example below.

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

.. _Examples:

********
Examples
********

Three examples are provided in |examples|_, which aim to reproduce the figures presented in [Ameli-2020]_. Namely, in that reference,

1. |example1|_ reproduces Figure 2.
2. |example2|_ reproduces Figure 3.
3. |example3|_ reproduces Figure 4 and generates the results of Table 2.

To run the examples only, you may not need to install the ``TraceInv`` package, rather, you can download the source code and run the examples by

::
    
    # Download
    git clone https://github.com/ameli/TraceInv.git

    # Install prerequisite packages
    cd TraceInv
    python -m pip install --upgrade -r requirements.txt

and run either of the examples as described below.


=========
Example 1
=========

The script |example1|_ plots the interpolation of the function

.. math::

    \tau(t) = \mathrm{trace} \left( (\mathbf{A} + t \mathbf{I})^{-1} \right)

using **Root Monomial basis Function** method. Here, :math:`\mathbf{I}` is the identity matrix, :math:`\mathbf{A}` is a dense full-rank correlation matrix of the size :math:`50^2 \times 50^2`, and :math:`t \in [10^{-4},10^3]`.

Run this example by

::

    python examples/Plot_TraceInv_FullRank.py

The script generates the figure below (see Figure 2 of [Ameli-2020]_).

.. image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/Example1.svg
   :align: center

The plot on the left shows the interpolation of :math:`\tau(t)`. Each colored curve is obtained using different number of interpolant points :math:`p`. The plot on the right represents the relative error  of interpolation compared to the accurate computation when no interpolation is applied. Clearly, employing more interpolant points (such as the red curve with :math:`p = 9` interpolant points) yield smaller interpolation error.

=========
Example 2
=========

The script |example2|_ plots the interpolation of the function :math:`\tau(t)` as defined in :ref:`Example 1`, however, here, the matrix :math:`\mathbf{A}` is defined by 

.. math::

    \mathbf{A} = \mathbf{X}^{\intercal} \mathbf{X} + s \mathbf{I}

where :math:`\mathbf{X}` is an ill-conditioned matrix of the size :math:`1000 \times 500`, and the fixed shift parameter :math:`s=10^{-3}` is applied to improve the condition number of the matrix to become invertible.


The interpolation is performed using **Rational Polynomial Function** method for :math:`t \in [-10^{-3},10^{3}]`.


Run this example by

::

    python examples/Plot_TraceInv_IllConditioned.py

The script generates the figure below (see also  Figure 3 of [Ameli-2020]_).

.. image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/Example2.svg
   :align: center

=========
Example 3
=========

The script |example3|_ plots the `Generalized Cross-validation <https://www.jstor.org/stable/1390722?seq=1>`_ function

.. math::

    V(\theta) = \frac{\frac{1}{n} \| \mathbf{I} - \mathbf{X} (\mathbf{X}^{\intercal} \mathbf{X} + n \theta \mathbf{I})^{-1} \mathbf{X}^{\intercal} \boldsymbol{z} \|_2^2}{\left( \frac{1}{n} \mathrm{trace}\left( (\mathbf{I} - \mathbf{X}(\mathbf{X}^{\intercal} \mathbf{X} + n \theta \mathbf{I})^{-1})\mathbf{X}^{\intercal} \right) \right)^2}

where :math:`\mathbf{X}` is the same matrix as :ref:`Example 2` and the term involving the trace of inverse in the denominator is interpolated as presented in :ref:`Example 2`.

Run this example by

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

.. |examples| replace:: ``/examples`` 
.. _examples: https://github.com/ameli/TraceInv/blob/master/examples
.. |example1| replace:: ``/examples/Plot_TraceInv_FullRank.py``
.. _example1: https://github.com/ameli/TraceInv/blob/master/examples/Plot_TraceInv_FullRank.py
.. |example2| replace:: ``/examples/Plot_TraceInv_IllConditioned.py``
.. _example2: https://github.com/ameli/TraceInv/blob/master/examples/Plot_TraceInv_IllConditioned.py
.. |example3| replace:: ``/examples/Plot_GeneralizedCorssValidation.py``
.. _example3: https://github.com/ameli/TraceInv/blob/master/examples/Plot_GeneralizedCrossValidation.py


****************
Acknowledgements
****************

* National Science Foundation #1520825
* American Heart Association #18EIA33900046
