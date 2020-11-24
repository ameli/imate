***********
Quick Start
***********

The package TraceInv provides three sub-packages:

=========================================  =====================================================================
Sub-Package                                Description
=========================================  =====================================================================
:mod:`TraceInv.GenerateMatrix`             Generates symmetric and positive-definite matrices for test purposes.
:mod:`TraceInv.ComputeTraceOfInverse`      Computes trace of inverse for a fixed matrix.
:mod:`TraceInv.InterpolateTraceOfInverse`  Interpolates trace of inverse for a linear matrix function.
=========================================  =====================================================================

The next two sections presents minimalistic examples respectively for:

1. :ref:`Fixed matrix <Fixed-Matrix>` using :mod:`ComputeTraceOfInverse <TraceInv.ComputeTraceOfInverse>` module.
2. :ref:`One-parameter affine matrix function <Affine-Matrix>` using :mod:`InterpolateTraceOfInverse <TraceInv.InterpolateTraceOfInverse>` module.

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

In the above, the class :class:`GenerateMatrix <TraceInv.GenerateMatrix>` produces a sample matrix for test purposes. Specifically, this class produces a correlation matrix from the mutual Euclidean distance of a set of points. By default, the set of points are generated on an equally spaced rectangular grid in the unit square. The produced matrix is symmetric and positive-definite, hence, invertible. The sample matrix ``A`` in the above code is of the size :math:`20^2 \times 20^2` corresponding to a rectangular grid of :math:`20 \times 20` points.

The :mod:`ComputeTraceOfInverse <TraceInv.ComputeTraceOfInverse>` class in the above code employs the Cholesky method by default to compute the trace of inverse. However, the user may choose other methods given in the table below.

===================  ====================================  ==============  =============  =============
``ComputeMethod``    Description                           Matrix size     Matrix type    Results       
===================  ====================================  ==============  =============  =============
``'cholesky'``       Cholesky decomposition                small           dense, sparse  exact          
``'hutchinson'``     Hutchinson's randomized method        small or large  dense, sparse  approximation
``'SLQ'``            Stochastic Lanczos Quadrature method  small or large  dense, sparse  approximation
===================  ====================================  ==============  =============  =============  

The desired method of computation can be passed through the ``ComputeMethod`` argument when calling :mod:`ComputeTraceOfInverse <TraceInv.ComputeTraceOfInverse>`. For instance, in the following example, we apply the *Hutchinson's randomized estimator* method:

.. code-block:: python

   >>> # Using hutchinson method with 20 Monte-Carlo iterations
   >>> trace = ComputeTraceOfInverse(A,ComputeMethod='hutchinson',NumIterations=20)

Each of the methods in the above accept some options. For instance, the Hutchinson's method accepts ``NumIterations`` argument, which sets the number of Monte-Carlo trials. To see the detailed list of all arguments for each method, see the `API <https://ameli.github.io/TraceInv/_modules/modules.html>`__ of the package.

.. _Affine-Matrix:

=================================================
2. Usage for One-Parameter Affine Matrix Function
=================================================

The module :mod:`InterpolateTraceOfInverse <TraceInv.InterpolateTraceOfInverse>` interpolates the trace of the inverse of :math:`\mathbf{A} + t \mathbf{B}`, as shown by the example below.

.. code-block:: python
   :emphasize-lines: 11,15
    
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

In the above code, we only provided the matrix ``A`` to the module :mod:`InterpolateTraceOfInverse <TraceInv.InterpolateTraceOfInverse>`, which then it assumes ``B`` is identity matrix by default. To compute the trace of the inverse of :math:`\mathbf{A} + t \mathbf{B}` where :math:`\mathbf{B}` is not identity matrix, pass both ``A`` and ``B`` to :mod:`InterpolateTraceOfInverse <TraceInv.InterpolateTraceOfInverse>` as follows.

.. code-block:: python

   >>> # Generate two different symmetric positive-definite matrices
   >>> A = GenerateMatrix(NumPoints=20,DecorrelationScale=1e-1)
   >>> B = GenerateMatrix(NumPoints=20,DecorrelationScale=2e-2)
   
   >>> # Create an interpolating TraceInv object
   >>> TI = InterpolateTraceOfInverse(A,B,InterpolantPoints=InterpolantPoints)

The parameter ``DecorrelationScale`` of the class :mod:`GenerateMatrix <TraceInv.GenerateMatrix>` in the above specifies the scale of correlation function used to form a positive-definite matrix. We specified two correlation scales to generate different matrices ``A`` and ``B``. The user may use their own matrix data.

Interpolation for an array of inquiries points can be made by:

.. code-block:: python

   >>> # Create an array of inquiry points
   >>> import numpy
   >>> t_array = numpy.logspace(-3,+3,5)
   >>> traces = TI.Interpolate(t_array,InterpolantPoints=InterpolantPoints)

The module :mod:`InterpolateTraceOfInverse <TraceInv.InterpolateTraceOfInverse>` can employ various interpolation methods listed in the table below. The method of interpolation can be set by ``InterpolationMethod`` argument when calling :mod:`InterpolateTraceOfInverse <TraceInv.InterpolateTraceOfInverse>`. The default method is ``RMBF``.

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

The :mod:`InterpolateTraceOfInverse <TraceInv.InterpolateTraceOfInverse>` module internally defines an object of :class:`ComputeTraceOfInverse <TraceInv.ComputeTraceOfInverse>` (see :ref:`Fixed Matrix <Fixed-Matrix>`) to evaluate the trace of inverse at the given interpolant points ``InterpolantPoints``. You can pass the options for this internal :class:`ComputeTraceOfInverse <TraceInv.ComputeTraceOfInverse>` object by ``ComputeOptions`` argument when initializing  :mod:`InterpolateTraceOfInverse <TraceInv.InterpolateTraceOfInverse>`, such as in the example below.

.. code-block:: python
   :emphasize-lines: 12
    
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
