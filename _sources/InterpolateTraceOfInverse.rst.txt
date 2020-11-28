.. _InterpolateTraceOfInverse:

************************************************************************
Interpolate Trace of Inverse (:mod:`TraceInv.InterpolateTraceOfInverse`)
************************************************************************

=====
Usage
=====

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

==========
Parameters
==========

The :mod:`TraceInv.InterpolateTraceOfInverse` module accepts the following attributes as input argument.

.. attribute:: InterpolationMethod
   :type: string
   :value: 'RMBF'

   Specifies the method of interpolation. The methods are one of ``'EXT'``, ``'EIG'``, ``'MBF'``, ``'RMBF'``, ``'RBF'``, and ``'RPF'`` (see :ref:`Mathematical Details`).

.. attribute:: ComputeOptions
   :type: dict
   :value: {}

   A disctionary to pass the parameters of the internal module :mod:`TraceInv.ComputeTraceOfInverse`. 
   This module internally computes the trace of inverse at interpolant points. 

.. attribute:: Verbose
   :type: bool
   :value: False

.. attribute:: NonZeroRatio
   :type: float
   :value: 0.9

   This parameter is only applied to ``'EIG'`` method.

   In the case of large sparse input matrices, it is not possible to find all the eigenvalues.
   However, often most of the eigenvalues of the matrices in many a applications are near zero. This 
   parameter is a number in the range :math:`[0,1]` and sets a fraction of the eigenvalues to 
   be assumed nonzero. 

.. attribute:: Tolerance
   :type: float
   :value: 1e-3

   This parameter is only applied to ``'EIG'`` method.
   Sets the tolerance of finding the eigenvalues in case the input matrices are sparse.

.. attribute:: FunctionType
   :type: int
   :value: 1

   This parameter is only applied to ``'RBF'`` method. This parameter can be either ``1``, ``2``, or 
   ``3`` and speficies three function types interpolation.

.. attribute:: BasisFunctionsType
   :type: string
   :value: 'Orthogonal2'

   This parameter is only applied to ``'RMBF'`` method. This parameter can be either of
   ``'NonOrthogonal'``, ``'Orthogonal'``, or ``'Orthogonal2'``, which specifies the type 
   of basis functions used in the interpolation.

====================
Mathematical Details
====================

========
Examples
========

-----------------------------------
Comparison of Interpolation Methods
-----------------------------------

The following code compares the methods ``'EXT'``, ``'EIG'``, ``'MBF'``, ``'RMBF'``, ``'RBF'``, and ``'RPF'``.

.. code-block:: python

   >>> # Import packages
   >>> from TraceInv import GenerateMatrix
   >>> from TraceInv import InterpolateTraceOfInverse

   >>> # Generate two symmetric and positive-definite matrices
   >>> A = GenerateMatrix(NumPoints=20,UseSparse=False)
   >>> B = GenerateMatrix(NumPoints=20,UseSparse=False,DecorrelationScale=0.05)

   # Specify interpolation points and the inquiry point
   >>> InterpolantPoints = [1e-4,1e-3,1e-2,1e-1,1,1e+1]
   >>> InquiryPoint = 0.4

   >>> # Specify options to pass to the internal TraceInv.ComputeTraceOfInverse module
   >>> ComputeOptions={'ComputeMethod':'cholesky','UseInverseMatrix':True}

   >>> # Compute exact trace without interpolation
   >>> TI0 = InterpolateTraceOfInverse(A,B=B,InterpolationMethod='EXT',ComputeOptions=ComputeOptions)
   >>> T0 = TI0.Interpolate(InquiryPoint)

   >>> # Eigenvalues Method
   >>> TI1 = InterpolateTraceOfInverse(A,B=B,InterpolationMethod='EIG',ComputeOptions=ComputeOptions)
   >>> T1 = TI1.Interpolate(InquiryPoint)

   >>> # Monomial Basis Functions
   >>> TI2 = InterpolateTraceOfInverse(A,B=B,InterpolationMethod='MBF',ComputeOptions=ComputeOptions)
   >>> T2 = TI2.Interpolate(InquiryPoint)

   >>> # Root Monomial Basis Functions, basis type: Orthogonal2
   >>> TI3= InterpolateTraceOfInverse(A,B=B,InterpolantPoints=InterpolantPoints,InterpolationMethod='RMBF',BasisFunctionsType='Orthogonal2',ComputeOptions=ComputeOptions)
   >>> T3 = TI3.Interpolate(InquiryPoint)

   >>> # Radial Basis Functions, FunctionType 1
   >>> TI4 = InterpolateTraceOfInverse(A,B=B,InterpolantPoints=InterpolantPoints,InterpolationMethod='RBF',FunctionType=1,ComputeOptions=ComputeOptions)
   >>> T4 = TI4.Interpolate(InquiryPoint)

   >>> # Rational Polynomial with four interpolating points
   >>> InterpolantPoints = [1e-2,1e-1,1,1e+1]
   >>> TI5 = InterpolateTraceOfInverse(A,B=B,InterpolantPoints=InterpolantPoints,InterpolationMethod='RPF',ComputeOptions=ComputeOptions)
   >>> T5 = TI5.Interpolate(InquiryPoint)

The results are given in the table below.

========  ====================================  ========  =====
Variable  Method                                Result    Error
========  ====================================  ========  =====
``T0``    Exact method (no interpolation)       590.4149  0.00%
``T1``    Eigenvalues method                    590.4954  0.01%
``T2``    Monomial Basis Functions method       590.2722  0.02%
``T3``    Root Monomial Basis Functions method  590.4579  0.01%
``T4``    Radial Basis Functions method         590.1533  0.04%
``T5``    Rational Polynomial Functions method  590.4157  0.00%
========  ====================================  ========  =====

---------------------------------------
Plot Exact Versus Interpolated Function
---------------------------------------

.. code-block:: python

   >>> # Import packages
   >>> import numpy
   >>> import matplotlib.pyplot as plt
   >>> from TraceInv import GenerateMatrix
   >>> from TraceInv import InterpolateTraceOfInverse

   >>> # Generate a symmetric and positive-definite matrices
   >>> A = GenerateMatrix(NumPoints=20,UseSparse=False)

   >>> # Specify interpolation points and the inquiry point
   >>> InterpolantPoints = [1e-4,1e-3,1e-2,1e-1,1,1e+1]
   >>> t = numpy.logspace(-4,1,100)

   >>> # Create interpolation object with exact method, then compute exact values of trace
   >>> TI = InterpolateTraceOfInverse(A,InterpolationMethod='EXT')
   >>> trace_exact = TI.Interpolate(t)

   >>> # Create interpolation object with RMBF method, then interpolate array t
   >>> TI = InterpolateTraceOfInverse(A,InterpolationMethod='RMBF')
   >>> trace_interpolated = TI.Interpolate(t)

   >>> # Plot results
   >>> plt.plot(t,trace_exact,color='black',label='Exact')
   >>> plt.plot(t,trace_interpolated,color='blue',label='Interpolated')
   >>> plt.xlabel('t')
   >>> plt.ylabel('Trace')
   >>> plt.legend()
   >>> plt.show()


==========
References
==========

===
API
===

--------------
Main Interface
--------------

.. automodapi:: TraceInv.InterpolateTraceOfInverse
   :no-inheritance-diagram:

-------
Modules
-------

.. automodapi:: TraceInv.InterpolateTraceOfInverse.ExactMethod
.. automodapi:: TraceInv.InterpolateTraceOfInverse.EigenvaluesMethod
.. automodapi:: TraceInv.InterpolateTraceOfInverse.MonomialBasisFunctionsMethod
.. automodapi:: TraceInv.InterpolateTraceOfInverse.RootMonomialBasisFunctionsMethod
.. automodapi:: TraceInv.InterpolateTraceOfInverse.RadialBasisFunctionsMethod
.. automodapi:: TraceInv.InterpolateTraceOfInverse.RationalPolynomialFunctionsMethod
