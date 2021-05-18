.. _InterpolateTraceOfInverse_UserGuide:

************************************************************************
Interpolate Trace of Inverse (:mod:`imate.InterpolateTraceOfInverse`)
************************************************************************

The sub-package :mod:`imate.InterpolateTraceOfInverse` interpolates the trace of inverse of an affine matrix function :math:`\mathbf{A} + t \mathbf{B}`, that is the following function is interpolated for the parameter :math:`t`:

.. math::

    t \mapsto \mathrm{trace} \left( (\mathbf{A} + t \mathbf{B})^{-1} \right).

=====
Usage
=====

In the code below, we create an object of the class :mod:`InterpolateTraceOfInverse <imate.InterpolateTraceOfInverse>`, and then interpolate at an inquiry point :math:`t`.

.. code-block:: python
   :emphasize-lines: 11,15
    
   >>> from imate import GenerateMatrix
   >>> from imate import InterpolateTraceOfInverse
   
   >>> # Generate a symmetric positive-definite matrix of the shape (20**2,20**2)
   >>> A = GenerateMatrix(NumPoints=20)
   
   >>> # Define some interpolant points
   >>> InterpolantPoints = [1e-2,1e-1,1,1e+1]
   
   >>> # Create an interpolating imate object
   >>> TI = InterpolateTraceOfInverse(A,InterpolantPoints=InterpolantPoints)
   
   >>> # Interpolate A+tI at some inquiry point t
   >>> t = 4e-1
   >>> trace = TI.Interpolate(t)

In the above code, we only provided the matrix ``A`` to the module :mod:`InterpolateTraceOfInverse <imate.InterpolateTraceOfInverse>`, which then it assumes ``B`` is identity matrix by default. To compute the trace of the inverse of :math:`\mathbf{A} + t \mathbf{B}` where :math:`\mathbf{B}` is not identity matrix, pass both ``A`` and ``B`` to :mod:`InterpolateTraceOfInverse <imate.InterpolateTraceOfInverse>` as follows.

.. code-block:: python

   >>> # Generate two different symmetric positive-definite matrices
   >>> A = GenerateMatrix(NumPoints=20,DecorrelationScale=1e-1)
   >>> B = GenerateMatrix(NumPoints=20,DecorrelationScale=2e-2)
   
   >>> # Create an interpolating imate object
   >>> TI = InterpolateTraceOfInverse(A,B,InterpolantPoints=InterpolantPoints)

The parameter ``DecorrelationScale`` of the class :mod:`GenerateMatrix <imate.GenerateMatrix>` in the above specifies the scale of correlation function used to form a positive-definite matrix. We specified two correlation scales to generate different matrices ``A`` and ``B``. The user may use their own matrix data.

Interpolation for an array of inquiries points can be made by:

.. code-block:: python

   >>> # Create an array of inquiry points
   >>> import numpy
   >>> t_array = numpy.logspace(-3,+3,5)
   >>> traces = TI.Interpolate(t_array,InterpolantPoints=InterpolantPoints)

The module :mod:`InterpolateTraceOfInverse <imate.InterpolateTraceOfInverse>` can employ various interpolation methods listed in the table below. The method of interpolation can be set by ``InterpolationMethod`` argument when calling :mod:`InterpolateTraceOfInverse <imate.InterpolateTraceOfInverse>`. The default method is ``RMBF``.

===========================  ===========================================  ============  =============  ============
:attr:`InterpolationMethod`  Description                                  Matrix size   Matrix type    Results
===========================  ===========================================  ============  =============  ============
``'EXT'``                    :ref:`Exact Method` (no interpolation)       Small         dense, sparse  exact
``'EIG'``                    :ref:`Eigenvalue Method`                     Small         dense, sparse  exact
``'MBF'``                    :ref:`Monomial Basis Functions Method`       Small, large  dense, sparse  interpolated
``'RMBF'``                   :ref:`Root Monomial Basis Functions Method`  small, large  dense, sparse  interpolated
``'RBF'``                    :ref:`Radial Basis Functions Method`         small, large  dense, sparse  interpolated
``'RPF'``                    :ref:`Rational Polynomial Functions Method`  small, large  dense, sparse  interpolated
===========================  ===========================================  ============  =============  ============

The :mod:`InterpolateTraceOfInverse <imate.InterpolateTraceOfInverse>` module internally defines an object of :class:`ComputeTraceOfInverse <imate.ComputeTraceOfInverse>` (see :ref:`Compute Trace of Inverse user guide <ComputeTraceOfInverse_UserGuide>`) to evaluate the trace of inverse at the given interpolant points ``InterpolantPoints``. You can pass the options for this internal :class:`ComputeTraceOfInverse <imate.ComputeTraceOfInverse>` object by ``ComputeOptions`` argument when initializing  :mod:`InterpolateTraceOfInverse <imate.InterpolateTraceOfInverse>`, such as in the example below.

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

------------------------------------------------------------------------------------------
Parameters for :mod:`InterpolateTraceOfInverse <imate.InterpolateTraceOfInverse>` Class
------------------------------------------------------------------------------------------

The :mod:`imate.InterpolateTraceOfInverse` class accepts the following attributes as input argument.

.. attribute:: A
   :type: numpy.ndarray, or scipy.sparse.csc_matrix

   An invertible sparse or dense matrix.

.. attribute:: B
   :type: numpy.ndarray, or scipy.sparse.csc_matrix
   :value: numpy.eye, or scipy.sparse.eye

   An invertible sparse or dense matrix. If not provided, it is assumed that ``B`` is an identity matrix of the same shape and type as ``A``.

.. attribute:: InterpolantPoints
   :type: list(float)
   :value: None

   List of interpolant points. The trace of inverse at the interpolant points are computed via the exact method using 
   :mod:`imate.ComputeTraceOfInverse` module. For each :attr:`InterpolationMethod`, the interpolant points should be as follows:

   ===========================  ========================================
   :attr:`InterpolationMethod`  :attr:`InterpolantPoints`
   ===========================  ========================================
   ``EXT``                      ``None`` (not required)
   ``EIG``                      ``None`` (not required)
   ``MBF``                      A list of only *one* point
   ``RMBF``                     A list of *arbitrary number* of points
   ``RBF``                      A list of *arbitrary number* of points
   ``RPF``                      A list of either *two* or *four* points
   ===========================  ========================================

.. attribute:: InterpolationMethod
   :type: string
   :value: 'RMBF'

   Specifies the method of interpolation. The methods are one of ``'EXT'``, ``'EIG'``, ``'MBF'``, ``'RMBF'``, ``'RBF'``, and ``'RPF'`` (see :ref:`Mathematical Details`).

.. attribute:: ComputeOptions
   :type: dict
   :value: {}

   Recall that the trace of inverse at the interpolant points are computed via the exact method using an internal
   :mod:`imate.ComputeTraceOfInverse` object within this class. The parameters of this internal object can be passed
   by the :attr:`ComputeOptions` disctionary. For instance, to 

.. attribute:: Verbose
   :type: bool
   :value: False

   If ``True``, prints some information during the process.

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

------------------------------------------------------------------------------------------------------------------------
Parameters for :func:`Interpolate() <imate.InterpolateTraceOfInverse.InterpolateTraceOfInverse.Interpolate>` Function
------------------------------------------------------------------------------------------------------------------------

The :func:`imate.InterpolateTraceOfInverse.Interpolate() <imate.InterpolateTraceOfInverse.InterpolateTraceOfInverse.Interpolate>` function accepts the following attributes as input argument.

.. attribute:: t
   :type: scalar, or numpy.array
   
   An inquiry point (or an array of inquiry points) to interpolate.

.. attribute:: CompareWithExact
   :type: bool
   :value: False
   
   If ``True``, it computes the trace with exact solution, then compares it with the interpolated 
   solution. When this option is enabled, the function :func:`Interpolate() <imate.InterpolateTraceOfInverse.Interpolate>` 
   returns a tuple with the following three quantities:
   
       1. interpolated trace (single number or array)
       2. exact solution (sngle number or array)
       3. Relative error of interpolated trace compared with the exact solution

   .. warning::
   
       When the option :attr:`CompareWithExact` is enabled, depending on the matrix size and the length of the input array :attr:`t`, 
       the processing time will be significantly longer. This is becase the solution will be computed for all inquiry points 
       using the exact method (besides the interpolation). Use this option only for test purposes (benchmarking) on small matrices.

.. attribute:: Plot
   :type: bool
   :value: False
        
   If ``True``, it plots the interpolated trace versus the inquiry points.
   
   * If the option :attr:`CompareWithExact` is also set to ``True``, the plotted diagram contains both interpolated 
     and exact solutions, together with the *relative error* of interpolated solution with respect to the exact solution.
   * If no graphical backend exists (such as running the code on a remote server or manually disabling the X11 backend), 
     the plot will not be shown, rather, it will be saved as an ``svg`` file in the current directory. 
   * If the executable ``latex`` can be found on the path, the plot is rendered using :math:`\rm\LaTeX`, which then, 
     it takes a bit longer to produce the plot. 
   * If :math:`\rm\LaTeX` is not installed, it uses any available San-Serif font to render the plot.

   .. note::

       To manually disable interactive plot display, and save the plot as ``SVG`` instead, add the following in the
       very begining of your code before importing ``imate``:

       .. code-block:: python
         
           >>> import os
           >>> os.environ['IMATE_NO_DISPLAY'] = 'True'

====================
Mathematical Details
====================

In the methods that follows (except the exact and the eigenvalues method), instead of interpolating the function 
:math:`t \mapsto \mathrm{trace}\left( (\mathbf{A} + t \mathbf{B})^{-1} \right)`, the function :math:`t \mapsto \tau(t)` 
is interpolated, where :math:`\tau(t)` is defined by

.. math::

    \tau(t) = \frac{\mathrm{trace}\left( (\mathbf{A} + t \mathbf{B})^{-1} \right)}{\mathrm{trace}\left(\mathbf{B}^{-1} \right)}.

Also, we define 

.. math::
    
    \tau_0 = \tau(0) = \frac{\mathrm{trace} \left( \mathbf{A}^{-1} \right)}{\mathrm{trace} \left( \mathbf{B}^{-1} \right)}.

------------
Exact Method
------------

The exact method (by setting :attr:`InterpolationMethod` to ``'EXT'``) do not perform any interpolation on the inquiry 
point. Rather, it computes the trace of inverse directly via the :mod:`imate.ComputeTraceOfInverse` module. This method 
is primarily used for comparing the result of other methods with a benchmark solution. For details of computation, 
see :ref:`Compute Trace of Inverse <ComputeTraceOfInverse_UserGuide>` user guide.

.. warning::

    Since this method does not perform interpolation, the process time is proportional to the number of the inquity points :math:`t`.

-----------------
Eigenvalue Method
-----------------

This method is employed by setting :attr:`InterpolationMethod` to ``'EIG'``. The trace of inverse is computed via

.. math::

    \mathrm{trace}\left( (\mathbf{A} + t \mathbf{B})^{-1} \right) = \sum_{i = 1}^n \frac{1}{\lambda_i + t \mu_i}

where :math:`\lambda_i` is the eigenvalue of :math:`\mathbf{A}` and :math:`\mu_i` is the eigenvalue of :math:`\mathbf{B}`. 
This class does not accept interpolant points as the result is not interpolated.

For dense matrices, the results of this method is identical to the exact (``'EXT'``) method as no approximation has been 
made. For sparse matrices, however, it is not possible to compute all spectrum of a large matrix, and the small 
eigenvalues are assumed to be zero, Hence, the results are very close, yet not the same, as the exact method.

.. note::

    This is the fastest and most accurate interpolation method, but can only be applied on small matrices.

.. warning::

    This method is not suitable for large matrices, and practically, may stall and never return output. 

-------------------------------
Monomial Basis Functions Method
-------------------------------

This method is invoked by setting :attr:`InterpolationMethod` to ``'MBF'``. The trace of inverse is computed via
Computes the trace of inverse of an invertible matrix :math:`\mathbf{A} + t \mathbf{B}` using

.. math::

    \frac{1}{(\tau(t))^{p+1}} \approx \frac{1}{(\tau_0)^{p+1}} + \sum_{i=1}^{p+1} w_i t^i,

where :math:`w_{p+1} = 1`. To find the weight coefficient :math:`w_1`, the trace is computed at the given interpolant point :math:`t_1` 
(see :attr:`InterpolantPoints` argument).

When :math:`p = 1`, meaning that there is only one interpolant point :math:`t_1` with the function value :math:`\tau_1 = \tau(t_1)`, the 
weight coefficient :math:`w_1` can be solved easily. In this case, the interpolation function becomes

.. math::


    \frac{1}{(\tau(t))^2} \approx  \frac{1}{\tau_0^2} + t^2 + \left( \frac{1}{\tau_1^2} - \frac{1}{\tau_0^2} - t_1^2 \right) \frac{t}{t_1}.
    
.. note::

    This class accepts only *one* interpolant point (:math:`p = 1`). That is, the parameter :attr:`InterpolantPoints` should be only 
    one number or a list of the length one.

------------------------------------
Root Monomial Basis Functions Method
------------------------------------

This method is invoked by setting :attr:`InterpolationMethod` to ``'RMBF'``. In this method, the function :math:`(\tau(t))^{-1}` 
is approximated via

.. math::

    \frac{1}{\tau(t)} \approx \frac{1}{\tau_0} + \sum_{i = 0}^p w_i \phi_i(t),

where  :math:`\phi_i` are some known basis functions, and :math:`w_i` are the coefficients of the linear basis functions.
The first coefficient is set to :math:`w_{0} = 1` and the rest of the weights 
are to be found form the known function values :math:`\tau_i = \tau(t_i)` at some given interpolant points :math:`t_i`.
    
**Basis Functions:**

Two types of basis functions can be set by the argument :attr:`BasisFunctionsType`.

1. When :attr:`BasisFunctionsType` is set to ``NonOrthogonal``, the basis functions are the root of the monomial functions defined by

   .. math::
   
       \phi_i(t) = t^{\frac{1}{i+1}}, \qquad i = 0,\dots,p.
   
2. When :attr:`BasisFunctionsType` is set to ``'Orthogonal'`` or ``'Orthogonal2'``, the orthogonal form of the
   above basis functions are used. Orthogonal basis functions are formed by the above non-orthogonal functions
   as
   
   .. math::
   
       \phi_i^{\perp}(t) = \alpha_i \sum_{j=1}^i a_{ij} \phi_j(t)
   
   The coefficients :math:`\alpha_i` and :math:`a_{ij}` can be obtained by the python package 
   `Orthogoanl Functions <https://ameli.github.io/Orthogonal-Functions>`_. These coefficients are
   hard-coded in this function up to :math:`i = 9`. Thus, in this module, up to nine interpolant points
   are supported. 
   
   .. warning::
   
       The non-orthogonal basis functions can lead to ill-conditioned system of equations for finding the weight
       coefficients :math:`w_i`. When the number of interpolating points is large (such as :math:`p > 6`), 
       it is recommended to use the orthogonalized set of basis functions.
   
   .. note::
   
       The recommended basis function type is ``'Orthogonal2'``.

-----------------------------
Radial Basis Functions Method
-----------------------------

This method is invoked by setting :attr:`InterpolationMethod` to ``'RBF'``. In this method, the function :math:`\tau(t)` 
is approximated by radial basis functions. Define

.. math::

    x(t) = \log t

Depending whether :attr:`FunctionType` is set to ``1``, ``2``, or ``3``, one of the following functions is defined:

.. math::
    :nowrap:

    \begin{eqnarray}
    y_1(t) &= \frac{1}{\tau(t)} - \frac{1}{\tau_0} - t, \\
    y_2(t) &= \frac{\frac{1}{\tau(t)}}{\frac{1}{\tau_0} + t} - 1, \\
    y_3(t) &= 1 - \tau(t) \left( \frac{1}{\tau_0} + t \right).
    \end{eqnarray}

* The set of data :math:`(x,y_1(x))` are interpolated using *cubic splines*.
* The set of data :math:`(x,y_2(x))` and :math:`(x,y_3(x))` are interpolated using *Gaussian radial basis functions*.

------------------------------------
Rational Polynomial Functions Method
------------------------------------

This method is invoked by setting :attr:`InterpolationMethod` to ``'RPF'``. 
In this method, the function :math:`\tau(t)` is approximated by

.. math::

    \tau(t) \approx \frac{t^p + a_{p-1} t^{p-1} + \cdots + a_1 t + a_0}{t^{p+1} + b_p t^p + \cdots + b_1 t + b_0}

where :math:`a_0 = b_0 \tau_0`. The rest of coefficients are found by solving a linear system using the 
function value at the interpolant points :math:`\tau_i = \tau(t_i)`.

.. note::

    The number of interpolant points :math:`p` in this method can only be either :math:`p = 2` or :math:`p = 4`.

========
Examples
========

-----------------------------------
Comparison of Interpolation Methods
-----------------------------------

The following code compares the methods ``'EXT'``, ``'EIG'``, ``'MBF'``, ``'RMBF'``, ``'RBF'``, and ``'RPF'``.

.. code-block:: python

   >>> # Import packages
   >>> from imate import GenerateMatrix
   >>> from imate import InterpolateTraceOfInverse

   >>> # Generate two symmetric and positive-definite matrices
   >>> A = GenerateMatrix(NumPoints=20,UseSparse=False)
   >>> B = GenerateMatrix(NumPoints=20,UseSparse=False,DecorrelationScale=0.05)

   # Specify interpolation points and the inquiry point
   >>> InterpolantPoints = [1e-4,1e-3,1e-2,1e-1,1,1e+1]
   >>> InquiryPoint = 0.4

   >>> # Specify options to pass to the internal imate.ComputeTraceOfInverse module
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

The results are given in the table below. The last column of the table shows the relative error with respect to the exact method (first row).

========  =========================================== ========  =====
Variable  Method                                      Result    Error
========  =========================================== ========  =====
``T0``    :ref:`Exact Method` (no interpolation)      590.4149  0.00%
``T1``    :ref:`Eigenvalue Method`                    590.4954  0.01%
``T2``    :ref:`Monomial Basis Functions Method`      590.2722  0.02%
``T3``    :ref:`Root Monomial Basis Functions Method` 590.4579  0.01%
``T4``    :ref:`Radial Basis Functions Method`        590.1533  0.04%
``T5``    :ref:`Rational Polynomial Functions Method` 590.4157  0.00%
========  =========================================== ========  =====

---------------------------------------
Plot Exact Versus Interpolated Function
---------------------------------------

In this example, we use the default method (``RMBF``), but interpolate for an array of inquiry points ``t``. Also, we compare the results with the exact method (``EXT``) and plot the errors.

.. code-block:: python

   >>> # Generate symmetric and positive-definite matrices
   >>> from imate import GenerateMatrix
   >>> A = GenerateMatrix(NumPoints=20,DecorrelationScale=0.20)
   >>> B = GenerateMatrix(NumPoints=20,DecorrelationScale=0.05)

   >>> # Specify interpolation points, and create interpolating object
   >>> from imate import InterpolateTraceOfInverse
   >>> InterpolantPoints = [1e-3,1e-2,1e-1,1,1e+1,1e+2,1e+3]
   >>> TI = InterpolateTraceOfInverse(A,B,InterpolantPoints)

   >>> # Specify an array of inquiry points, then interpolate at inquiry points
   >>> import numpy
   >>> t = numpy.logspace(-3,3,100)
   >>> trace_interpolated, trace_exact, relative_error = TI.Interpolate(t,CompareWithExact=True,Plot=True)

The above code produces the following plot. The interpolant points are shown by the red dots. The red curve on the left plot is behind the black curve, since the interpolation is very close to the exact values. Note that because of setting :attr:`CompareWithExact` to ``True``, the above code takes more processing time since it also computes the exact values for all elements of the array ``t``. Without enabling this option, the interpolation takes significantly less processing time.

.. image:: images/InterpolationResults.svg
   :align: center

==========
References
==========

.. [Ameli-2020] Ameli, S., and Shadden. S. C. (2020). Interpolating the Trace of the Inverse of Matrix :math:`\mathbf{A} + t \mathbf{B}`. `arXiv:2009.07385 <https://arxiv.org/abs/2009.07385>`__ [math.NA]

===
API
===

--------------
Main Interface
--------------

.. automodapi:: imate.InterpolateTraceOfInverse
   :no-inheritance-diagram:

-------------------
Inheritance Diagram
-------------------

.. inheritance-diagram:: imate.InterpolateTraceOfInverse.ExactMethod imate.InterpolateTraceOfInverse.EigenvaluesMethod imate.InterpolateTraceOfInverse.MonomialBasisFunctionsMethod imate.InterpolateTraceOfInverse.RootMonomialBasisFunctionsMethod imate.InterpolateTraceOfInverse.RadialBasisFunctionsMethod imate.InterpolateTraceOfInverse.RationalPolynomialFunctionsMethod
    :parts: 1

-------
Modules
-------

.. automodapi:: imate.InterpolateTraceOfInverse.ExactMethod
.. automodapi:: imate.InterpolateTraceOfInverse.EigenvaluesMethod
.. automodapi:: imate.InterpolateTraceOfInverse.MonomialBasisFunctionsMethod
.. automodapi:: imate.InterpolateTraceOfInverse.RootMonomialBasisFunctionsMethod
.. automodapi:: imate.InterpolateTraceOfInverse.RadialBasisFunctionsMethod
.. automodapi:: imate.InterpolateTraceOfInverse.RationalPolynomialFunctionsMethod
