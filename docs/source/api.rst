.. _api:

.. role:: synco
   :class: synco

=============
API Reference
=============

The API reference contains:

* :ref:`Functions <Functions>`: compute log-determinant and trace of functions
  of matrices.
* :ref:`Interpolators <Interpolators>`: interpolate functions of one-parameter
  family of affine matrix functions.
* :ref:`Linear Operators <Linear Operators>`: classes that represent matrices
  and affine matrix functions.
* :ref:`Sample Matrices <Sample Matrices>`: generate matrices for test
  purposes.
* :ref:`Device Inquiry <Device Inquiry>`: inquiry information about CPU and GPU devices.

.. _Functions:

---------
Functions
---------

The functions of this package are:

* :ref:`Log-Determinant <Log-Determinant>`: computes log-determinant of matrix.
* :ref:`Trace of Inverses <Trace of Inverses>`: computes trace of the inverse of
  a matrix or any negative power of the matrix.
* :ref:`Trace <Trace>`: computes the trace of matrix or any positive power of
  the matrix.
* :ref:`Schatten Norm <Schatten Norm>`: computes the Schatten norm of order
  :math:`p`, which includes the above three functions. 

Each of the above functions are implemented using both direct and randomized algorithms, suitable for various matrices sizes.

.. _Log-Determinant:

Log-Determinant
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :caption: logdet
    :recursive:
    :template: autosummary/member.rst

    imate.logdet

This function computes the log-determinant of :math:`\mathbf{A}^p` or the Gramian matrix :math:`(\mathbf{A}^{\intercal} \mathbf{A})^p` where :math:`p` is a real exponent.

The `imate.logdet` function supports the following methods:

.. toctree::

    api/imate.logdet.eigenvalue
    api/imate.logdet.cholesky
    api/imate.logdet.slq

===========================================  =============================  ===========================  ===========================  =============
Method                                       Algorithm                      Matrix size                  Matrix type                  Solution
===========================================  =============================  ===========================  ===========================  =============
:ref:`eigenvalue <imate.logdet.eigenvalue>`  Eigenvalue decomposition       Small :math:`n < 2^{12}`     Any                          Exact
:ref:`cholesky <imate.logdet.cholesky>`      Cholesky decomposition         Moderate :math:`n < 2^{15}`  Symmetric Positive-definite  Exact
:ref:`slq <imate.logdet.slq>`                Stochastic Lanczos Quadrature  Large :math:`n > 2^{12}`     Any                          Approximation
===========================================  =============================  ===========================  ===========================  =============

.. _Trace of Inverses:

Trace of Inverse
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :caption: traceinv
    :recursive:
    :template: autosummary/member.rst

    imate.traceinv

This function computes the trace of :math:`\mathbf{A}^{-p}` or the Gramian matrix :math:`(\mathbf{A}^{\intercal} \mathbf{A})^{-p}` where :math:`p` is a positive real exponent.

The `imate.traceinv` function supports the following methods:

.. toctree::

    api/imate.traceinv.eigenvalue
    api/imate.traceinv.cholesky
    api/imate.traceinv.hutchinson
    api/imate.traceinv.slq

============  =============================  ===========================  ============================================
Method        Algorithm                      Matrix Size                  Notes
============  =============================  ===========================  ============================================
`eigenvalue`  Eigenvalue decomposition       Small :math:`n < 2^{12}`     For testing and benchmarking other methods
`cholesky`    Cholesky decomposition         Moderate :math:`n < 2^{15}`  Exact solution
`hutchinson`  Hutchinson                     Large :math:`2^{12} < n`     Randomized method using Monte-Carlo sampling
`slq`         Stochastic Lanczos Quadrature  Large :math:`2^{12} < n`     Randomized method using Monte-Carlo sampling
============  =============================  ===========================  ============================================

.. _Trace:

Trace
~~~~~

.. autosummary::
    :toctree: generated
    :caption: trace
    :recursive:
    :template: autosummary/member.rst

    imate.trace

This function computes the trace of :math:`\mathbf{A}^p` or the Gramian matrix :math:`(\mathbf{A}^{\intercal} \mathbf{A})^p` where :math:`p` is a positive real exponent.

The `imate.trace` function supports the following methods:

.. toctree::

    api/imate.trace.exact
    api/imate.trace.eigenvalue
    api/imate.trace.slq

============  =============================  ========================  ============================================
Method        Algorithm                      Matrix Size               Notes
============  =============================  ========================  ============================================
`exact`       Direct                         All sizes                 For all purposes
`eigenvalue`  Eigenvalue decomposition       Small :math:`n < 2^{12}`  For testing and benchmarking other methods
`slq`         Stochastic Lanczos Quadrature  Large :math:`2^{12} < n`  Randomized method using Monte-Carlo sampling
============  =============================  ========================  ============================================
 

.. _Schatten Norm:

Schatten Norm
~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :caption: Schatten Norm
    :recursive:
    :template: autosummary/member.rst

    imate.schatten

This function computes the Schatten norm of :math:`\mathbf{A}^p` or the Gramian matrix :math:`(\mathbf{A}^{\intercal} \mathbf{A})^p` where :math:`p` is a real exponent.

.. _Interpolators:

-------------
Interpolators
-------------

Interpolate the various matrix functions of the one-parameter family of affine matrix functions.

.. autosummary::
    :toctree: generated
    :caption: Interpolators
    :recursive:
    :template: autosummary/class.rst

    imate.InterpolateLogdet
    imate.InterpolateTrace
    imate.InterpolateSchatten

.. _Linear Operators:

----------------
Linear Operators
----------------

Create linear operator objects as container for various matrix types with a unified interface, establish a fully automatic dynamic buffer to allocate, deallocate, and transfer data between CPU and multiple GPU devices on demand, as well as perform basic matrix-vector operations with high performance on both CPU or GPU devices. These objects can be passed to :synco:`imate` functions as input matrices.

.. autosummary::
    :toctree: generated
    :caption: Linear Operators
    :recursive:
    :template: autosummary/class.rst

    imate.Matrix
    imate.AffineMatrixFunction

.. _Sample Matrices:

---------------
Sample Matrices
---------------

Generate sample matrices for test purposes, such as correlation matrix and Toeplitz matrix. The matrix functions of Toeplitz matrix (such as its log-determinant, trace of its inverse, etc) are known analytically, making Toeplitz matrix suitable for benchmarking the result of randomized methods with analytical solutions.
   
.. autosummary::
    :toctree: generated
    :caption: Sample Matrices
    :recursive:
    :template: autosummary/member.rst

    imate.correlation_matrix
    imate.toeplitz
    imate.sample_matrices.toeplitz_logdet
    imate.sample_matrices.toeplitz_trace
    imate.sample_matrices.toeplitz_traceinv

.. _Device Inquiry:

--------------
Device Inquiry
--------------

Measure the process time and consumed memory of the Python process during computation with the following classes.

.. autosummary::
    :toctree: generated
    :caption: Device Inquiry
    :recursive:
    :template: autosummary/class.rst

    imate.Timer
    imate.Memory

Inquiry hardware information, including CPU and GPU devices employed during computation and get information about the CUDA Toolkit installation with the following functions.

.. autosummary::
    :toctree: generated
    :recursive:
    :template: autosummary/member.rst

    imate.info
    imate.device.get_processor_name
    imate.device.get_gpu_name
    imate.device.get_num_cpu_threads
    imate.device.get_num_gpu_devices
    imate.device.locate_cuda
    imate.device.restrict_to_single_processor