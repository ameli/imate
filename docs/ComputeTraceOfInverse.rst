.. _ComputeTraceOfInverse_UserGuide:

****************************************************************
Compute Trace of Inverse (:mod:`imate.ComputeTraceOfInverse`)
****************************************************************

The sub-package :mod:`imate.ComputeTraceOfInverse` computes the trace of inverse of an invertible matrix. 

=====
Usage
=====

.. code-block:: python

   >>> from imate import GenerateMatrix
   >>> from imate import ComputeTraceOfInverse
   
   >>> # Generate a symmetric positive-definite matrix of the shape (20**2,20**2)
   >>> A = GenerateMatrix(NumPoints=20)
   
   >>> # Compute trace of inverse
   >>> trace = ComputeTraceOfInverse(A)

In the above, the class :class:`GenerateMatrix <imate.GenerateMatrix>` produces a sample matrix for test purposes (see :ref:`Generate Matrix <GenerateMatrix>` for details).

The :mod:`ComputeTraceOfInverse <imate.ComputeTraceOfInverse>` module in the above code employs the *Cholesky method* by default to compute the trace of inverse. However, other methods can be employed by setting :attr:`ComputeMethod` argument according to the table below.

=====================  ===========================================================================  ============  =============  =============
:attr:`ComputeMethod`  Description                                                                  Matrix size   Structure      Results       
=====================  ===========================================================================  ============  =============  =============
``'cholesky'``         :ref:`Cholesky decomposition <Cholesky Decomposition Method>`                small         dense, sparse  exact          
``'hutchinson'``       :ref:`Hutchinson's randomized method <Hutchinson Randomized Method>`         small, large  dense, sparse  approximation
``'SLQ'``              :ref:`Stochastic Lanczos Quadrature <Stochastic Lanczos Quadrature Method>`  small, large  dense, sparse  approximation
=====================  ===========================================================================  ============  =============  =============

In the following example, we apply the *Hutchinson's randomized estimator* method:

.. code-block:: python

   >>> # Using Hutchinson method with 20 Monte-Carlo iterations
   >>> trace = ComputeTraceOfInverse(A,ComputeMethod='hutchinson',NumIterations=20)

Each of the methods in the above accepts some options. For instance, the Hutchinson's method accepts :attr:`NumIterations` argument, which sets the number of Monte-Carlo trials. To see the detailed list of all arguments for each method, see :ref:`Parameters` and the `API <https://ameli.github.io/imate/_modules/modules.html>`__ of the package.

==========
Parameters
==========

The :mod:`imate.ComputeTraceOfInverse` module accepts the following attributes as input argument.

.. attribute:: A
   :type: numpy.ndarray, or scipy.sparse.csc_matrix
   
   An invertible sparse or dense matrix.

.. attribute:: ComputeMethod
   :type: string
   :value: 'cholesky'

   Specifies the method of computation. The methods are one of ``'cholsky'``, ``'hutchinson'``, and ``'SLQ'`` (see :ref:`Mathematical Details`).

.. attribute:: UseInverseMatrix
   :type: bool
   :value: True

   This parameter is only applied to the ``'cholesky'`` computing method (see :attr:`ComputeMethod`).
   
   * When set to ``True``, the Cholesky matrix is inverted directly. This approach is computationally expensive, but fast. This option is suitable for small matrices.
   * When set to ``False``, the Cholesky matrix is inverted indirectly by solving succissive linear systems for each column of the right hand side identity matrix. This approach is less computationally expensive and memory efficient, but slow. This approach is suitable for larger matrices.

   For mathematical details of this parameter, see :ref:`Cholesky decomposition method <Cholesky Decomposition Method>`.

.. attribute:: NumIterations
   :type: int
   :value: 20

   This parameter is only applied to ``'hutchinson'`` and ``'SLQ'`` computing methods (see :attr:`ComputeMethod`).

   The number of iterations refers to the number of Monte-Carlo samplings during the randomized trace estimators. With the larger number of Monte-Carlo samples, better numerical convergence is obtained.

   For mathematical details of this parameter, see :ref:`Hutchinson randomized method <Hutchinson Randomized Method>` and :ref:`Stochastic Lanczos quadrature method <Stochastic Lanczos Quadrature Method>`.

.. attribute:: LanczosDegree
   :type: int
   :value: 20

   This parameter is only applied to ``'SLQ'`` computing method (see :attr:`ComputeMethod`).

   The Lanczos degree is the number of Lanczos iterations during the Lanczos tridiagonalization process in stochastic Lanczos quadrature (SLQ) method. Larger Lanczos degree yields better numerical convergence.

   For mathematical detail of this parameter, see :ref:`Stochastic Lanczos quadrature method <Stochastic Lanczos Quadrature Method>`.

.. attribute:: UseLanczosTridiagonalization
   :type: bool
   :value: True

   If ``True``, prints some information about the process.

   This parameter is only applied to ``'SLQ'`` computing method (see :attr:`ComputeMethod`).

   * When set to ``True``, the *Lanczos tri-diagonalization* method is used.
   * When set to ``False``, the *Golub-Kahn-Lanczos bi-diagonalization* method is used.

   For mathematical details of this parameter, see :ref:`Stochastic Lanczos quadrature method <Stochastic Lanczos Quadrature Method>`.

.. attribute:: Verbose
   :type: bool
   :value: False

   If ``True``, prints some information during the process.

====================
Mathematical Details
====================

The three methods of computing the trace is described below. These methods are categorized into two groups:

1. **Exact:** The :ref:`Cholesky decomposition method <Cholesky Decomposition Method>` aims to compute the trace of inverse of a matrix exactly. The exact method is expensive and suitable for only small matrices.
2. **Aproximation:** The :ref:`Hutchinson method <Hutchinson Randomized Method>` and the :ref:`stochastic Lanczos quadrature method <Stochastic Lanczos Quadrature Method>` are *randomized estimation algorithms* that estimate the trace with *Monte-Carlo sampling*. These methods do not compute the trace exactly, but over the iterations, their approximation converges to the true solution. These methods are very suitable for large matrices.

The table below describes which method is suitable for **symmetric** (sym) and/or **positive-definite** (PD) matrices.

+---------------------+--------------------------------------------------------------------------------------------------------+---+---+
|:attr:`ComputeMethod`| Description                                                                                            |Sym|PD |
+=====================+========================================================================================================+===+===+
|``'cholesky'``       |:ref:`Cholesky decomposition <Cholesky Decomposition Method>`                                           |Yes|Yes| 
+---------------------+--------------------------------------------------------------------------------------------------------+---+---+
|``'hutchinson'``     |:ref:`Hutchinson's randomized method <Hutchinson Randomized Method>`                                    |No |No |
+---------------------+--------------------------------------------------------------------------------------------------------+---+---+
|``'SLQ'``            |:ref:`Stochastic Lanczos Quadrature <Stochastic Lanczos Quadrature Method>` (using Lanczos Algorithm)   |Yes|Yes|
+                     +--------------------------------------------------------------------------------------------------------+---+---+
|                     |:ref:`Stochastic Lanczos Quadrature <Stochastic Lanczos Quadrature Method>` (using Golub-Kahn Algorithm)|No |Yes|
+---------------------+--------------------------------------------------------------------------------------------------------+---+---+

Also, in the above table, we recall that the *Lanczos* and *Golub-khan* algorithms can be selected by setting :attr:`UseLanczosTridiagonalizaton` to ``True`` or ``False``, respectively.

-----------------------------
Cholesky Decomposition Method
-----------------------------

The trace of inverse of an invertible matrix :math:`\mathbf{A}` can be computed via

.. math::

    \mathrm{trace}(\mathbf{A}^{-1}) = \| \mathbf{L}^{-1} \|^2_F

where :math:`\| \cdot \|_F` is the Frobenius norm, and the lower-triangular matrix :math:`\mathbf{L}` is the Cholesky decomposition of :math:`\mathbf{A}`, that is

.. math::
    
    \mathbf{A} = \mathbf{L} \mathbf{L}^{\intercal}.

In this package, the Cholesky decomposition computed via `Suite Sparse <https://people.engr.tamu.edu/davis/suitesparse.html>`_ package [Davis-2006]_ (see :ref:`installation <InstallScikitSparse>`). If this package is not installed, the Cholesky decomposition is computed using ``scipy`` package instead.

The term :math:`\| \mathbf{L}^{-1} \|_F` can be computed in two ways:

1. If :attr:`UseInverseMatrix` is set to ``True``, the matrix :math:`\mathbf{L}` is inverted and its Frobenius norm is computed directly. This approach is fast but expensive or often impractical for large matrices.

2. If :attr:`UseInverseMatrix` is set to ``False``, the matrix :math:`\mathbf{L}` is not inverted directly, rather, the linear system

   .. math::

       \mathbf{L} \boldsymbol{x}_i = \boldsymbol{e}_i, \qquad i = 1,\dots,n

   is solved, where :math:`\boldsymbol{e}_i = (0,\dots,0,1,0,\dots,0)^{\intercal}` is a column vector of zeros except its :math:`i`:superscript:`th` entry is one, and :math:`n` is the size of the square matrix :math:`\mathbf{A}`. The solution :math:`\boldsymbol{x}_i` is the :math:`i`:superscript:`th` column of :math:`\mathbf{L}^{-1}`. Then, its Frobenius norm is

   .. math::

       \| \mathbf{L} \|_F^2 = \sum_{i=1}^n \| \boldsymbol{x}_i \|^2.

   The method is memory efficient as the vectors :math:`\boldsymbol{x}_i` do not need to be stored, rather, their norm can be stored in each iteration.

----------------------------
Hutchinson Randomized Method
----------------------------

The Hutchinson's method computes the trace of a matrix as follows.

.. math::

    \mathrm{trace}(\mathbf{A}^{-1}) = \mathbb{E} \left[ \boldsymbol{q}^{\intercal} \mathbf{A}^{-1} \boldsymbol{q} \right]
    \approx \frac{1}{m} \sum_{i=1}^m \boldsymbol{q}_i^{\intercal} \mathbf{A}^{-1} \boldsymbol{q}_i

In Hutchinson's method, the random vector :math:`\boldsymbol{q}` has Rademacher distribution (see [Hutchinson-1990]_, [Gibbs-1997]_, and [Avron-2011]_)

In this module, a random matrix :math:`\mathbf{E}` of the size :math:`(n \times m)`  is randomly generated where its :math:`m` columns have Rademacher distribution. Here, :math:`n` is the size of the square matrix :math:`\mathbf{A}` and :math:`m` will be the number of Monte-Carlo random sampling which is set by :attr:`NumIterations`. We orthogonalize the columns of :math:`\mathbf{E}` by performing the QR decomposition on :math:`\mathbf{E}` as

.. math::

    \mathbf{E} = \mathbf{Q} \mathbf{R}

where :math:`\mathbf{Q}` is orthonormal and :math:`\mathbf{R}` is upper triangular. We use the :math:`m` orthogonal columns of :math:`\mathbf{Q}` instead as the vectors :math:`\boldsymbol{q}_i` to approximate :math:`\mathbb{E}[\boldsymbol{q}^{\intercal} \mathbf{A}^{-1} \boldsymbol{q}]` by

.. math::

    \mathbb{E}[ \boldsymbol{q}^{\intercal} \mathbf{A}^{-1} \boldsymbol{q}] = \frac{n}{m} \sum_{i=1}^m \boldsymbol{q}_i \cdot \boldsymbol{p}_i

where the column vector :math:`\boldsymbol{p}_i` is obtained by solving the linear system :math:`\mathbf{A} \boldsymbol{p}_i = \boldsymbol{q}_i`. The factor :math:`n` in the numerator is due to the fact that the vectors :math:`\boldsymbol{q}_i` are orthonormalized.

------------------------------------
Stochastic Lanczos Quadrature Method
------------------------------------

The stochastic Lanczos quadrature (SLQ) method combines stochastic estimator and the Gauss quadrature technique. A stochastic estimator approximates the trace by

.. math::

    \mathrm{trace}(\mathbf{A}^{-1}) = \mathbb{E} \left[ \boldsymbol{q}^{\intercal} \mathbf{A}^{-1} \boldsymbol{q} \right]
    \approx \frac{n}{m} \sum_{i=1}^m \boldsymbol{q}_i^{\intercal} \mathbf{A}^{-1} \boldsymbol{q}_i

where :math:`\boldsymbol{q}_i` are unit random vectors obtained from random Rademacher distribution that are normalized to unit norm.

In the SLQ method, first, an :math:`(l \times l)` tri-diagonal matrix :math:`\mathbf{T}` is formed by the Lanczos tri-diagonalization of the matrix :math:`\mathbf{A}` (see p. 57 of [Bai-2000]_). The *Lanczos degree* :math:`l` can be set by the parameter :attr:`LanczosDegree`.

The term on the right side in the above is approximated by the Gaussian quadrature (see [Bai-1996]_ and [Golub-2009]_), by

.. math::

   \boldsymbol{q}_i^{\intercal} \mathbf{A}^{-1} \boldsymbol{q}_i = \sum_{j=0}^l \left( \tau_{j1} \right)^2 \frac{1}{\theta_j}.

where :math:`\theta_j` is the :math:`j`:sup:`th` eigenvalue of the tri-diagonalized matrix :math:`\mathbf{T}`. Also, :math:`\tau_{j1}` is the first element of the vector :math:`\boldsymbol{\tau}_j = (\tau_{j1},\dots,\tau_{jn})` where :math:`\boldsymbol{\tau}_j` is the :math:`j`:sup:`th` eigenvector of :math:`\mathbf{T}` (see Algorithm 1 of [Ubaru-2017]_).

Alternatively, instead of the tri-diagonalized matrix :math:`\mathbf{T}`, one might use the bi-diagonalized matrix :math:`\mathbf{B}` that is obtained by Golub-Kahn-Lanczos bi-diagonalization (see p. 143 of [Bai-2000]_, p. 495 of [Golub-1996]_). This way, the above them is computed by

.. math::

   \boldsymbol{q}_i^{\intercal} \mathbf{A}^{-1} \boldsymbol{q}_i = \sum_{j=0}^l \left( \tau_{j1} \right)^2 \frac{1}{\phi_j}.

where here :math:`\phi_j` is the :math:`j`:sup:`th` singular value of :math:`\mathbf{B}`. Also, :math:`\tau_{j1}` denotes the first entry of the :math:`j`:sup:`th` right singular vector of :math:`\mathbf{B}` (see Algorithm 2 of [Ubaru-2017]_).

In this module, by setting :attr:`UseLanczosTridiagonalization` to ``True``, the Lanczos tri-diagonalization method is applied. Whereas if this parameter is set to ``False``, the Golub-Kahn-Lanczos bi-diagonalization is used.

Comparison of Lanczos and Golub-Kahn methods:
    + The Lanczos tri-diagonalization method can only be applied to **symmetric** matrices. Whereas the Golub-Kahn bi-diagonalization method can be used for **non-symmetric** matrices.
    + The Lanczos tri-diagonalization method is almost **twice faster** than the Golub-Kahn bi-diagonalization method on symmetric matrices. This is because the former has one matrix-vector multiplication per iterations, whereas the latter has two.

.. warning::

    When the matrix :math:`\mathbf{A}` is very close to the identity matrix, the Golub-Kahn bi-diagonalization method that is implemented in this module is unstable.

========
Examples
========

------------
Dense Matrix
------------

In the code below, we compare the three computing methods for a small dense matrix of the shape :math:`(20^2,20^2)`.

.. code-block:: python

   >>> # Import modules
   >>> from imate import GenerateMatrix
   >>> from imate import ComputeTraceOfInverse

   >>> # Generate a matrix
   >>> A = GenerateMatrix(NumPoints=20)

   >>> # Try various methods
   >>> T1 = ComputeTraceOfInverse(A,ComputeMethod='cholesky',UseInverseMatrix=False)
   >>> T2 = ComputeTraceOfInverse(A,ComputeMethod='cholesky',UseInverseMatrix=True)
   >>> T3 = ComputeTraceOfInverse(A,ComputeMethod='hutchinson',NumIterations=30)
   >>> T4 = ComputeTraceOfInverse(A,ComputeMethod='SLQ',NumIterations=30,LanczosDegree=30,UseLanczosTridiagonalization=True)
   >>> T5 = ComputeTraceOfInverse(A,ComputeMethod='SLQ',NumIterations=30,LanczosDegree=30,UseLanczosTridiagonalization=False)

The results are shown in the table below. The last column is the elapsed time in seconds. The fifth column is the relative error compared to the Cholesky method. Recall that the Cholesky method computes the trace exactly, hence, it can be used as the benchmark solution. The randomized methods do not much advantage over the exact method for small matrices as their elapsed time higher.

========  ==========  ===================================================  ======  ======  ====
Variable  Method      Options                                              Result  Error   Time
========  ==========  ===================================================  ======  ======  ====
``T1``    Cholesky    without using inverse                                1008.1  0.00\%  0.14
``T2``    Cholesky    using inverse                                        1008.1  0.00\%  0.01
``T3``    Hutchinson  :math:`m = 30`                                       1012.9  0.48\%  0.01
``T4``    SLQ         :math:`m = 30`, :math:`l = 30`, tri-diagonalization  1013.4  0.53\%  0.14
``T5``    SLQ         :math:`m = 30`, :math:`l = 30`, bi-diagonalization   999.19  0.89\%  0.21
========  ==========  ===================================================  ======  ======  ====

The above table can be produced by running the test script |test_script|_, although, the results might be slightly difference due to the random number generator.

.. |test_script| replace:: ``/test/test_ComputeTraceOfInverse.py``
.. _test_script: https://github.com/ameli/imate/blob/main/tests/test_ComputeTraceOfInverse.py

-------------
Sparse Matrix
-------------

In the code below, we compare the three computing methods for a large sparse matrix of the shape :math:`(80^2,80^2)` and sparse density :math:`d = 0.01`.

.. code-block:: python

   >>> # Import modules
   >>> from imate import GenerateMatrix
   >>> from imate import ComputeTraceOfInverse

   >>> # Generate a matrix
   >>> A = GenerateMatrix(NumPoints=80,KernelThreshold=0.05,DecorrelationScale=0.02,UseSparse=True,RunInParallel=True)

   >>> # Try various methods
   >>> T1 = ComputeTraceOfInverse(A,ComputeMethod='cholesky',UseInverseMatrix=False)
   >>> # T2 = ComputeTraceOfInverse(A,ComputeMethod='cholesky',UseInverseMatrix=True) 
   >>> T3 = ComputeTraceOfInverse(A,ComputeMethod='hutchinson',NumIterations=30)
   >>> T4 = ComputeTraceOfInverse(A,ComputeMethod='SLQ',NumIterations=30,LanczosDegree=30,UseLanczosTridiagonalization=True)
   >>> T5 = ComputeTraceOfInverse(A,ComputeMethod='SLQ',NumIterations=30,LanczosDegree=30,UseLanczosTridiagonalization=False)

Note that the line ``T2`` in the above is commented out because the Cholesky method using direct matrix inversion cannot be used on sparse matrices in this module. The results are shown in the table below.

========  ==========  ===================================================  =======  ======  ====
Variable  Method      Options                                              Result   Error   Time
========  ==========  ===================================================  =======  ======  ====
``T1``    Cholesky    without using inverse                                15579.9  0.00\%  119
``T2``    Cholesky    using inverse                                        N/A        N/A   N/A
``T3``    Hutchinson  :math:`m = 30`                                       15559.8  0.13\%  2.21
``T4``    SLQ         :math:`m = 30`, :math:`l = 30`, tri-diagonalization  15576.1  0.02\%  0.95
``T5``    SLQ         :math:`m = 30`, :math:`l = 30`, bi-diagonalization   14243.8  8.58\%  1.57
========  ==========  ===================================================  =======  ======  ====

The advantages of randomized methods can be clearly observed on large sparse matrices. In the above table, the Cholesky method took two minutes, whereas the randomized approaches took one or two seconds to compute. Moreover, a considerable accuracy is achieved with only a few Monte-Carlo sampling (:math:`m = 30`). The SLQ method with bi-diagonalization produced less accurate result compared to the tri-diagonalization method. However, by increasing the Lanczos degree, the accuracy of bi-diagonalization method improves.

==========
References
==========

.. [Davis-2006] Davis, T. A. (2006). Direct Methods for Sparse Linear Systems. Society for Industrial and Applied Mathematics, `doi: 10.1137/1.9780898718881 <https://doi.org/10.1137/1.9780898718881>`_

.. [Hutchinson-1990] Hutchinson, M. F. (1990). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. Communications in Statistics - Simulation and Computation, 19(2), 433–450. `doi: 10.1080/03610919008812866 <https://www.tandfonline.com/doi/abs/10.1080/03610919008812866>`_.

.. [Gibbs-1997] Gibbs, M. & MacKay, D. J. C. (1997). `Efficient Implementation of Gaussian Processes <http://www.inference.org.uk/mackay/gpB.pdf>`_. Technical report, Cavendish Laboratory, Cambridge, UK

.. [Avron-2011] Avron, H. and Toledo, S. (2011). Randomized Algorithms for Estimating the Trace of an Implicit Symmetric Positive Semi-Definite Matrix. *Journal ofA ACM*, volume 58, No. 2, Association for Computing Machinery. New York, NY, USA. `doi: 10.1145/1944345.1944349 <https://doi.org/10.1145/1944345.1944349>`_.

.. [Bai-1996] Bai, Z., Fahey, G., and Golub, G. (1996). Some large-scale matrix computation problems. *Journal of Computational and Applied Mathematics*, 74(1), 71 – 89. `doi : 10.1016/0377-0427(96)00018-0 <https://doi.org/10.1016/0377-0427(96)00018-0>`_

.. [Bai-2000] Bai, Z, Demmel, J., Dongarra, J., Ruhe, A., van der Vorst, H. (2000). Templates for the Solution of Algebraic Eigenvalue Problem, A Practical Guide. Society for Industrial and Applied Mathematics. `doi:10.1137/1.9780898719581 <https://doi.org/10.1137/1.9780898719581>`_

.. [Golub-1996] Golub, G. H. & Van Loan, C. F. (1996). Matrix Computations. Johns Hopkins Studies in the Mathematical Sciences. Johns Hopkins University Press. `ISBN: 9781421407944 <https://jhupbooks.press.jhu.edu/title/matrix-computations>`_

.. [Golub-2009] Golub, G. H. & Meurant, G. (2009). Matrices, Moments and Quadrature with Applications. USA: Princeton University Press. `doi: 10.1007/s10208-010-9082-0 <https://doi.org/10.1007/s10208-010-9082-0>`_

.. [Ubaru-2017] Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of :math:`\mathrm{tr}(f(A))` via stochastic Lanczos quadrature. *SIAM Journal on Matrix Analysis and Applications*, 38(4), 1075-1099. `doi: 10.1137/16M1104974 <https://doi.org/10.1137/16M1104974>`_

===
API
===

--------------
Main Interface
--------------

.. automodapi:: imate.ComputeTraceOfInverse

-------
Modules
-------

.. automodapi:: imate.ComputeTraceOfInverse.CholeskyMethod
.. automodapi:: imate.ComputeTraceOfInverse.HutchinsonMethod
.. automodapi:: imate.ComputeTraceOfInverse.StochasticLanczosQuadratureMethod
.. automodapi:: imate._LinearAlgebra.LinearSolver
