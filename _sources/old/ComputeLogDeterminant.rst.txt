.. _ComputeLogDeterminant_UserGuide:

****************************************************************
Compute Log-Determinant (:mod:`imate.ComputeLogDeterminant`)
****************************************************************

The sub-package :mod:`imate.ComputeLogDeterminant` computes the log-determinant of an invertible matrix. 

=====
Usage
=====

.. code-block:: python

   >>> from imate import GenerateMatrix
   >>> from imate import ComputeLogDeterminant
   
   >>> # Generate a symmetric positive-definite matrix of the shape (20**2,20**2)
   >>> A = GenerateMatrix(NumPoints=20)
   
   >>> # Compute trace of inverse
   >>> logdet = ComputeLogDeterminant(A)

In the above, the class :class:`GenerateMatrix <imate.GenerateMatrix>` produces a sample matrix for test purposes (see :ref:`Generate Matrix <GenerateMatrix>` for details).

The :mod:`ComputeLogDeterminant <imate.ComputeLogDeterminant>` module in the above code employs the *Cholesky method* by default to compute the log-determinant. However, other methods can be employed by setting :attr:`ComputeMethod` argument according to the table below.

=====================  ======================================================  ============  =============  =============
:attr:`ComputeMethod`  Description                                             Matrix size   Matrix type    Results       
=====================  ======================================================  ============  =============  =============
``'cholesky'``         :ref:`Cholesky decomposition <MathDetails_Cholesky>`    small         dense, sparse  exact          
``'SLQ'``              :ref:`Stochastic Lanczos Quadrature <MathDetails_SLQ>`  small, large  dense, sparse  approximation
=====================  ======================================================  ============  =============  =============

In the following example, we apply the *SLQ randomized estimator* method:

.. code-block:: python

   >>> # Using SLQ method with 20 Monte-Carlo iterations
   >>> logdet = ComputeLogDeterminant(A,ComputeMethod='SLQ',NumIterations=20)

Each of the methods in the above accepts some options. For instance, the SLQ method accepts :attr:`NumIterations` argument, which sets the number of Monte-Carlo trials. To see the detailed list of all arguments for each method, see :ref:`Parameters <Parameters_LogDet>` and the `API <https://ameli.github.io/imate/_modules/modules.html>`__ of the package.

.. _Parameters_LogDet:

==========
Parameters
==========

The :mod:`imate.ComputeLogDeterminant` module accepts the following attributes as input argument.

.. attribute:: A
   :type: numpy.ndarray, or scipy.sparse.csc_matrix
   
   An invertible sparse or dense matrix.

.. attribute:: ComputeMethod
   :type: string
   :value: 'cholesky'

   Specifies the method of computation. The methods are one of ``'cholsky'`` and ``'SLQ'`` (see :ref:`Mathematical Details <MathDetails_LogDet>`).

.. attribute:: NumIterations
   :type: int
   :value: 20

   This parameter is only applied to ``'SLQ'`` computing methods (see :attr:`ComputeMethod`).

   The number of iterations refers to the number of Monte-Carlo samplings during the randomized estimators. With the larger number of Monte-Carlo samples, better numerical convergence is obtained.

   For mathematical details of this parameter, see :ref:`Stochastic Lanczos quadrature method <MathDetails_SLQ>`.

.. attribute:: LanczosDegree
   :type: int
   :value: 20

   This parameter is only applied to ``'SLQ'`` computing method (see :attr:`ComputeMethod`).

   The Lanczos degree is the number of Lanczos iterations during the Lanczos tridiagonalization process in stochastic Lanczos quadrature (SLQ) method. Larger Lanczos degree yields better numerical convergence.

   For mathematical detail of this parameter, see :ref:`Stochastic Lanczos quadrature method <MathDetails_SLQ>`.

.. attribute:: UseLanczosTridiagonalization
   :type: bool
   :value: False

   This parameter is only applied to ``'SLQ'`` computing method (see :attr:`ComputeMethod`).

   * When set to ``True``, the *Lanczos tri-diagonalization* method is used.
   * When set to ``False``, the *Golub-Kahn-Lanczos bi-diagonalization* method is used.

   For mathematical details of this parameter, see :ref:`Stochastic Lanczos quadrature method <MathDetails_SLQ>`.
   
.. attribute:: Verbose
   :type: bool
   :value: False

   If ``True``, prints some information about the process.


.. _MathDetails_LogDet:

====================
Mathematical Details
====================

The three methods of computing the log-determinant is described below. These methods are categorized into two groups:

1. **Exact:** The :ref:`Cholesky decomposition method <MathDetails_Cholesky>` aims to compute the log-determinant of a matrix exactly. The exact method is expensive and suitable for only small matrices.
2. **Aproximation:** The :ref:`stochastic Lanczos quadrature method <MathDetails_SLQ>` are *randomized estimation algorithms* that estimate the log-determinant with *Monte-Carlo sampling*. These methods do not compute the determinant exactly, but over the iterations, their approximation converges to the true solution. These methods are very suitable for large matrices.

The table below describes which method is suitable for **symmetric** (sym) and/or **positive-definite** (PD) matrices.

+-----------------------+-------------------------------------------------------------------------------------+-----+-----+
| :attr:`ComputeMethod` | Description                                                                         | Sym | PD  |
+=======================+=====================================================================================+=====+=====+
| ``'cholesky'``        | :ref:`Cholesky decomposition <MathDetails_Cholesky>`                                | Yes | Yes |
+-----------------------+-------------------------------------------------------------------------------------+-----+-----+
| ``'SLQ'``             | :ref:`Stochastic Lanczos Quadrature <MathDetails_SLQ>` (using Lanczos Algorithm)    | Yes | Yes |
+-----------------------+-------------------------------------------------------------------------------------+-----+-----+
|                       | :ref:`Stochastic Lanczos Quadrature <MathDetails_SLQ>` (using Golub-Kahn Algorithm) | No  | Yes |
+-----------------------+-------------------------------------------------------------------------------------+-----+-----+

Also, in the above table, we recall that the *Lanczos* and *Golub-khan* algorithms can be selected by setting :attr:`UseLanczosTridiagonalizaton` to ``True`` or ``False``, respectively.

.. _MathDetails_Cholesky:

-----------------------------
Cholesky Decomposition Method
-----------------------------

The log-determinant of an invertible matrix :math:`\mathbf{A}` can be computed via

.. math::

    \log | \mathbf{A} | = 2 \mathrm{trace}( \log \mathrm{diag}(\mathbf{L})).

In this package, the Cholesky decomposition computed via `Suite Sparse <https://people.engr.tamu.edu/davis/suitesparse.html>`_ package [Davis-2006]_ (see :ref:`installation <InstallScikitSparse>`). If this package is not installed, the Cholesky decomposition is computed using ``scipy`` package instead.

.. _MathDetails_SLQ:

------------------------------------
Stochastic Lanczos Quadrature Method
------------------------------------

The stochastic Lanczos quadrature (SLQ) method combines stochastic estimator and the Gauss quadrature technique. A stochastic estimator approximates the log-determinant by

.. math::

    \log | \mathbf{A} | = \mathrm{trace}(\log \mathbf{A}) = \mathbb{E} \left[ \boldsymbol{q}^{\intercal} (\log \mathbf{A}) \boldsymbol{q} \right]
    \approx \frac{n}{m} \sum_{i=1}^m \boldsymbol{q}_i^{\intercal} (\log \mathbf{A}) \boldsymbol{q}_i

where :math:`\boldsymbol{q}_i` are unit random vectors obtained from random Rademacher distribution that are normalized to unit norm.

In the SLQ method, first, an :math:`(l \times l)` tri-diagonal matrix :math:`\mathbf{T}` is formed by the Lanczos tri-diagonalization of the matrix :math:`\mathbf{A}` (see p. 57 of [Bai-2000]_). The *Lanczos degree* :math:`l` can be set by the parameter :attr:`LanczosDegree`.

The term on the right side in the above is approximated by the Gaussian quadrature (see [Bai-1996]_ and [Golub-2009]_), by

.. math::

   \boldsymbol{q}_i^{\intercal} (log \mathbf{A}) \boldsymbol{q}_i = \sum_{j=0}^l \left( \tau_{j1} \right)^2 \log \theta_j.

where :math:`\theta_j` is the :math:`j`:sup:`th` eigenvalue of the tri-diagonalized matrix :math:`\mathbf{T}`. Also, :math:`\tau_{j1}` is the first element of the vector :math:`\boldsymbol{\tau}_j = (\tau_{j1},\dots,\tau_{jn})` where :math:`\boldsymbol{\tau}_j` is the :math:`j`:sup:`th` eigenvector of :math:`\mathbf{T}` (see Algorithm 1 of [Ubaru-2017]_).

Alternatively, instead of the tri-diagonalized matrix :math:`\mathbf{T}`, one might use the bi-diagonalized matrix :math:`\mathbf{B}` that is obtained by Golub-Kahn-Lanczos bi-diagonalization (see p. 143 of [Bai-2000]_, p. 495 of [Golub-1996]_). This way, the above them is computed by

.. math::

   \boldsymbol{q}_i^{\intercal} \mathbf{A}^{-1} \boldsymbol{q}_i = \sum_{j=0}^l \left( \tau_{j1} \right)^2 \log \phi_j.

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
   >>> from imate import ComputeLogDeterminant

   >>> # Generate a matrix
   >>> A = GenerateMatrix(NumPoints=20)

   >>> # Try various methods
   >>> D1 = ComputeLogDeterminant(A,ComputeMethod='cholesky')
   >>> D2 = ComputeLogDeterminant(A,ComputeMethod='SLQ',NumIterations=50,LanczosDegree=30,UseLanczosTridiagonalization=True)
   >>> D3 = ComputeLogDeterminant(A,ComputeMethod='SLQ',NumIterations=50,LanczosDegree=30,UseLanczosTridiagonalization=False)

The results are shown in the table below. The last column is the elapsed time in seconds. The fifth column is the relative error compared to the Cholesky method. Recall that the Cholesky method computes the log-determinant exactly, hence, it can be used as the benchmark solution. The randomized methods do not much advantage over the exact method for small matrices as their elapsed time higher.

========  ==========  ===================================================  ======  ======  ====
Variable  Method      Options                                              Result  Error   Time
========  ==========  ===================================================  ======  ======  ====
``D1``    Cholesky    N/A                                                  41.675  0.00\%  0.00
``D2``    SLQ         :math:`m = 50`, :math:`l = 30`, tri-diagonalization  38.104  8.57\%  0.24
``D3``    SLQ         :math:`m = 50`, :math:`l = 30`, bi-diagonalization   41.466  0.50\%  0.35
========  ==========  ===================================================  ======  ======  ====

The above table can be produced by running the test script |test_script2|_, although, the results might be slightly difference due to the random number generator.

.. |test_script2| replace:: ``/test/test_ComputeLogDeterminant.py``
.. _test_script2: https://github.com/ameli/imate/blob/main/tests/test_ComputeLogDeterminant.py

-------------
Sparse Matrix
-------------

In the code below, we compare the three computing methods for a large sparse matrix of the shape :math:`(50^2,50^2)` and sparse density :math:`d = 0.01`.

.. code-block:: python

   >>> # Import modules
   >>> from imate import GenerateMatrix
   >>> from imate import ComputeLogDeterminant

   >>> # Generate a matrix
   >>> A = GenerateMatrix(NumPoints=50,KernelThreshold=0.03,DecorrelationScale=0.03,UseSparse=True,RunInParallel=True)

   >>> # Try various methods
   >>> D1 = ComputeLogDeterminant(A,ComputeMethod='cholesky')
   >>> D2 = ComputeLogDeterminant(A,ComputeMethod='SLQ',NumIterations=50,LanczosDegree=30,UseLanczosTridiagonalization=True)
   >>> D3 = ComputeLogDeterminant(A,ComputeMethod='SLQ',NumIterations=50,LanczosDegree=30,UseLanczosTridiagonalization=False)

The results are shown in the table below.

========  ==========  ===================================================  ======  ======  ====
Variable  Method      Options                                              Result  Error   Time
========  ==========  ===================================================  ======  ======  ====
``D1``    Cholesky    N/A                                                  52.052  0.00\%  2.44
``D2``    SLQ         :math:`m = 50`, :math:`l = 30`, tri-diagonalization  51.955  0.19\%  0.76
``D3``    SLQ         :math:`m = 50`, :math:`l = 30`, bi-diagonalization   51.711  0.66\%  1.35
========  ==========  ===================================================  ======  ======  ====

The advantages of randomized methods can be clearly observed on large sparse matrices. In the above table, the Cholesky method took two minutes, whereas the randomized approaches took one or two seconds to compute. Moreover, a considerable accuracy is achieved with only a few Monte-Carlo sampling (:math:`m = 50`). The SLQ method with bi-diagonalization produced less accurate result compared to the tri-diagonalization method. However, by increasing the Lanczos degree, the accuracy of bi-diagonalization method improves.

==========
References
==========

.. [Davis-2006] Davis, T. A. (2006). Direct Methods for Sparse Linear Systems. Society for Industrial and Applied Mathematics, `doi: 10.1137/1.9780898718881 <https://doi.org/10.1137/1.9780898718881>`_

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

.. automodapi:: imate.ComputeLogDeterminant

-------
Modules
-------

.. automodapi:: imate._LinearAlgebra.LinearSolver
