.. _overview:

Overview
********

What |project| Does?
====================

The main purpose of |project| is to estimate the algebraic quantity

.. math::
   :label: tracef-1

    \mathrm{trace} \left( f(\mathbf{A}) \right),

where :math:`\mathbf{A}` is a square matrix, :math:`f` is a matrix function, and :math:`\mathrm{trace}(\cdot)` is the trace operator. |project| can also compute variants of :math:numref:`tracef-1`, such as

.. math::
   :label: tracef-2

   \mathrm{trace} \left(\mathbf{B} f(\mathbf{A}) \right),

and

.. math::
   :label: tracef-3

    \mathrm{trace} \left(\mathbf{B} f(\mathbf{A}) \mathbf{C} f(\mathbf{A}) \right),

where :math:`\mathbf{B}` and :math:`\mathbf{C}` are matrices. Other variations include the cases where :math:`\mathbf{A}` is replaced by :math:`\mathbf{A}^{\intercal} \mathbf{A}` in the above expressions.

Where |project| Can Be Applied?
===============================

Despite |project|'s aim being tailored to specific algebraic computations, it addresses a demanding and challenging computational task with wide a range of applications. Namely, the expression :math:numref:`tracef-1` is ubiquitous in a variety of applications [R1]_, and in fact, it is often the most computationally expensive term in such applications. Some common examples of :math:`f` lead to the following forms of :math:numref:`tracef-1`:

.. glossary::

    Log-Determinant

        If :math:`f: \lambda \mapsto \log \vert \lambda \vert`, then :math:`\mathrm{trace} \left(f(\mathbf{A}) \right) = \log \vert \det \mathbf{A} \vert` is the log-determinant of :math:`\mathbf{A}`, which frequently appears in statistics and machine learning, particularly in `log-likelihood functions` [R2]_.

    Trace of Matrix Powers

        If :math:`f: \lambda \mapsto \lambda^p`, :math:`p \in \mathbb{R}`, then :math:numref:`tracef-1` is :math:`\mathrm{trace} (\mathbf{A}^p)`. Interesting cases are the negative powers, such as the trace of inverse, :math:`\mathrm{trace} (\mathbf{A}^{-1})`, where :math:`\mathbf{A}^{-1}` is implicitly known. These class of functions frequently appears in statistics and machine learning. For instance, :math:`p=-1` and :math:`p=-2` appear in the *Jacobian* and *Hessian* of log-likelihood functions, respectively [R3]_.

    Schatten :math:`p`-norm and Schatten :math:`p`-anti-norm

        If :math:`f: \lambda \mapsto \lambda^{\frac{p}{2}}`, then :math:`\mathrm{trace} (\mathbf{A}^{\frac{p}{2}})` is the Schatten :math:`p`-norm (if :math:`p > 0`), and is the Schatten :math:`p`-anti-norm (if :math:`p < 0`). Schatten norm has applications in `rank-constrained optimization` in machine learning.

    Eigencount and Numerical Rank

        If :math:`f: \lambda \mapsto \mathbf{1}_{[a,b]}(\lambda)` is the indicator (step) function in the interval :math:`[a, b]`, then :math:`\mathrm{trace}(\mathbf{1}(\mathbf{A}))` estimates the number of non-zero eigenvalues of :math:`\mathbf{A}` in that interval, which is an inexpensive method to estimate the rank of a large matrix. Eigencount is closely related to the `Principal Component Analysis (PCA)` and `low-rank approximations` in machine learning.

    Estrada Index of Graphs

        If :math:`f: \lambda \mapsto \exp(\lambda)`, then :math:`\mathrm{trace} \left( \exp(\mathbf{A}) \right)` is the `Estrada index <https://en.wikipedia.org/wiki/Estrada_index>`_ of :math:`\mathbf{A}`, which has applications in computational biology such as in `protein folding`.

    Spectral Density

        If :math:`f: \lambda \mapsto \delta(\lambda - \mu)`, where :math:`\delta(\lambda)` is the Dirac's delta function, then :math:`\mathrm{trace} \left( f(\mathbf{A})\right)` yields the spectral density of the eigenvalues of :math:`\mathbf{A}`. Estimating the spectral density of matrices, which is also known as `Density of States (DOS) <https://en.wikipedia.org/wiki/Density_of_states>`_, is a common problem in solid state physics.

Randomized Algorithms For Massive Data
======================================

Calculating :math:numref:`tracef-1` and its variants is a computational challenge when

* Matrices are very large. In practical applications, matrix size could range from million to billion.
* :math:`f(\mathbf{A})` cannot be computed directly, or only known *implicitly*, such as :math:`\mathbf{A}^{-1}`.

|project| employs **randomized (stochastic) algorithms** [R5]_ to fulfill the demand for computing :math:numref:`tracef-1` on massive data. Such classes of algorithms are fast and scalable to large matrices. |project| implements the following randomized algorithms:

.. glossary::

    Hutchinson's Method

        Hutchinson technique is the earliest randomized method employed to estimate the trace of the inverse of an invertible matrix [R6]_. |project| implements Hutchinson's method to compute :math:numref:`tracef-1`, :math:numref:`tracef-2`, and :math:numref:`tracef-3` for :math:`f(\lambda) = \lambda^{-1}`.

    Stochastic Lanczos Quadrature Method

        The Stochastic Lanczos Quadrature (SLQ) method [R7]_ combines two of the greatest algorithms of the century in applied mathematics, namely the Monte-Carlo method and Lanczaos algorithm (also, Golub-Kahn-Lanczos algoritm) [R8]_ [R9]_, together with Gauss quadrature to estimate :math:numref:`tracef-1` for an analytic function :math:`f` and symmetric positive-definite matrix :math:`\mathbf{A}`. |project| provides an efficient and scalable implementation of SLQ method.

Along with the randomized methods, |project| also provides direct (non-stochastic) methods which are only for benchmarking purposes to test the accuracy of the randomized methods on small matrices.

Applications in Optimization
============================

A unique and novel feature of |project| is the ability to interpolate the trace of the arbitrary functions of the affine matrix function :math:`t \mapsto \mathbf{A} + t \mathbf{B}`. Such an affine matrix function appears in variety of optimization formulations in machine learning. Often in these applications, the hyperparameter :math:`t` has to be tuned. To this end, the optimization scheme should compute

.. math::

    t \mapsto \mathrm{trace} \left(f(\mathbf{A} + t \mathbf{B}) \right),

for a large number of input hyperparameter :math:`t \in \mathbb{R}`. See common examples of the function :math:`f` in :ref:`Overview <overview>`.

Instead of directly computing the above function for every :math:`t`, |project| can interpolate the above function for a wide range of :math:`t` with a high accuracy with only a handful number of evaluation of the above function. This solution can enhance the processing time of an optimization scheme by several orders of magnitude with only less than :math:`1 \%` error [R4]_.


Petascale Computing
===================

The core of |project| is a high-performance C++/CUDA library capable of performing on parallel CPUs or GPU farm with multiple GPU devices. |project| can fairly perform at petascale, for instance on a cluster node of twenty GPUs with NVIDIA Hopper architecture, each with 60 TFLOPS.

For a gallery of the performance of |project| on GPU and CPU on massive matrices, see :ref:`Performance <index_performance>`.

References
==========

.. [R1] Ubaru, S., Saad, Y. (2018). *Applications of Trace Estimation Techniques*. In: High-Performance Computing in Science and Engineering. HPCSE 2017. Lecture Notes in Computer Science, vol 11087. Springer, Cham. `doi: 10.1007/978-3-319-97136-0_2 <https://doi.org/10.1007/978-3-319-97136-0_2>`_

.. [R2] Ameli, S., and Shadden, S. C. (2022). *A Singular Woodbury and Pseudo-Determinant Matrix Identities and Application to Gaussian Process Regression*. `arXiv: 2207.08038 [math.ST] <https://arxiv.org/abs/2207.08038>`_.

.. [R3] Ameli, S., and Shadden, S. C. (2022). *Noise Estimation in Gaussian Process Regression*. `arXiv: 2206.09976 [cs.LG] <https://arxiv.org/abs/2206.09976>`_

.. [R4] Ameli, S., and Shadden. S. C. (2022). *Interpolating Log-Determinant and Trace of the Powers of Matrix* :math:`\mathbf{A} + t \mathbf{B}`. Statistics and Computing 32, 108. `DOI <https://doi.org/10.1007/s11222-022-10173-4>`__


.. [R5] Mahoney, M. W. (2011). *Randomized algorithms for matrices and data*. `arXiv: 1104.5557 [cs.DS] <https://arxiv.org/abs/1104.5557>`_.

.. [R6]  Hutchinson, M. F. (1990). *A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines*. Comm. Statist. Simulation Comput. Volume 19, Number 2, pp. 433-450. Taylor \& Francis. `doi: 10.1080/03610919008812866 <https://www.tandfonline.com/doi/abs/10.1080/03610919008812866>`_.

.. [R7] Golub, G. H., and Meurant, G. (2010). *Matrices, Moments and Quadrature with Applications*. Princeton University Press. isbn: 0691143412. `jstor.org/stable/j.ctt7tbvs <http://www.jstor.org/stable/j.ctt7tbvs>`_.

.. [R8] Dongarra, J., and Sullivan, F. (2000). *The Top 10 Algorithms. Computing in Science and Eng*. 2, 1, pp. 22–23. `doi: 10.1109/MCISE.2000.814652 <https://doi.org/10.1109/MCISE.2000.814652>`_.

.. [R9] Higham, N. J., (2016). `Nicholas J. Higham on the top 10 algorithms in applied mathematics <https://press.princeton.edu/ideas/nicholas-higham-on-the-top-10-algorithms-in-applied-mathematics>`_. The Princeton Companion to Applied Mathematics. Princeton University Press. isbn: 786842300.
