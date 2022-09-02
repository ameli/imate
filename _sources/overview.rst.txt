.. _overview:

Overview
********

What |project| Does
===================

The main functionality of |project| is to estimate the algebraic quantity

.. math::
   :label: tracef

    \mathrm{trace} \left( f(\mathbf{A}) \right),

where :math:`\mathbf{A}` is a square matrix, :math:`f` is a matrix function, and :math:`\mathrm{trace}(\cdot)` is the trace operator. |project| can also compute variants of :math:numref:`tracef`, such as

.. math::
   \mathrm{trace} \left(\mathbf{B} f(\mathbf{A}) \right),

and

.. math::

    \mathrm{trace} \left(\mathbf{B} f(\mathbf{A}) \mathbf{C} f(\mathbf{A}) \right),

where :math:`\mathbf{B}` and :math:`\mathbf{C}` are matrices. Other variations include the cases where :math:`\mathbf{A}` is replaced by :math:`\mathbf{A}^{\intercal} \mathbf{A}` in the above expressions.


Common Matrix Functions
=======================

The expression :math:numref:`tracef` is ubiquitous in variety of important applications [R1]_, and in fact, it is often the most computationally expensive term in such applications [R2]_. Some common examples of :math:`f` leads to the following forms of :math:numref:`tracef`:

.. glossary::

    Log-Determinant

        If :math:`f: x \mapsto \log \vert x \vert`, then :math:`\mathrm{trace} \left(f(\mathbf{A}) \right) = \log \vert \det \mathbf{A} \vert` is the log-determinant of :math:`\mathbf{A}`, which frequently appears in statistics and machine learning, particularly in log-likelihood functions.

    Trace of Matrix Powers

        If :math:`f: x \mapsto x^p`, :math:`p \in \mathbb{R}`, then :math:numref:`tracef` is :math:`\mathrm{trace} (\mathbf{A}^p)`. Interesting cases are the nagative powers, such as the trace of inverse, :math:`\mathrm{trace} (\mathbf{A}^{-1})`, where :math:`\mathbf{A}^{-1}` is implicitly known. This function frequently appears in statistics and machine learning. In particular, :math:`p=-1` and :math:`p=-2` appear in the **Jacobian** and **Hessian** of log-likelihood functions, respectively.

    Schatten :math:`p`-norm and Schatten :math:`p`-anti-norm

        If :math:`f: x \mapsto x^{\frac{p}{2}}``, then :math:`\mathrm{trace} (\mathbf{A}^{\frac{p}{2}})` is the Schatten :math:`p`-norm (if :math:`p > 0`), and is the Schatten :math:`p`-anti-norm (if :math:`p < 0`). Schatten norm has applications in rank-constrained optimization in machine learning.

    Eigencount and Numerical Rank

        If :math:`f: x \mapsto \mathbf{1}_{[a,b]}(x)` is the indicator (step) function in the interval :math:`[a, b]`, then :math:`\mathrm{trace}(\mathbf{1}(\mathbf{A}))` estimates number of non-zero eigenvalues of :math:`\mathbf{A}` in that interval, which is an inexpensive method to estimate the rank of a large matrix. Eigencount is closely related to the Principal Component Analysis (PCA) and low-rank approximations in machine learning.

    Extrada Index of Graphs

        If :math:`f: x \mapsto \exp(x)`, then :math:`\mathrm{trace} \left( \exp(\mathbf{A}) \right)` is the `Estrada index <https://en.wikipedia.org/wiki/Estrada_index>`_ of :math:`\mathbf{A}`, which has applications in computational biology such as in protein folding.

    Spectral Density

        If :math:`f: x \mapsto \delta(x - \lambda)`, where :math:`\delta(x)` is the Dirac's delta function, then :math:`\mathrm{trace} \left( f(\mathbf{A})\right)` yields the spectral density of the eigenvalues of :math:`\mathbf{A}`. Estimating the spectral density of matrices, which is also known as `Density of States (DOS) <https://en.wikipedia.org/wiki/Density_of_states>`_, is a common problem in solid state physics.

How |project| Computes
======================

Calculating :math:numref:`tracef` and its variants is a computational challenge when

* Matrices are very large. In practical applications, matrix size could ranging from million to billion.
* :math:`f(\mathbf{A})` cannot be computed directly, or only known *implicitly*, such as :math:`\mathbf{A}^{-1}`.

|project| employs **randomized (stochastic) algorithms** to fulfill the demand for computing :math:numref:`tracef` on massive data. Such class of algorithms are fast and scalable to large matrices. |project| implements the following randomized algorithms:

* Hutchinson's method.
* Stochastic Lanczos Quadrature method.


Interpolators
=============

A novel feature of |project| is the interpolation of the above quantities when the matrix is a **one-parameter affine operator**

.. math::

    \mathbf{A}(t): t \mapsto \mathbf{A} + t \mathbf{B}.

In such cases, |project| can interpolate :math:`\mathrm{trace} f(\mathbf{A}(t))` for a large logarithmic range of the parameter :math:`t`. This novel method is very useful in optimization of hyperparameters of models with such affine matrix formulations [R3]_.

References
==========

.. [R1] Ubaru, S., Saad, Y. (2018). Applications of Trace Estimation Techniques. In: High Performance Computing in Science and Engineering. HPCSE 2017. Lecture Notes in Computer Science, vol 11087. Springer, Cham. `doi: 10.1007/978-3-319-97136-0_2 <https://doi.org/10.1007/978-3-319-97136-0_2>`_

.. [R2] Ameli, S. and Shadden, S. C. (2022). Noise Estimation in Gaussian Process Regression. `arXiv: 2206.09976 [cs.LG] <https://arxiv.org/abs/2206.09976>`_

.. [R3] Ameli, S. and Shadden, S. C. (2022). Interpolating Log-Determinant and Trace of the Powers of Matrix :math:`\mathbf{A} + t \mathbf{B}`. `arXiv: 2009.07385 [math.NA] <https://arxiv.org/abs/2009.07385>`_

