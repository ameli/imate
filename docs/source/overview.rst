.. _overview:

Overview
********

What |project| Does
===================

The main functionality of |project| is to estimate the quantity

.. math::
   :label: tracef

    \mathrm{trace} \left( f(\mathbf{A}) \right),

where :math:`\mathbf{A}` is a square matrix and :math:`f` is a matrix function. |project| can also compute variants of :math:numref:`tracef`, such as

.. math::
   \mathrm{trace} \left(f(\mathbf{A}) \mathbf{B} \right),

and

.. math::

    \mathrm{trace} \left(f(\mathbf{A}) \mathbf{B} f(\mathbf{A}) \mathbf{C} \right).


Common Matrix Functions
=======================

Some examples of :math:`f` in :math:numref:`tracef` leads to the following common applications:

.. glossary::

    Log-Determinant

        If :math:`f: x \mapsto \log \vert x \vert`, then :math:`\mathrm{trace} \left(f(\mathbf{A}) \right) = \log \vert \det \mathbf{A} \vert` is the log-determinant of :math:`\mathbf{A}` which frequently appears in statistics and machine leanring.

    Trace of Matrix Powers

        If :math:`f: x \mapsto x^p`, :math:`p \in \mathbb{R}`, then :math:numref:`tracef` is :math:`\mathrm{trace} (\mathbf{A}^p)`. This can include the nagative powers, such as the trace of inverse, :math:`\mathrm{trace} (\mathbf{A}^{-1})`. This function frequenity appears in statistics and machine leanring.

    Schatten :math:`p`-norm and Schatten :math:`p`-anti-norm

        If :math:`f: x \mapsto x^{\frac{p}{2}}``, then :math:`\mathrm{trace} (\mathbf{A}^{\frac{p}{2}})` is the Schatten :math:`p`-norm (if :math:`p > 0`), and is the Schatten :math:`p`-anti-norm (if :math:`p < 0`).

    Eigencount and Numerical Rank

        If :math:`f: x \mapsto \mathbf{1}_{[a,b]}(x)` is the indicator (step) function in the interval :math:`[a, b]`, then :math:`\mathrm{trace}(\mathbf{1}(\mathbf{A}))` estimates number of eigenvalues of :math:`\mathbf{A}` in the latter interval, which is an inexpensive method of estimating the rank of a large matrix.

    Extrada Index of Graphs

        If :math:`f: x \mapsto \exp(x)`, then :math:`\mathrm{trace} \left( f(\mathbf{A}) \right)` is the Estrada index of :math:`\mathbf{A}`, which has many applications in computational biology.

    Spectral Density

        if :math:`f: x \mapsto \delta(x - \lambda)`, where :math:`\delta(x)` is the Dirac's delta function, then :math:`\mathrm{trace} \left( f(\mathbf{A})\right)` obtains the spectral density of the eigenvalues of :math:`\mathbf{A}`.

How |project| Computes
======================

It is assumed that

* Matrices are very large.
* :math:`f(\mathbf{A})` cannot be computed directly, or only known *implicitly*.

Because of these, instead of computing these quantities directly, |project| employs **randomized (stochastic) algorithms** to estimate these quantities using Monte-Carlo sampling, such as by

* Hutchinson's method,
* Stochastic Lanczos Quadrature method.

Such class of methods are very fast and scalable to massive matrices.


Interpolators
=============

A novel feature of |project| is the interpolation of the above quantities when the matrix is a **one-parameter affine operator**

.. math::

    \mathbf{A}(t): t \mapsto \mathbf{A} + t \mathbf{B}.

In such cases, |project| can interpolate :math:`\mathrm{trace} f(\mathbf{A}(t))` for a large logarithmic range of the parameter :math:`t`. This novel method is very useful in optimization of hyperparameters of models with such affine matrix formulations.

Applications
============

Trace of matrix functions are **ubiquitous** in machine learning applications, and in fact, they are **the most computationally expensive** mathematical terms in such applications.

Take for instance the :math:`n`-dimensional normal distribution :math:`\boldsymbol{y} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})` by

.. math::

    p(\boldsymbol{y} \vert \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{\sqrt{(2 \pi)^n}} \vert \boldsymbol{\Sigma} \vert^{-\frac{1}{2}} \exp \left(-\frac{1}{2} (\boldsymbol{y} - \boldsymbol{\mu})^{\intercal} \boldsymbol{\Sigma}^{-1} (\boldsymbol{y} - \boldsymbol{\mu}) \right)
