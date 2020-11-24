************
Introduction
************

This package computes the trace of inverse of two forms of matrices:

===============
1. Fixed Matrix
===============

For an invertible matrix |image01|, this package computes |image02| for both sparse and dense matrices.

**Computing Methods:**
    The following methods are implemented:

    #. **Cholesky Decomposition**:  accurate, suitable for small matrices.
    #. **Hutchinson's Randomized Estimator**: approximation, suitable for large matrices.
    #. **Stochastic Lanczos Quadrature**: approximation, suitable for large matrices.

=======================================
2. One-Parameter Affine Matrix Function
=======================================

Consider the matrix function |image05|, where |image01| and |image03| are symmetric and positive-definite matrices and :math:`t` is a real parameter. This package can interpolate the function

|image06|


**Application:**
    The above function is featured in a wide range of applications in statistics and machine learning. Particular applications are in model selection and optimizing hyperparameters with gradient-based maximum likelihood methods. In such applications, computing the above function is often a computational challenge for large matrices. Often, this function is evaluated for a wide range of the parameter :math:`t` while :math:`\mathbf{A}` and :math:`\mathbf{B}` remain fixed. As such, an interpolation scheme enables fast computation of the function.

**Interpolation Methods**
    Various interpolation methods of the above function are implemented in this package, namely

    #. **Eigenvalues Method**
    #. **Monomial Basis Functions**
    #. **Root Monomial Basis Functions**
    #. **Rational Polynomial Functions**
    #. **Radial Basis Functions**

    These interpolation methods are described in [Ameli-2020]_. 

.. |image01| replace:: $\mathbf{A}$
.. |image02| replace:: $\mathrm{trace}(\mathbf{A}^{-1})$
.. |image03| replace:: $\mathbf{B}$
.. |image04| replace:: $t\in [t_0,t_1]$
.. |image05| replace:: $t\mapsto\mathbf{A}+t\mathbf{B}$
.. |image06| replace:: $$t\mapsto\mathrm{trace}\left((\mathbf{A}+t\mathbf{B})^{-1}\right)$$

===========================
Some Potential Applications
===========================



============================
Benefits of TraceInv Package
============================

The greatest benefit that TraceInv offers is its ability to interpolate the trace of the inverse of *affine matrix functions*, which is implemented in :mod:`TraceInv.InterpolateTraceOfInverse` module based.

Mention Examples

=====================
Other Useful Packages
=====================

TraceInv does not replace many sophisticated computational packages in linear algebra and machine learning. Rather, some if its features were implemented on top of other packages. 

==========
References
==========

.. [Ameli-2020] Ameli, S., and Shadden. S. C. (2020). Interpolating the Trace of the Inverse of Matrix **A** + t **B**. `arXiv:2009.07385 <https://arxiv.org/abs/2009.07385>`__ [math.NA]
