TraceInv
========

This directory defines these sub-modules:

* ``InterpolateTraceOfInverse``: The main module that contains the class ``InterpolateTraceOfInverse``. This class *interpolates* the trace of inverse of a matrix of the form :math:`\mathbf{A} + t \mathbf{B}` where the matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` are fixed and the real number :math:`t` varies.
* ``ComputeTraceOfInverse``: A module that contains the class ``ComputeTraceofInverse``. This class computes the trace of the inverse of a *generic* invertible matrix.  This computation direct, which is not interpolated.
* ``GenerateMatrix``: defines the class ``GenerateMatrix`` which generates a positive-definite correlation matrix. This matrix is used to test the package.
