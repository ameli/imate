=======================
Compute Log-Determinant
=======================

The sub-package ``TraceInv.ComputeLogDeterminnat`` computes the log-determinant of a positive-definite matrix. The module makes use of the module ``TraceInv.LinearAlgebra``.

-----
Usage
-----

.. code-block:: python

    >>> # Import packages
    >>> from TraceInv import ComputeLogDeterminant
    >>> from TraceInv import GenerateMatrix

    >>> # Generate a sample matrix
    >>> A = GenerateMatrix(NumPoints=20)

    >>> # Compute log-determinant with Cholesky method
    >>> LogDet_1 = ComputeLogDeterminant(A)

    >>> # Compute log-determinant with stochastic Lanczos quadrature method
    >>> LogDet_1 = ComputeLogDeterminant(A,ComputeMethod='SLQ',NumIterations=20,LanczosDegree=20)

    >>> # Compute log-determinant with stochastic Lanczos quadrature method with Golub-Khan bi-diagonalization
    >>> LogDet_1 = ComputeLogDeterminant(A,ComputeMethod='SLQ',NumIterations=20, 
    ...                 LanczosDegree=20,Tridiagonalization=False)
