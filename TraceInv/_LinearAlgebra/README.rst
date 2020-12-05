==============
Linear Algebra
==============

This sub-package provides the following modules:

* ``LinearSolver.py``: Solves linear system for both sparse and dense matrices.
* ``MatrixReduction.py``: The Lanczos tri-diagonalization and Golub-Kahn-Lanczos bi-diagonalization is implemented in this module.
* ``SparseCholesky`` computes the Cholesky decomposition of a sparse matrix. If the Suite Sparse package is installed, you do not need to use this module.

These modules are used in ``TraceInv.ComputeTraceOfInverse`` and ``TraceInv.ComputeLogDeterminant`` sub-packages.
