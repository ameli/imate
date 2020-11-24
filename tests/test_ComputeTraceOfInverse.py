#! /usr/bin/env python

# =======
# Imports
# =======

import sys
import scipy
from scipy import sparse
from TraceInv import GenerateMatrix
from TraceInv import ComputeTraceOfInverse
import time
import numpy

# ==============================================
# Compute Trace Of Inverse With Multiple Methods
# ==============================================

def ComputeTraceOfInverseWithMultipleMethods(K):
    """
    Computes the trace of the inverse of matrix ``K`` with multiple method.

    :param K: Invertible matrix.
    :type K: numpy.ndarray
    """

    # Use Cholesky method with direct inverse
    Time10 = time.time()
    Trace1 = ComputeTraceOfInverse(K,'cholesky',UseInverseMatrix=False)
    Time11 = time.time()

    # Use Cholesky method without direct inverse
    if not scipy.sparse.isspmatrix(K):
        Time20 = time.time()
        Trace2 = ComputeTraceOfInverse(K,'cholesky',UseInverseMatrix=True)
        Time21 = time.time()
    else:
        # Do not use Cholesky with inverse method if K is sparse.
        Trace2 = None
        Time20 = 0
        Time21 = 0

    # Use Hutchinson method
    Time30 = time.time()
    Trace3 = ComputeTraceOfInverse(K,'hutchinson',NumIterations=30)
    Time31 = time.time()

    # Use Stochastic Lanczos Quadrature method, with tridiagonalization
    Time40 = time.time()
    Trace4 = ComputeTraceOfInverse(K,'SLQ',NumIterations=30,
            LanczosDegree=30,UseLanczosTridiagonalization=True)
    Time41 = time.time()

    # Use Stochastic Lanczos Quadrature method, with bidiagonalization
    Time50 = time.time()
    Trace5 = ComputeTraceOfInverse(K,'SLQ',NumIterations=30,
            LanczosDegree=30,UseLanczosTridiagonalization=False)
    Time51 = time.time()

    # Elapsed times
    ElapsedTime1 = Time11 - Time10
    ElapsedTime2 = Time21 - Time20
    ElapsedTime3 = Time31 - Time30
    ElapsedTime4 = Time41 - Time40
    ElapsedTime5 = Time51 - Time50

    # Error
    Error1 = 0.0
    if Trace2 is not None:
        Error2 = 100.0* numpy.abs(Trace2 - Trace1) / Trace1
    Error3 = 100.0 * numpy.abs(Trace3 - Trace1) / Trace1
    Error4 = 100.0 * numpy.abs(Trace4 - Trace1) / Trace1
    Error5 = 100.0 * numpy.abs(Trace5 - Trace1) / Trace1

    # Print results
    print('')
    print('-----------------------------------------------------------')
    print('Method      Options                   TraceInv  Error  Time')
    print('----------  ------------------------  --------  -----  ----')
    print('Cholesky    without using inverse     %0.3f  %0.2f%%  %0.2f'%(Trace1,Error1,ElapsedTime1))
    if Trace2 is not None:
        print('Cholesky    using inverse             %0.3f  %0.2f%%  %0.2f'%(Trace2,Error2,ElapsedTime2))
    else:
        print('Cholesky    using inverse             N/A   N/A        N/A')
    print('Hutchinson  N/A                       %0.3f  %0.2f%%  %0.2f'%(Trace3,Error3,ElapsedTime3))
    print('SLQ         with tri-diagonalization  %0.3f  %0.2f%%  %0.2f'%(Trace4,Error4,ElapsedTime4))
    print('SLQ         with bi-diagonalization   %0.3f  %0.2f%%  %0.2f'%(Trace5,Error5,ElapsedTime5))
    print('-----------------------------------------------------------')
    print('')

# =============================
# Test Compute Trace Of Inverse
# =============================

def test_ComputeTraceOfInverse():
    """
    Test for :mod:`TraceInv.ComputeTraceOfInverse` sub-package.
    """

    # Compute trace of inverse of K using dense matrix
    print('Using dense matrix')
    K1 = GenerateMatrix(NumPoints=20,UseSparse=False)
    ComputeTraceOfInverseWithMultipleMethods(K1)

    # Compute trace of inverse of K using sparse matrix
    print('Using sparse matrix')
    K2 = GenerateMatrix(NumPoints=20,UseSparse=True,RunInParallel=True)
    # K2 = GenerateMatrix(NumPoints=80,KernelThreshold=0.05,DecorrelationScale=0.02,UseSparse=True,RunInParallel=True)
    ComputeTraceOfInverseWithMultipleMethods(K2)

# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_ComputeTraceOfInverse())
