#! /usr/bin/env python

# =======
# Imports
# =======

import sys
import scipy
from scipy import sparse
from TraceInv import GenerateMatrix
from TraceInv import ComputeLogDeterminant
import time
import numpy

# =============================================
# Compute Log Determinant With Multiple Methods
# =============================================

def ComputeLogDeterminantWithMultipleMethods(K):
    """
    Computes the log-determinant of matrix ``K`` with multiple method.

    :param K: Invertible matrix.
    :type K: numpy.ndarray
    """

    # Use Cholesky method
    Time10 = time.time()
    Trace1 = ComputeLogDeterminant(K,ComputeMethod='cholesky')
    Time11 = time.time()

    # Use Stochastic Lanczos Quadrature method, with tri-diagonalization
    Time20 = time.time()
    Trace2 = ComputeLogDeterminant(K,ComputeMethod='SLQ',NumIterations=50,LanczosDegree=30,UseLanczosTridiagonalization=True)
    Time21 = time.time()

    # Use Stochastic Lanczos Quadrature method, with bi-diagonalization
    Time30 = time.time()
    Trace3 = ComputeLogDeterminant(K,ComputeMethod='SLQ',NumIterations=50,LanczosDegree=30,UseLanczosTridiagonalization=False)
    Time31 = time.time()

    # Elapsed times
    ElapsedTime1 = Time11 - Time10
    ElapsedTime2 = Time21 - Time20
    ElapsedTime3 = Time31 - Time30

    # Error
    Error1 = 0.0
    Error2 = 100.0 * numpy.abs(Trace2 - Trace1) / numpy.abs(Trace1)
    Error3 = 100.0 * numpy.abs(Trace3 - Trace1) / numpy.abs(Trace1)

    # Print results
    print('')
    print('---------------------------------------------------------')
    print('Method      Options                   LogDet  Error  Time')
    print('----------  ------------------------  ------  -----  ----')
    print('Cholesky    N/A                       %0.3f  %0.2f%%  %0.2f'%(Trace1,Error1,ElapsedTime1))
    print('SLQ         with tri-diagonalization  %0.3f  %0.2f%%  %0.2f'%(Trace2,Error2,ElapsedTime2))
    print('SLQ         with bi-diagonalization   %0.3f  %0.2f%%  %0.2f'%(Trace3,Error3,ElapsedTime3))
    print('---------------------------------------------------------')
    print('')

# ============================
# Test Compute Log Determinant
# ============================

def test_ComputeLogDeterminant():
    """
    Test for :mod:`TraceInv.ComputeLogDeterminant` sub-package.
    """

    # Compute trace of inverse of K using dense matrix
    print('Using dense matrix')
    K1 = GenerateMatrix(NumPoints=20,UseSparse=False)
    K1 = K1 + 0.5*numpy.eye(K1.shape[0])
    ComputeLogDeterminantWithMultipleMethods(K1)

    # Compute trace of inverse of K using sparse matrix
    print('Using sparse matrix')
    K2 = GenerateMatrix(NumPoints=20,UseSparse=True,RunInParallel=True)
    # K2 = GenerateMatrix(NumPoints=50,KernelThreshold=0.03,DecorrelationScale=0.03,UseSparse=True,RunInParallel=True)
    K2 = K2 + 0.35*scipy.sparse.eye(K2.shape[0])
    ComputeLogDeterminantWithMultipleMethods(K2)

# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_ComputeLogDeterminant())
