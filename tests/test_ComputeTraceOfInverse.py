#! /usr/bin/env python

# =======
# Imports
# =======

import sys
import scipy
from scipy import sparse
from TraceInv import GenerateMatrix
from TraceInv import ComputeTraceOfInverse

# ==============================================
# Compute Trace Of Inverse With Multiple Methods
# ==============================================

def ComputeTraceOfInverseWithMultipleMethods(K):
    """
    Computes the trace of the inverse of ``K`` with multiple method.
    """

    # Use Cholesky method with direct inverse
    Trace1 = ComputeTraceOfInverse(K,'cholesky',UseInverseMatrix=False)

    # Use Cholesky method without direct inverse
    if not scipy.sparse.isspmatrix(K):
        Trace2 = ComputeTraceOfInverse(K,'cholesky',UseInverseMatrix=True)
    else:
        # Do not use Cholesky with inverse method if K is sparse.
        Trace2 = None

    # Use Hutchinson method
    Trace3 = ComputeTraceOfInverse(K,'hutchinson',NumIterations=100)

    # Use Stochastic Lanczos Quadrature method, with tridiagonalization
    Trace4 = ComputeTraceOfInverse(K,'SLQ',NumIterations=100,
            LanczosDegree=100,UseLanczosTridiagonalization=False)

    # Use Stochastic Lanczos Quadrature method, with bidiagonalization
    Trace5 = ComputeTraceOfInverse(K,'SLQ',NumIterations=100,
            LanczosDegree=100,UseLanczosTridiagonalization=True)

    # Print results
    print('')
    print('-----------------------------------------------')
    print('Method       Options                   TraceInv')
    print('----------   -----------------------   --------')
    print('Cholesky     using inverse             %0.3f'%Trace1)
    if Trace2 is not None:
        print('Cholesky     without using inverse     %0.3f'%Trace2)
    else:
        print('Cholesky     without using inverse     N/A')
    print('Hutchinson   N/A                       %0.3f'%Trace3)
    print('SLQ          with tri-diagonalization  %0.3f'%Trace4)
    print('SLQ          with bi-diagonalization   %0.3f'%Trace5)
    print('-----------------------------------------------')
    print('')

# =============================
# Test Compute Trace Of Inverse
# =============================

def test_ComputeTraceOfInverse():
    """
    Testing ``ComputeTraceOfInverse`` sub-package
    """

    # Compute trace of inverse of K using dense matrix
    print('Using dense matrix')
    K1 = GenerateMatrix(NumPoints=20,UseSparse=False)
    ComputeTraceOfInverseWithMultipleMethods(K1)

    # Compute trace of inverse of K using sparse matrix
    print('Using sparse matrix')
    K2 = GenerateMatrix(NumPoints=20,UseSparse=True,RunInParallel=True)
    ComputeTraceOfInverseWithMultipleMethods(K2)

    # Test
    import pickle
    with open('/home/sia/Downloads/K.pickle','wb') as h:
        pickle.dump(K2,h)

# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_ComputeTraceOfInverse())
