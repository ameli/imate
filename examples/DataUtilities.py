# =======
# Imports
# =======

import numpy
from numpy import linalg

# ========================
# Generate Basis Functions
# ========================

def GenerateBasisFunctions(n,m):

    numpy.random.seed(31)

    if n > m:
        u = numpy.random.randn(n)
        U = numpy.eye(n) - 2.0 * numpy.outer(u,u) / numpy.linalg.norm(u)**2
        U = U[:,:m]
    else:
        u = numpy.random.randn(m)
        U = numpy.eye(m) - 2.0 * numpy.outer(u,u) / numpy.linalg.norm(u)**2
        U = U[:n,:]

    v = numpy.random.randn(m)
    V = numpy.eye(m) - 2.0 * numpy.outer(v,v.T) / numpy.linalg.norm(v)**2
    
    # sigma = numpy.exp(-20.0*(numpy.arange(m)/float(m))**(0.5))
    sigma = numpy.exp(-40.0*(numpy.arange(m)/float(m))**(0.75))  # good for n,m = 1000,500
    # sigma = numpy.exp(-20.0*(numpy.arange(m)/float(m))**(0.5)) * numpy.sqrt(n/1000.0) * 1e2
    # sigma = numpy.exp(-10.0*(numpy.arange(m)/float(m))**(0.2)) * numpy.sqrt(n/1000.0) * 1e2
    # sigma = numpy.sqrt(sigma)

    Sigma = numpy.diag(sigma)

    X = numpy.matmul(U,numpy.matmul(Sigma,V.T))

    return X

# ===================
# Generate Noisy Data
# ===================

def GenerateNoisyData(X,NoiseLevel=4e-1):
    """
    """

    # Size of basis function
    n,m = X.shape
    
    # beta = numpy.random.randn(m)
    beta = numpy.random.randn(m) / numpy.sqrt(n/1000.0)

    epsilon = NoiseLevel * numpy.random.randn(n)

    # Data
    z = numpy.dot(X,beta) + epsilon

    return z

# ===============
# Generate Matrix
# ===============

def GenerateMatrix(n=1000,m=500,Shift=1e-3):
    """
    """

    # Generate basis functions for linear regression
    X = GenerateBasisFunctions(n,m)

    # Create Grammian matrix K0 from X
    if n > m:
        K0 = X.T.dot(X)

    else:
        K0 = X.dot(X.T)

    # Add shift to K0
    K = K0 + Shift * numpy.eye(K0.shape[0],K0.shape[1])

    # Condition numbers
    Cond_K0 = numpy.linalg.cond(K0)
    Cond_K = numpy.linalg.cond(K)
    print('Cond K0: %0.2e, Cond K: %0.2e'%(Cond_K0,Cond_K))

    return K
