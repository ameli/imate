# =======
# Imports
# =======

from __future__ import print_function
from ..ComputeTraceOfInverse import ComputeTraceOfInverse

import numpy
import scipy
from scipy import sparse

# ======================
# Interpolant Base Class
# ======================

class InterpolantBaseClass(object):
    """
    This is the base class for the following classes:

    * ``MonomialBasisFunctionsMethod``
    * ``RootMonomialBasisFunctionsMethod``
    * ``RadialBasisFunctionsMethod``
    * ``RationalPolynomialFunctionsMethod``
    """

    # ----
    # Init
    # ----

    def __init__(self,A,InterpolantPoints=None):
        """
        * Initializes te interpolant points and the input matrix.
        * Scale the interpolant points if needed.
        * Comutes the trace of inverse at the interpolant points.
        """

        # Member data
        self.A = A
        self.n = self.A.shape[0]

        # Determine to use sparse
        self.UseSparse = False
        if scipy.sparse.isspmatrix(A):
            self.UseSparse = True

        # Compute trace at interpolant points
        self.T0 = None

        if InterpolantPoints is not None:

            # Compute trace at interpolant points
            self.T0 = ComputeTraceOfInverse(self.A)

            # Compute trace at interpolant points
            self.eta_i = numpy.array(InterpolantPoints)
            self.p = self.eta_i.size
            self.trace_eta_i = self.ComputeTraceAtInterpolantPoints()

            # Scale interpolant points
            self.Scale_eta = None
            self.ScaleInterpolantPoints()

    # ------------------------
    # Scale Interpolant Points
    # ------------------------

    def ScaleInterpolantPoints(self):
        """
        If the largest interpolant point is greater than 1, this function scales their range to max at 1.
        """

        # Scale eta, if some of eta_i are greater than 1
        self.Scale_eta = 1.0
        if self.eta_i.size > 0:
            if numpy.max(self.eta_i) > 1.0:
                self.Scale_eta = numpy.max(self.eta_i)

    # -------
    # Compute
    # -------

    def Compute(self,t,Method='cholesky',**Options):
        """
        """
        
        if self.UseSparse:
            I = scipy.sparse.eye(n,format='csc')
        else:
            I = numpy.eye(self.n)

        An = self.A + t*I
        T = ComputeTraceOfInverse(An,Method,**Options)

        return T
        
    # -----------------------------------
    # Compute Trace At Interpolant Points
    # -----------------------------------

    def ComputeTraceAtInterpolantPoints(self):

        print('Evaluate function at interpolant points ...',end='')

        # Interpolation Method

        # Compute trace at interpolant points
        trace_eta_i = numpy.zeros(self.p)
        for i in range(self.p):
            trace_eta_i[i] = self.Compute(self.eta_i[i])

        print(' Done.')

        return trace_eta_i

    # -----------
    # Lower Bound
    # -----------

    def LowerBound(self,t):
        """
        """

        T_lb = self.n/(1.0+t)

        return T_lb

    # -----------
    # Upper Bound
    # -----------

    def UpperBound(self,t):
        """
        """

        # Compute trace at interpolant points
        if self.T0 is None:
            self.T0 = ComputeTraceOfInverse(self.A)

        T_ub = 1.0/(1.0/self.T0 + t/self.n)

        return T_ub
