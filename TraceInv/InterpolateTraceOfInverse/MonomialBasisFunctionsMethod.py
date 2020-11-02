# =======
# Imports
# =======

from __future__ import print_function
from ..ComputeTraceOfInverse import ComputeTraceOfInverse
from .InterpolantBaseClass import InterpolantBaseClass

import numpy

# ===============================
# Monomial Basis Functions Method
# ===============================

class MonomialBasisFunctionsMethod(InterpolantBaseClass):

    # ----
    # Init
    # ----

    def __init__(self,A,t1=None):

        # Base class constructor
        super(MonomialBasisFunctionsMethod,self).__init__(A)

        # Compute trace at interpolant points
        self.T0 = ComputeTraceOfInverse(self.A)

        # t1
        if t1 is None:
            self.t1 = self.n / self.T0
        else:
            self.t1 = t1

        # Initialize interpolator
        self.T1 = None
        self.InitializeInterpolator() 

    # -----------------------
    # Initialize Interpolator
    # -----------------------

    def InitializeInterpolator(self):
        """
        """
        
        print('Initialize interpolator ...')
        
        # Interpolant points for the auxilliary estimation method
        if self.UseSparse:
            I = scipy.sparse.eye(n,format='csc')
        else:
            I = numpy.eye(self.n)

        An = self.A + self.t1*I
        self.T1 = ComputeTraceOfInverse(An)

        print('Done.')
        
    # -----------
    # Interpolate
    # -----------

    def Interpolate(self,t):
        """
        Interpolates the trace of inverse of ``A + t*I``.

        :param t: A real variable to form the linear matrix function ``A + tI``.
        :type t: float

        :return: The interpolated value of the trace of inverse of ``A + tI``.
        :rtype: float
        """

        T = 1.0 / (numpy.sqrt((t/self.n)**2 + ((1.0/self.T1)**2 - (1.0/self.T0)**2 - (self.t1/self.n)**2)*(t/self.t1) + (1/self.T0)**2))

        return T
