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

    def __init__(self,A,B=None,InterpolantPoint=None,ComputeOptions={}):

        # Base class constructor
        super(MonomialBasisFunctionsMethod,self).__init__(A,B,ComputeOptions={})

        # Compute self.trace_Ainv, self.trace_Binv, and self.tau0
        self.ComputeTraceInvOfInputMatrices()

        # eta1
        if InterpolantPoint is None:
            self.eta1 = 1.0 / self.tau0
        else:
            # Check number of interpolant points
            if not isinstance(InterpolantPoint,Number):
                raise TypeError("InterpolantPoints for the 'MBF' method should be a single number, not an array of list of numbers.")

            self.eta1 = InterpolantPoint

        # Initialize interpolator
        self.tau1 = None
        self.InitializeInterpolator() 

    # -----------------------
    # Initialize Interpolator
    # -----------------------

    def InitializeInterpolator(self):
        """
        """
        
        print('Initialize interpolator ...')
        
        An = self.A + self.eta1*self.B
        T1 = ComputeTraceOfInverse(An,**self.ComputeOptions)
        self.tau1 = T1 / self.trace_Binv

        print('Done.')
        
    # -----------
    # Interpolate
    # -----------

    def Interpolate(self,t):
        """
        Interpolates the trace of inverse of ``A + t*B``.

        :param t: A real variable to form the linear matrix function ``A + tB``.
        :type t: float

        :return: The interpolated value of the trace of inverse of ``A + tB``.
        :rtype: float
        """

        tau = 1.0 / (numpy.sqrt(t**2 + ((1.0/self.tau1)**2 - (1.0/self.tau0)**2 - self.eta1**2)*(t/self.eta1) + (1.0/self.tau0)**2))
        trace = tau*self.trace_Binv

        return trace
