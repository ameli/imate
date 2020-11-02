# =======
# Imports
# =======

from __future__ import print_function
import numpy
from .InterpolantBaseClass import InterpolantBaseClass

# ============
# Exact Method
# ============

class ExactMethod(InterpolantBaseClass):

    # ----
    # Init
    # ----

    def __init__(self,A,**Options):

        # Base class constructor
        super(ExactMethod,self).__init__(A)

    # -----------
    # Interpolate
    # -----------

    def Interpolate(self,t,**Options):
        """
        This function does not interpolate, rather exactly computes the trace of inverse of ``A + t*I``.

        :return: The exact value of the trace of inverse of ``A + tI``.
        :rtype: float
        """
       
        # Do not interpolate, instead compute the exact value
        T = self.Compute(t,**Options)

        return T
