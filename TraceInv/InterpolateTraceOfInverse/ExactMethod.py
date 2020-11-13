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
    """
    .. inheritance-diagram:: TraceInv.InterpolateTraceOfInverse.ExactMethod
        :parts: 1

    Computes the trace of inverse of an invertible matrix :math:`\\mathbf{A} + t \\mathbf{B}` using 
    exact method (no interpolation is performed).
    This class does not accept interpolant points as the result is not interpolated.

    :example:

    This class can be invoked from the :class:`TraceInv.InterpolateTraceOfInverse.InterpolateTraceOfInverse` module 
    using ``InterpolationMethod='EXT'`` argument.

    .. code-block:: python

        from TraceInv import GenerateMatrix
        from TraceInv import InterpolateTraceOfInverse
        
        # Generate a symmetric positive-definite matrix of the shape (20**2,20**2)
        A = GenerateMatrix(NumPoints=20)
        
        # Create an object that interpolats trace of inverse of A+tI (I is identity matrix)
        TI = InterpolateTraceOfInverse(A,InterpolatingMethod='EXT')
        
        # Interpolate A+tI at some input point t
        t = 4e-1
        trace = TI.Interpolate(t)

    .. seealso::

        The result of the ``EXT`` method is identical with the eigenvalue method ``EIG``, 
        which is given by :class:`TraceInv.InterpolateTraceOfInverse.EigenvaluesMethod`.
    """

    # ----
    # Init
    # ----

    def __init__(self,A,B=None,ComputeOptions={}):

        # Base class constructor
        super(ExactMethod,self).__init__(A,B,ComputeOptions=ComputeOptions)

    # -----------
    # Interpolate
    # -----------

    def Interpolate(self,t):
        """
        This function does not interpolate, rather exactly computes the trace of inverse of ``A + t*B``.

        :return: The exact value of the trace of inverse of ``A + tB``.
        :rtype: float
        """
 
        # Do not interpolate, instead compute the exact value
        T = self.Compute(t)

        return T
