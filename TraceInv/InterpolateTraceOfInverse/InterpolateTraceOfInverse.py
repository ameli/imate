# =======
# Imports
# =======

import sys
import numpy
import scipy
from scipy import sparse

from numbers import Number

# Classes, Files
from .ExactMethod import ExactMethod
from .EigenvaluesMethod import EigenvaluesMethod
from .MonomialBasisFunctionsMethod import MonomialBasisFunctionsMethod
from .RootMonomialBasisFunctionsMethod import RootMonomialBasisFunctionsMethod
from .RadialBasisFunctionsMethod import RadialBasisFunctionsMethod
from .RationalPolynomialFunctionsMethod import RationalPolynomialFunctionsMethod

# ======================
# Trace Estimation Class
# ======================

class InterpolateTraceOfInverse(object):
    """
    A wrapper class that incorporates all interpolator classes of the package.
    """

    # ----
    # Init
    # ----

    def __init__(self,A,InterpolantPoints=None,InterpolationMethod='RMBF',**Options):
        """
        Initializes the object depending on the method.

        --------------------  -----------------------------
        Interpolation method  Description
        ====================  =============================
         ``'EXT'``            Exact, no interpolation
         ``'EIG'``            Eigenvalues of ``A``
         ``'MBF'``            Monomial basis functions
         ``'RMBF'`            Root monomial basis functions
         ``'RBF'``            Radial basis functions
         ``'RPF'``            Ratioanl polynomial functions
        --------------------  -----------------------------

        :param Method: determines one of the methods of interpolation.
        :type Method: string

        :param Options: Optios for each of the methods.
        :type Options: ``**kwargs``
        """

        # Attributes
        self.n = A.shape[0]
        if InterpolantPoints is not None:
            self.p = len(InterpolantPoints)
        else:
            self.p = 0

        # Define an interpolation object depending on the given method
        if InterpolationMethod == 'EXT':
            # Exact computation, not interpolation
            self.Interpolator = ExactMethod(A,**Options)

        elif InterpolationMethod == 'EIG':
            # Eigenvalues method
            self.Interpolator = EigenvaluesMethod(A,**Options)

        elif InterpolationMethod == 'MBF':
            # Monomial Basis Functions method
            self.Interpolator = MonomialBasisFunctionsMethod(A,InterpolantPoints,**Options)

        elif InterpolationMethod == 'RMBF':
            # Root Monomial Basis Functions method
            self.Interpolator = RootMonomialBasisFunctionsMethod(A,InterpolantPoints,**Options)

        elif InterpolationMethod == 'RBF':
            # Radial Basis Functions method
            self.Interpolator = RadialBasisFunctionsMethod(A,InterpolantPoints,**Options)

        elif InterpolationMethod == 'RPF':
            # Rational Polynomial Functions method
            self.Interpolator = RationalPolynomialFunctionsMethod(A,InterpolantPoints,**Options)

        else:
            raise ValueError("'InterpolationMethod' is invalid. Select one of 'EXT', 'EIG', 'MBF', 'RMBF', 'RBF', or 'RPF'.")

    # -------
    # Compute
    # -------

    def Compute(self,t,ComputingMethod='cholesky',**Options):
        """
        """
        
        if isinstance(t,Number):
            # Single number
            T =  self.Interpolator.Compute(t,ComputingMethod,**Options)

        else:
            # An array of points
            T = numpy.empty((len(t),),dtype=float)
            for i in range(len(t)):
                T[i] =  self.Interpolator.Compute(t[i],ComputingMethod,**Options)

        return T

    # -----------
    # Interpolate
    # -----------

    def Interpolate(self,t):
        """
        Interpolates the trace of inverse of ``A + t*I``.

        :param t: A real variable to form the linear matrix function ``A + tI``.
        :type t: float or array

        :return: The interpolated value of the trace of inverse of ``A + tI``.
        :rtype: float or array
        """
        
        if isinstance(t,Number):
            # Single number
            T =  self.Interpolator.Interpolate(t)

        else:
            # An array of points
            T = numpy.empty((len(t),),dtype=float)
            for i in range(len(t)):
                T[i] =  self.Interpolator.Interpolate(t[i])

        return T

    # -----------
    # Lower Bound
    # -----------

    def LowerBound(self,t):
        """
        """

        if isinstance(t,Number):
            # Single number
            T_lb =  self.Interpolator.LowerBound(t)

        else:
            # An array of points
            T_lb = numpy.empty((len(t),),dtype=float)
            for i in range(len(t)):
                T_lb[i] =  self.Interpolator.LowerBound(t[i])

        return T_lb

    # -----------
    # Upper Bound
    # -----------

    def UpperBound(self,t):
        """
        """

        if isinstance(t,Number):
            # Single number
            T_ub =  self.Interpolator.UpperBound(t)

        else:
            # An array of points
            T_ub = numpy.empty((len(t),),dtype=float)
            for i in range(len(t)):
                T_ub[i] =  self.Interpolator.UpperBound(t[i])

        return T_ub
