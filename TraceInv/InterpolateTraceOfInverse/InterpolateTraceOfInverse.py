# =======
# Imports
# =======

import sys
import numpy
import scipy
from scipy import sparse

from numbers import Number

# Classes, Files
from .PlotSettings import *
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

    def __init__(self,A,InterpolantPoints=None,Method='RMBF',**Options):
        """
        Initializes the object depending on the method.

        Methods
            * ``'EIG'`` : Uses eigenvalues of ``A``.
            * ``'MBF'`` : Uses monomial basis functions.
            * ``'RMBF'``: Uses root monomial basis functions.
            * ``'RBF'`` : Uses radial basis functions.
            * ``'RPF'`` : Uses ratioanl polynomial functions.

        :param Method: determines one of the methods of interpolation.
        :type Method: string

        :param Options: Optios for each of the methods.
        :type Options: ``**kwargs``
        """

        # Define an interpolation object depending on the given method
        if Method == 'EIG':
            # Eigenvalues method
            self.Interpolator = EigenvaluesMethod(A,**Options)

        elif Method == 'MBF':
            # Monomial Basis Functions method
            self.Interpolator = MonomialBasisFunctionsMethod(A,InterpolantPoints,**Options)

        elif Method == 'RMBF':
            # Root Monomial Basis Functions method
            self.Interpolator = RootMonomialBasisFunctionsMethod(A,InterpolantPoints,**Options)

        elif Method == 'RBF':
            # Radial Basis Functions method
            self.Interpolator = RadialBasisFunctionsMethod(A,InterpolantPoints,**Options)

        elif Method == 'RPF':
            # Rational Polynomial Functions method
            self.Interpolator = RationalPolynomialFunctionsMethod(A,InterpolantPoints,**Options)

        else:
            raise ValueError('Method is invalid.')

    # -------
    # Compute
    # -------

    def Compute(self,t,Method='cholesky',**Options):
        """
        """
        
        if isinstance(t,Number):
            # Single number
            T =  self.Interpolator.Compute(t,Method,**Options)

        else:
            # An array of points
            T = numpy.array((len(t)),dtype=float)
            for i in range(len(t)):
                T[i] =  self.Interpolator.Compute(t[i],Method,**Options)

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
            T = numpy.array((len(t)),dtype=float)
            for i in range(len(t)):
                T[i] =  self.Interpolator.Interpolate(t[i])

        return T
