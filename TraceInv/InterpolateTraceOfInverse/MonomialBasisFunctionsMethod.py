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
    """
    Computes the trace of inverse of an invertible matrix :math:`\\mathbf{A} + t \\mathbf{B}` using

    .. math::

        \\frac{1}{(\\tau(t))^{p+1}} = \\frac{1}{(\\tau_0)^{p+1}} + \sum_{i=1}^{p+1} w_i t^i,

    where

    .. math::

        \\tau(t) = \\frac{\mathrm{trace}\left( (\mathbf{A} + t \mathbf{B})^{-1} \\right)}{\mathrm{trace \left( \mathbf{B}^{-1} \\right)}}

    and :math:`\\tau_0 = \\tau(0)` and :math:`w_{p+1} = 1`. 
    To find the weight coefficient :math:`w_1`, the trace is computed at the given 
    interpolant point :math:`t_1` (see ``InterpolantPoint`` argument).

    When :math:`p = 1`, meaning that there is only one interpolant point :math:`t_1` with the function value 
    :math:`\\tau_1 = \\tau(t_1)`, the weight coefficient :math:`w_1` can be solved easily. In this case, 
    the interpolation function becomes

    .. math::


        \\frac{1}{(\\tau(t))^2} \\approx  \\frac{1}{\\tau_0^2} + t^2 + \left( \\frac{1}{\\tau_1^2} - \\frac{1}{\\tau_0^2} - t_1^2 \\right) \\frac{t}{t_1}.
    
    .. note::

        This class accepts only *one* interpolant point (:math:`p = 1`). That is, the argument
        ``InterpolantPoint`` should be only one number or a list of the length 1.

    **Class Inheritance:**

    .. inheritance-diagram:: TraceInv.InterpolateTraceOfInverse.MonomialBasisFunctionsMethod
        :parts: 1

    :param A: Invertible matrix, can be either dense or sparse matrix.
    :type A: numpy.ndarray

    :param B: Invertible matrix, can be either dense or sparse matrix.
    :type B: numpy.ndarray

    :param ComputeOptions: A dictionary of input arguments for :mod:`TraceInv.ComputeTraceOfInverse.ComputeTraceOfInverse` module.
    :type ComputeOptions: dict

    :param Verbose: If ``True``, prints some information on the computation process. Default is ``False``.
    :type Verbose: bool

    :example:

    This class can be invoked from :class:`TraceInv.InterpolateTraceOfInverse.InterpolateTraceOfInverse` module 
    using ``InterpolationMethod='MBF'`` argument.

    .. code-block:: python

        >>> from TraceInv import GenerateMatrix
        >>> from TraceInv import InterpolateTraceOfInverse
        
        >>> # Generate a symmetric positive-definite matrix of the shape (20**2,20**2)
        >>> A = GenerateMatrix(NumPoints=20)
        
        >>> # Create an object that interpolates trace of inverse of A+tI (I is identity matrix)
        >>> TI = InterpolateTraceOfInverse(A,InterpolatingMethod='MBF')
        
        >>> # Interpolate A+tI at some input point t
        >>> t = 4e-1
        >>> trace = TI.Interpolate(t)

    .. seealso::
        
        This class can only accept one interpolant point. A better method is ``'RMBF'`` which 
        accepts arbitrary number of interpolant points. It is recommended to use the
        ``'RMBF'`` (see :class:`TraceInv.InterpolateTraceOfInverse.RootMonomialBasisFunctionsMethod`) instead of this class.
    """

    # ----
    # Init
    # ----

    def __init__(self,A,B=None,InterpolantPoint=None,ComputeOptions={},Verbose=False):
        """
        Initializes the base class and attributes, namely, the trace at the interpolant point.
        """

        # Base class constructor
        super(MonomialBasisFunctionsMethod,self).__init__(A,B=B,ComputeOptions={},Verbose=Verbose)

        # Compute self.trace_Ainv, self.trace_Binv, and self.tau0
        self.ComputeTraceInvOfInputMatrices()

        # t1
        if InterpolantPoint is None:
            self.t1 = 1.0 / self.tau0
        else:
            # Check number of interpolant points
            if not isinstance(InterpolantPoint,Number):
                raise TypeError("InterpolantPoints for the 'MBF' method should be a single number, not an array of list of numbers.")

            self.t1 = InterpolantPoint

        # Initialize interpolator
        self.tau1 = None
        self.InitializeInterpolator() 

        # Attributes
        self.t_i = numpy.array([self.t1])
        self.trace_i = self.T1
        self.p = self.t_i.size

    # -----------------------
    # Initialize Interpolator
    # -----------------------

    def InitializeInterpolator(self):
        """
        Computes the trace at the interpolant point. This function is used internally.
        """
       
        if self.Verbose:
            print('Initialize interpolator ...')
        
        An = self.A + self.t1*self.B
        self.T1 = ComputeTraceOfInverse(An,**self.ComputeOptions)
        self.tau1 = self.T1 / self.trace_Binv

        if self.Verbose:
            print('Done.')
        
    # -----------
    # Interpolate
    # -----------

    def Interpolate(self,t):
        """
        Interpolates :math:`\mathrm{trace} \left( (\mathbf{A} + t \mathbf{B})^{-1} \\right)` at :math:`t`.

        This is the main interface function of this module and it is used after the interpolation
        object is initialized.

        :param t: The inquiry point(s).
        :type t: float, list, or numpy.array

        :return: The interpolated value of the trace.
        :rtype: float or numpy.array
        """

        # Interpolate
        tau = 1.0 / (numpy.sqrt(t**2 + ((1.0/self.tau1)**2 - (1.0/self.tau0)**2 - self.t1**2)*(t/self.t1) + (1.0/self.tau0)**2))
        Trace = tau*self.trace_Binv

        return Trace
