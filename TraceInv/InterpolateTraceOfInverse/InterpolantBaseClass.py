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
    This is the base class for the following derived classes:

    * :class:`TraceInv.InterpolateTraceOfInverse.ExactMethod`
    * :class:`TraceInv.InterpolateTraceOfInverse.EigenvaluesMethod`
    * :class:`TraceInv.InterpolateTraceOfInverse.MonomialBasisFunctionsMethod`
    * :class:`TraceInv.InterpolateTraceOfInverse.RootMonomialBasisFunctionsMethod`
    * :class:`TraceInv.InterpolateTraceOfInverse.RadialBasisFunctionsMethod`
    * :class:`TraceInv.InterpolateTraceOfInverse.RationalPolynomialFunctionsMethod`

    :param A: A positive-definite matrix. Matrix can be dense or sparse.
    :type A: numpy.ndarray or scipy.sparse.csc_matrix

    :param B: A positive-definite matrix. Matrix can be dense or sparse.
        If ``None`` or not provided, it is assumed that ``B`` is an identity matrix of the shape of ``A``.
    :type B: numpy.ndarray or scipy.sparse.csc_matrix

    :param InterpolantPoints: A list or an array of points that the interpolator use to interpolate. 
        The trace of inverse is computed for the interpolant points with exact method. 
    :type InterpolantPoints: list(float) or numpy.array(float)

    :param InterpolationMethod: One of the methods ``'EXT'``, ``'EIG'``, ``'MBF'``, ``'RMBF'``, ``'RBF'``, and ``'RPF'``.
        Default is ``'RMBF'``.
    :type InterpolationMethod: string

    :param ComputeOptions: A dictionary of arguments to pass to :mod:`TraceInv.ComputeTraceOfInverse` module.
    :type ComputeOptions: dict

    :param Verbose: If ``True``, prints some information on the computation process. Default is ``False``.
    :type Verbose: bool
    """

    # ----
    # Init
    # ----

    def __init__(self,A,B=None,InterpolantPoints=None,ComputeOptions={},Verbose=False):
        """
        The initialization function does the followings:

        * Initializes the interpolant points and the input matrix.
        * Scale the interpolant points if needed.
        * Computes the trace of inverse at the interpolant points.
        """

        # Attributes
        self.Verbose = Verbose

        # Matrix A
        self.A = A
        self.n = self.A.shape[0]

        # Determine to use sparse
        self.UseSparse = False
        if scipy.sparse.isspmatrix(A):
            self.UseSparse = True

        # Matrix B
        if B is not None:
            self.B = B
            self.BIsIdentity = False
        else:
            # Assume B is identity matrix
            self.BIsIdentity = True

            if self.UseSparse:
                # Create sparse identity matrix
                self.B = scipy.sparse.eye(self.n,format='csc')
            else:
                # Create dense identity matrix
                self.B = numpy.eye(self.n)

        # Compute trace at interpolant points
        self.trace_Ainv = None
        self.trace_Binv = None
        self.tau0 = None
        self.ComputeOptions = ComputeOptions

        if InterpolantPoints is not None:

            # Compute self.trace_Ainv, self.trace_Binv, and self.tau0
            self.ComputeTraceInvOfInputMatrices()

            # Compute trace at interpolant points
            self.t_i = numpy.array(InterpolantPoints)
            self.p = self.t_i.size
            self.trace_i = self.ComputeTraceForArrayOfPoints(self.t_i)
            self.tau_i = self.trace_i / self.trace_Binv

            # Scale interpolant points
            self.Scale_t = None
            self.ScaleInterpolantPoints()

    # ----------------------------------
    # Compute TraceInv of Input Matrices
    # ----------------------------------

    def ComputeTraceInvOfInputMatrices(self):
        """
        Computes the trace of inverse of input matrices :math:`\mathbf{A}` and :math:`\mathbf{B}`, and the ratio :math:`\\tau_0`
        defined by

        .. math::

            \\tau_0 = \\frac{\mathrm{trace}( \mathbf{A}^{-1})}{\mathrm{trace}( \mathbf{B}^{-1})}.

        This function sets the following class attributes:

        * ``self.trace_Ainv``: trace of inverse of ``A``.
        * ``self.trace_Binv``: trace of inverse of ``B``.
        * ``self.tau0``: the ratio of ``self.trace_Ainv`` over ``self.trace_Binv``.
        """

        # trace of Ainv
        self.trace_Ainv = ComputeTraceOfInverse(self.A,**self.ComputeOptions)

        # trace of Binv
        if self.BIsIdentity:
            self.trace_Binv = self.n
        else:
            self.trace_Binv = ComputeTraceOfInverse(self.B,**self.ComputeOptions)

        # tau0
        self.tau0 = self.trace_Ainv / self.trace_Binv

    # ------------------------
    # Scale Interpolant Points
    # ------------------------

    def ScaleInterpolantPoints(self):
        """
        Rescales the range of interpolant points. This function is intended to be used internally.

        If the largest interpolant point in ``self.InterpolantPoints`` is greater than ``1``, 
        this function rescales their range to max at ``1``.  The rescale is necessary if the method of 
        interpolation is based on the orthogonal basis functions, which they are defined to be orthogonal 
        in the range :math:`t \in [0,1]`.
        """

        # Scale t, if some of t_i are greater than 1
        self.Scale_t = 1.0
        if self.t_i.size > 0:
            if numpy.max(self.t_i) > 1.0:
                self.Scale_t = numpy.max(self.t_i)

    # -------
    # Compute
    # -------

    def Compute(self,t):
        """
        Computes :math:`\mathrm{trace}\left( (\mathbf{A}+t \mathbf{B})^{-1} \\right)` at point :math:`t` with exact method, that is,
        no interpolation is used and the result is exact.
       
        This function is primarily used internally to compute trace of inverse at *interpolant points*. 
        This function uses :class:`TraceInv.ComputeTraceOfInverse` class with options described by
        ``self.ComputeOptions``.

        :return: Trace of inverse at input point ``t``.
        :rtype: float
        """
        
        An = self.A + t * self.B
        T = ComputeTraceOfInverse(An,**self.ComputeOptions)

        return T
        
    # ---------------------------------
    # Compute Trace For Array Of Points
    # ---------------------------------

    def ComputeTraceForArrayOfPoints(self,t_i):
        """
        Computes :math:`\mathrm{trace} \left( (\mathbf{A}+ t \mathbf{B})^{-1} \\right)` at interpolant points 
        ``self.InterpolantPoints``. At interpolant points, the trace is computed exactly 
        using :class:`TraceInv.ComputeTraceOfInverse`.

        :param t_i: An array if inquiry points.
        :type t_i: float, numpy.ndarray, or list(float)

        :return: Trace of inverse at inquiry points
        :rtype: float or numpy.ndarray
        """

        if self.Verbose:
            print('Evaluate function at interpolant points ...',end='')

        if numpy.isscalar(t_i):
            # Compute for scalar input
            trace_i = self.Compute(t_i)

        else:
            # Compute for an array
            trace_i = numpy.zeros(self.p)
            for i in range(self.p):
                trace_i[i] = self.Compute(t_i[i])

        if self.Verbose:
            print(' Done.')

        return trace_i

    # -----------
    # Lower Bound
    # -----------

    def LowerBound(self,t):
        """
        Lower bound of the function :math:`t \mapsto \mathrm{trace} \left( (\mathbf{A} + t \mathbf{B})^{-1} \\right)`.

        The lower bound is given by Remark 2.2 of [Ameli-2020]_ as

        .. math::
                
                \mathrm{trace}((\mathbf{A}+t\mathbf{B})^{-1}) \geq \\frac{n^2}{\mathrm{trace}(\mathbf{A}) + \mathrm{trace}(t \mathbf{B})}

        :param t: Inquiry points
        :type t: float or numpy.ndarray

        :return: Lower bound of the trace of inverse at inquiry points.
        :rtype: float or numpy.ndarray
        """

        # Trace of A and B
        trace_A = numpy.trace(self.A)
        trace_B = numpy.trace(self.B)

        # Lower bound of trace of A+tB
        trace_lb = (self.n**2)/(trace_A + t*trace_B)

        return trace_lb

    # -----------
    # Upper Bound
    # -----------

    def UpperBound(self,t):
        """
        Upper bound of the function :math:`t \mapsto \mathrm{trace} \left( (\mathbf{A} + t \mathbf{B})^{-1} \\right)`.

        The upper bound is given by Theorem 1 of [Ameli-2020]_ as

        .. math::
                
                \\frac{1}{\\tau(t)} \geq \\frac{1}{\\tau_0} + t

        where

        .. math::

                \\tau(t) = \\frac{\mathrm{trace}\\left( (\mathbf{A} + t \mathbf{B})^{-1}  \\right)}{\mathrm{trace}(\mathbf{B}^{-1})}

        and :math:`\\tau_0 = \\tau(0)`.

        :param t: Inquiry points
        :type t: float or numpy.ndarray

        :return: Upper bound of the trace of inverse at inquiry points.
        :rtype: float or numpy.ndarray
        """

        # Compute trace at interpolant points
        if self.tau0 is None:
            self.ComputeTraceInvOfInputMatrices()

        # Upper bound of tau
        tau_ub_inv = 1.0/self.tau0 + t
        tau_ub = 1.0 / tau_ub_inv

        # Upper bound of trace
        trace_ub = tau_ub * self.trace_Binv

        return trace_ub
