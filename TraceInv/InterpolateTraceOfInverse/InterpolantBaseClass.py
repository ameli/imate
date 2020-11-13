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

    * :class:`TraceInv.InterpolateTraceOfInverse.ExactMethod`
    * :class:`TraceInv.InterpolateTraceOfInverse.EigenvaluesMethod`
    * :class:`TraceInv.InterpolateTraceOfInverse.MonomialBasisFunctionsMethod``
    * :class:`TraceInv.InterpolateTraceOfInverse.RootMonomialBasisFunctionsMethod``
    * :class:`TraceInv.InterpolateTraceOfInverse.RadialBasisFunctionsMethod``
    * :class:`TraceInv.InterpolateTraceOfInverse.RationalPolynomialFunctionsMethod``

    .. inheritance-diagram:: TraceInv.InterpolateTraceOfInverse.InterpolantBaseClass
        :parts: 1

    """

    # ----
    # Init
    # ----

    def __init__(self,A,B=None,InterpolantPoints=None,ComputeOptions={}):
        """
        The initialization function does the followings:

        * Initializes te interpolant points and the input matrix.
        * Scale the interpolant points if needed.
        * Comutes the trace of inverse at the interpolant points.
        """

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
            self.eta_i = numpy.array(InterpolantPoints)
            self.p = self.eta_i.size
            self.trace_i = self.ComputeTraceAtInterpolantPoints()
            self.tau_i = self.trace_i / self.trace_Binv

            # Scale interpolant points
            self.Scale_eta = None
            self.ScaleInterpolantPoints()

    # ----------------------------------
    # Compute TraceInv of Input Matrices
    # ----------------------------------

    def ComputeTraceInvOfInputMatrices(self):
        """
        Computes the trace of inverse of input matrices ``A`` and ``B``, and their ratio ``tau0``.

        This function sets the following class attirbutes:

        * ``self.trace_Ainv``: trace of inverse of ``A``.
        * ``self.trace_Binv``: trace of inverse of ``B``.
        * ``self.tau0``: the ratio of ``self.trace_Ainv`` over ``self.trace_Binv``.

        The ratio :math:`\\tau_0` is defined by:

        .. math::

            \\tau_0 = \\frac{\mathrm{trace}(\mathbf{A}^{-1})}{\mathrm{trace}(\mathbf{B}^{-1})}.
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
        If the largest interpolant point in ``self.InterpolantPoints`` is greater than ``1``, 
        this function rescales their range to max at ``1``. This function is intended to be 
        used internally.
        """

        # Scale eta, if some of eta_i are greater than 1
        self.Scale_eta = 1.0
        if self.eta_i.size > 0:
            if numpy.max(self.eta_i) > 1.0:
                self.Scale_eta = numpy.max(self.eta_i)

    # -------
    # Compute
    # -------

    def Compute(self,t):
        """
        Computes the trace of inverse of ``A+tB`` at point ``t`` with exact method, that is,
        no interpolation is used and the result is exact.
       
        This function is primarily used internally to compute trace of inverse at interpolant points. 
        This function uses :class:`TraceInv.ComputeTraceOfInverse` class with options described by
        ``self.ComputeOptions``.
        """
        
        An = self.A + t * self.B
        T = ComputeTraceOfInverse(An,**self.ComputeOptions)

        return T
        
    # -----------------------------------
    # Compute Trace At Interpolant Points
    # -----------------------------------

    def ComputeTraceAtInterpolantPoints(self):
        """
        Computes the trace of inverse of ``A+tB`` at interpolant points ``self.InterpolantPoints``.
        At each interpolant point, the trace is computed exactly using :class:`TraceInv.ComputeTraceOfInverse`.
        """

        print('Evaluate function at interpolant points ...',end='')

        # Compute trace at interpolant points
        trace_i = numpy.zeros(self.p)
        for i in range(self.p):
            trace_i[i] = self.Compute(self.eta_i[i])

        print(' Done.')

        return trace_i

    # -----------
    # Lower Bound
    # -----------

    def LowerBound(self,t):
        """
        Computes the lower bound of the trce of the inverse of the one-parameter function ``A+tB``.

        The lower bound is given by Remark 2.2 of [Ameli-2020]_ as

        .. math::
                
                \mathrm{trace}((\mathbf{A}+t\mathbf{B})^{-1}) \geq \\frac{n^2}{\mathrm{trace}(\mathbf{A}) + \mathrm{trace}(t \mathbf{B})}
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
        Computes the upper bound of the trace of the inverse of the one-parameter function ``A+tB``.

        The upper bound is given by Theorem 1 of [Ameli-2020]_ as

        .. math::
                
                \\frac{1}{\\tau(t)} \geq \\frac{1}{\\tau_0} + t

        where

        .. math::

                \\tau(t) = \\frac{\mathrm{trace}\\left( (\mathbf{A} + t \mathbf{B})^{-1}  \\right)}{\mathrm{trace}(\mathbf{B}^{-1})}

        and :math:`\\tau_0 = \\tau(0)`.
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
