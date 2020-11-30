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
    Computes the trace of inverse of an invertible matrix :math:`\\mathbf{A} + t \\mathbf{B}` using 
    exact method (no interpolation is performed).
    This class does not accept interpolant points as the result is not interpolated, rather, used
    as a benchmark to compare the exact versus the interpolated solution of the other classes.

    **Class Inheritance:**

    .. inheritance-diagram:: TraceInv.InterpolateTraceOfInverse.ExactMethod
        :parts: 1

    :param A: A positive-definite matrix. Matrix can be dense or sparse.
    :type A: numpy.ndarray or scipy.sparse.csc_matrix

    :param B: A positive-definite matrix. 
        If ``None`` or not provided, it is assumed that ``B`` is an identity matrix of the shape of ``A``.
    :type B: numpy.ndarray or scipy.sparse.csc_matrix

    :param ComputeOptions: A dictionary of arguments to pass to :mod:`TraceInv.ComputeTraceOfInverse` module.
    :type ComputeOptions: dict

    :param Verbose: If ``True``, prints some information on the computation process. Default is ``False``.
    :type Verbose: bool

    :example:

    This class can be invoked from :class:`TraceInv.InterpolateTraceOfInverse.InterpolateTraceOfInverse` module 
    using ``InterpolationMethod='EXT'`` argument.

    .. code-block:: python

        >>> from TraceInv import GenerateMatrix
        >>> from TraceInv import InterpolateTraceOfInverse
        
        >>> # Generate a symmetric positive-definite matrix of the shape (20**2,20**2)
        >>> A = GenerateMatrix(NumPoints=20)
        
        >>> # Create an object that interpolates trace of inverse of A+tI (I is identity matrix)
        >>> TI = InterpolateTraceOfInverse(A,InterpolationMethod='EXT')
        
        >>> # Interpolate A+tI at some input point t
        >>> t = 4e-1
        >>> trace = TI.Interpolate(t)

    .. seealso::

        The result of the ``EXT`` method is identical with the eigenvalue method ``EIG``, 
        which is given by :class:`TraceInv.InterpolateTraceOfInverse.EigenvaluesMethod`.
    """

    # ----
    # Init
    # ----

    def __init__(self,A,B=None,ComputeOptions={},Verbose=False):
        """
        Initializes the parent class.
        """

        # Base class constructor
        super(ExactMethod,self).__init__(A,B=B,ComputeOptions=ComputeOptions,Verbose=Verbose)

    # -----------
    # Interpolate
    # -----------

    def Interpolate(self,t,CompareWithExact=False,Plot=False):
        """
        This function does not interpolate, rather exactly computes 
        :math:`\mathrm{trace} \left( (\mathbf{A} + t \mathbf{B})^{-1}  \rright)`

        :param t: The inquiry point(s).
        :type t: float, list, or numpy.array

        :param CompareWithExact: If ``True``, it computes the trace with exact solution, then compares it with the interpolated 
            solution. The return values of the ``Interpolate()`` functions become interpolated trace, exact solution, 
            and relative error. **Note:** When this option is enabled, the exact solution will be computed for all inquiry points, 
            which can take a very long time. Default is ``False``.
        :type CompareWithExact: bool

        :param Plot: If ``True``, it plots the interpolated trace versus the inquiry points. In addition, if the option
            ``CompareWithExact`` is also set to ``True``, the plotted diagram contains both interpolated and exact solutions
            and the relative error of interpolated solution with respect to the exact solution.
        :type Plot: bool

        :return: The exact value of the trace.
        :rtype: float or numpy.array
        """
 
        # Do not interpolate, instead compute the exact value
        Trace = self.Compute(t)

        # Compare with exact solution
        if CompareWithExact:

            # Since this method is exact, there is no need to compute exact again.
            Trace_Exact = Trace
            Trace_RelativeError = numpy.zeros(t.shape)

        # Plot
        if Plot:
            if CompareWithExact:
                self.PlotInterpolation(t,Trace,Trace_Exact,Trace_RelativeError)
            else:
                self.PlotInterpolation(t,Trace)

        # Return
        if CompareWithExact:
            return Trace,Trace_Exact,Trace_RelativeError
        else:
            return Trace
