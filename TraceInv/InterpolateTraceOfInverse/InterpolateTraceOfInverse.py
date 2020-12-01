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

try:
    from ..Utilities.PlotUtilities import *
    from ..Utilities.PlotUtilities import LoadPlotSettings
    from ..Utilities.PlotUtilities import SavePlot
    PlotModulesExist = True
except:
    PlotModulesExist = False

# ======================
# Trace Estimation Class
# ======================

class InterpolateTraceOfInverse(object):
    """
    Interpolates the trace of inverse of affine matrix functions.

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

    :param InterpolationOptions: Additional options to pass to each specific interpolator class. 
        See the sub-classes of :mod:`TraceInv.InterpolantBaseClass`.
    :type InterpolationOptions: \*\*kwargs

    **Methods:**

    =======================  =========================================  ============  =============  ============
    ``InterpolationMethod``  Description                                Matrix size   Matrix type    Results
    =======================  =========================================  ============  =============  ============
    ``'EXT'``                Computes trace directly, no interpolation  Small         dense, sparse  exact
    ``'EIG'``                Uses Eigenvalues of matrix                 Small         dense, sparse  exact
    ``'MBF'``                Monomial Basis Functions                   Small, large  dense, sparse  interpolated
    ``'RMBF'``               Root monomial basis functions              small, large  dense, sparse  interpolated
    ``'RBF'``                Radial basis functions                     small, large  dense, sparse  interpolated
    ``'RPF'``                Rational polynomial functions              small, large  dense, sparse  interpolated
    =======================  =========================================  ============  =============  ============

    **Details:**

    An affine matrix function is defined by

    .. math::

        t \mapsto \mathbf{A} + t \mathbf{B}

    where :math:`\mathbf{A}` and :math:`\mathbf{B}` are invertible matrices and :math:`t` is a real parameter.

    This module interpolates the function

    .. math::

        t \mapsto \mathrm{trace} \left( (\mathbf{A} + t \mathbf{B})^{-1} \\right)

    The interpolator is initialized by providing :math:`p` interpolant points :math:`t_i`, :math:`i = 1,\dots,p`,
    which are often logarithmically spaced in some interval :math:`t_i \in [t_1,t_p]`. 
    The interpolator can interpolate the above function at inquity points :math:`t \in [t_1,t_p]` 
    using various methods.

    **Examples:**

    Interpolate the trace of inverse of the affine matrix function :math:`\mathbf{A} + t \mathbf{B}`:

    .. code-block:: python

        >>> # Generate two sample matrices (symmetric and positive-definite)
        >>> from TraceInv import GenerateMatrix
        >>> A = GenerateMatrix(NumPoints=20,DecorrelationScale=1e-1)
        >>> B = GenerateMatrix(NumPoints=20,DecorrelationScale=2e-2)

        >>> # Initialize interpolator object
        >>> from TraceInv import InterpolateTraceOfInverse
        >>> InterpolantPoints = [1e-2,1e-1,1,1e1]
        >>> TI = InterpolateTraceOfInverse(A,B,InterpolantPoints)

        >>> # Interpolate at an inquiry point t = 0.4
        >>> t = 4e-1
        >>> Trace = TI.Interpolate(t)

    Interpolate an array of inquiry points

    .. code-block:: python

        >>> # Interpolate at an array of points
        >>> import numpy
        >>> t_array = numoy.logspace(-2,1,10)
        >>> Trace = TI.Interpolator(t_array)

    By default, the interpolation method is ``'RMBF'``. Use a different interpolation method, such as ``'RBF'`` by

    .. code-block:: python

        >>> TI = InterpolateTraceOfInverse(A,B,InterpolantPoints,InterpolationMethod='RBF')

    By default, the trace is computed with the Cholesky decomposition method as the interpolant points. Configure the
    computation method by ``ComputeOptions`` as

    .. code-block:: python

        >>> # Specify arguments to TraceInv.ComputeTraceOfInverse in a dictionary
        >>> ComputeOptions = \ 
        ... {
        ...     'ComputeMethod': 'hutchinson',
        ...     'NumIterations': 20
        ... }

        >>> # Pass the options to the interpolator
        >>> TI = InterpolateTraceOfInverse(A,B,InterpolantPoints,ComputeOptions=ComputeOptions)

    .. seealso:: 
        This module calls a derived class of the base class :mod:`TraceInv.InterpolateTraceOfInverse.InterpolantBaseClass`.

    """

    # ----
    # Init
    # ----

    def __init__(self,A,B=None,InterpolantPoints=None,InterpolationMethod='RMBF',ComputeOptions={'ComputeMethod': 'cholesky'},
            Verbose=False,**InterpolationOptions):
        """
        Initializes the object depending on the method.
        """

        # Attributes
        self.InterpolationMethod = InterpolationMethod
        self.Verbose = Verbose
        self.n = A.shape[0]
        if InterpolantPoints is not None:
            self.p = len(InterpolantPoints)
        else:
            self.p = 0

        # Define an interpolation object depending on the given method
        if InterpolationMethod == 'EXT':
            # Exact computation, not interpolation
            self.Interpolator = ExactMethod(A,B,ComputeOptions=ComputeOptions,
                    Verbose=Verbose,**InterpolationOptions)

        elif InterpolationMethod == 'EIG':
            # Eigenvalues method
            self.Interpolator = EigenvaluesMethod(A,B,ComputeOptions=ComputeOptions,
                    Verbose=Verbose,**InterpolationOptions)

        elif InterpolationMethod == 'MBF':
            # Monomial Basis Functions method
            self.Interpolator = MonomialBasisFunctionsMethod(A,B,InterpolantPoints,ComputeOptions=ComputeOptions,
                    Verbose=Verbose,**InterpolationOptions)

        elif InterpolationMethod == 'RMBF':
            # Root Monomial Basis Functions method
            self.Interpolator = RootMonomialBasisFunctionsMethod(A,B,InterpolantPoints,ComputeOptions=ComputeOptions,
                    Verbose=Verbose,**InterpolationOptions)

        elif InterpolationMethod == 'RBF':
            # Radial Basis Functions method
            self.Interpolator = RadialBasisFunctionsMethod(A,B,InterpolantPoints,ComputeOptions=ComputeOptions,
                    Verbose=Verbose,**InterpolationOptions)

        elif InterpolationMethod == 'RPF':
            # Rational Polynomial Functions method
            self.Interpolator = RationalPolynomialFunctionsMethod(A,B,InterpolantPoints,ComputeOptions=ComputeOptions,
                    Verbose=Verbose,**InterpolationOptions)

        else:
            raise ValueError("'InterpolationMethod' is invalid. Select one of 'EXT', 'EIG', 'MBF', 'RMBF', 'RBF', or 'RPF'.")

    # -------
    # Compute
    # -------

    def Compute(self,t):
        """
        Computes the function :math:`\mathrm{trace}\left( (\mathbf{A} + t \mathbf{B})^{-1} \\right)` 
        at the input point :math:`t` using exact method.

        The computation method used in this function is exact (no interpolation). This function 
        is primarily used to compute trace on the *interpolant points*.

        :param: t: An inquiry point, which can be a single number, or an array of numbers.
        :type t: float or numpy.array

        :return: The exact value of the trace
        :rtype: float or numpy.array
        """
        
        if isinstance(t,Number):
            # Single number
            T =  self.Interpolator.Compute(t)

        else:
            # An array of points
            T = numpy.empty((len(t),),dtype=float)
            for i in range(len(t)):
                T[i] =  self.Interpolator.Compute(t[i])

        return T

    # ---------------------------
    # Compare With Exact Solution
    # ---------------------------

    def CompareWithExactSolution(self,t,Trace):
        """
        Computes the trace with exact method (no interpolation), then compares it with 
        the interpolated solution.

        :param t: Inquiry points
        :type t: numpy.ndarray

        :param Trace: The interpolated computation of trace.
        :type Trace: float or numpy.ndarray

        :returns:
            - Exact solution of trace.
            - Relative error of interpolated solution compared to the exact solution.
        :rtype:
            - float or numpy.ndarray
            - float or numpy.ndarray
        """

        if self.InterpolationMethod == 'EXT':

            # The Trace results are already exact. No need to recompute them again.
            Trace_Exact = Trace
            Trace_RelativeError = numpy.zeros(t.shape)

        else:

            # Compute exact solution
            Trace_Exact = self.Compute(t)
            Trace_RelativeError = (Trace - Trace_Exact) / (Trace_Exact)

        return Trace_Exact,Trace_RelativeError

    # -----------
    # Interpolate
    # -----------

    def Interpolate(self,t,CompareWithExact=False,Plot=False):
        """
        Interpolates :math:`\mathrm{trace} \left( (\mathbf{A} + t \mathbf{B})^{-1} \\right)` at :math:`t`.

        This is the main interface function of this module and it is used after the interpolation
        object is initialized.

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

        :return: The interpolated value of the trace.
        :rtype: float or numpy.array
        """
        
        if isinstance(t,Number):
            # Single number
            Trace =  self.Interpolator.Interpolate(t)

        else:
            # An array of points
            Trace = numpy.empty((len(t),),dtype=float)
            for i in range(len(t)):
                Trace[i] =  self.Interpolator.Interpolate(t[i])

        # Compare with exact solution
        if CompareWithExact:

            # Since this method is exact, there is no need to compute exact again.
            Trace_Exact,Trace_RelativeError = self.CompareWithExactSolution(t,Trace)

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

    # -----------
    # Lower Bound
    # -----------

    def LowerBound(self,t):
        """
        Lower bound of the function :math:`t \mapsto \mathrm{trace} \left( (\mathbf{A} + t \mathbf{B})^{-1} \\right)`.

        The lower bound is given by

        .. math::
        
            \mathrm{trace}((\mathbf{A}+t\mathbf{B})^{-1}) \geq \\frac{n^2}{\mathrm{trace}(\mathbf{A}) + \mathrm{trace}(t \mathbf{B})}

        :param t: An inquiry point or an array of inquiry points.
        :type t: float or numpy.array

        :return: Lower bound of the affine matrix function.
        :rtype: float or numpy.array

        .. seealso::

            This function is implemented in :meth:`TraceInv.InterpolateTraceOfInverse.InterpolantBaseClass.LowerBound`.
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
        Upper bound of the function :math:`t \mapsto \mathrm{trace} \left( (\mathbf{A} + t \mathbf{B})^{-1} \\right)`.

        The upper bound is given by

        .. math::
                
                \\frac{1}{\\tau(t)} \geq \\frac{1}{\\tau_0} + t

        where

        .. math::

                \\tau(t) = \\frac{\mathrm{trace}\\left( (\mathbf{A} + t \mathbf{B})^{-1}  \\right)}{\mathrm{trace}(\mathbf{B}^{-1})}

        and :math:`\\tau_0 = \\tau(0)`.

        :param t: An inquiry point or an array of inquiry points.
        :type t: float or numpy.array

        :return: Lower bound of the affine matrix function.
        :rtype: float or numpy.array

        .. seealso::

            This function is implemented in :meth:`TraceInv.InterpolateTraceOfInverse.InterpolantBaseClass.UpperBound`.
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

    # ------------------
    # Plot Interpolation
    # ------------------

    def PlotInterpolation(self,InquiryPoints,Trace_Interpolated,Trace_Exact=None,Trace_RelativeError=None):
        """
        Plots the interpolation results, together with the comparison with the exact solution and
        the relative error of the interpolation.

        To plot, set ``Plot=True`` argument in :mod:`TraceInv.InterpolateTraceOfInverse`.

        :param InquiryPoints: Inquiry points
        :type InquiryPoints: numpy.ndarray

        :param Trace_Interpolated: Interpolation of the trace at inquiry points.
        :type Trace_Interpolated: numpy.ndarray

        :param Trace_Exact: Exact solutions of the trace at inquiry points. 
            If this variable is not None, it will be plotted together with the interpolated results.
        :type Trace_Exact: numpy.ndarray

        :param Trace_RelativeError: Relative errors of the interpolation with respect to the exact solution.
            If not None, the relative errors will be plotted on a second axis.
        :type Trace_RelativeError: numpy.ndarray
        """

        # Check t should be an array
        if numpy.isscalar(InquiryPoints) or (InquiryPoints.size == 1):
            raise ValueError("Argument 'InquiryPoints' should be an array of length greater than one to be able to plot results.")

        # Load plot settings
        if PlotModulesExist:
            LoadPlotSettings()
        else:
            raise ImportError("'matplotlib' or 'seaborn' is either not installed or cannot be imported.")

        # Plot results
        if Trace_RelativeError is None:
            # One subplot
            fig,ax = plt.subplots(figsize=(5,4))
            ax = [ax]
        else:
            # Two subplots
            fig,ax = plt.subplots(ncols=2,figsize=(9,4))

        # Plot interpolant points with their exact values
        if self.Interpolator.p > 0:
            ax[0].semilogx(self.Interpolator.t_i,self.Interpolator.trace_i,'o',color='red',label='Interpolant points',zorder=20)

        # Plot exact values
        if Trace_Exact is not None:
            ax[0].semilogx(InquiryPoints,Trace_Exact,color='red',label='Exact')

        # Plot interpolated results
        ax[0].semilogx(InquiryPoints,Trace_Interpolated,color='black',label='Interpolated')

        ax[0].grid(True)
        ax[0].set_xlim([InquiryPoints[0],InquiryPoints[-1]])
        ax[0].set_xlabel('$t$')
        if self.Interpolator.BIsIdentity:
            ax[0].set_ylabel('trace$(A + t I)^{-1}$')
        else:
            ax[0].set_ylabel('trace$(A + t B)^{-1}$')
        ax[0].set_title('Trace of Inverse')
        ax[0].legend()

        # Plot relative error in percent
        if Trace_RelativeError is not None:
            if self.Interpolator.p > 0:
                ax[1].semilogx(self.Interpolator.t_i,numpy.zeros(self.Interpolator.p),'o',color='red',label='Interpolant points',zorder=20)
            ax[1].semilogx(InquiryPoints,100.0*Trace_RelativeError,color='black',label='Interpolated')
            ax[1].grid(True)
            ax[1].set_xlim([InquiryPoints[0],InquiryPoints[-1]])
            ax[1].set_xlabel('$t$')
            ax[1].set_ylabel('Relative Error (in Percent)')
            ax[1].set_title('Relative Error')
            ax[1].yaxis.set_major_formatter(mtick.PercentFormatter())
            ax[1].legend()

        plt.tight_layout()

        # Check if the graphical backend exists
        if matplotlib.get_backend() != 'agg':
            plt.show()
        else:
            # Save the plot as SVG file in the current directory
            SavePlot(plt,'InterpolationResults',TransparentBackground=True)
