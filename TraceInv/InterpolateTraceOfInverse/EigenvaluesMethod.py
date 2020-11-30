# =======
# Imports
# =======

from __future__ import print_function
import numpy
import scipy
from scipy import linalg
from scipy import sparse
from .InterpolantBaseClass import InterpolantBaseClass

# ==================
# Eigenvalues Method
# ==================

class EigenvaluesMethod(InterpolantBaseClass):
    """
    Computes the trace of inverse of an invertible matrix :math:`\\mathbf{A} + t \\mathbf{B}` using eigenvalues of 
    :math:`\\mathbf{A}`  and :math:`\mathbf{B}`.
    
    The trace of computed by

    .. math::

        \\mathrm{trace}\\left( (\\mathbf{A} + t \\mathbf{B})^{-1} \\right) = \\sum_{i = 1}^n \\frac{1}{\\lambda_i + t \\mu_i}

    where :math:`\\lambda_i` is the eigenvalue of :math:`\\mathbf{A}` 
    and :math:`\\mu_i` is the eigenvalue of :math:`\mathbf{B}`. 
    This class does not accept interpolant points as the result is not interpolated.

    **Class Inheritance:**

    .. inheritance-diagram:: TraceInv.InterpolateTraceOfInverse.EigenvaluesMethod
        :parts: 1

    :param A: Invertible matrix, can be either dense or sparse matrix.
    :type A: numpy.ndarray

    :param B: Invertible matrix, can be either dense or sparse matrix.
    :type B: numpy.ndarray

    :param ComputeOptions: A dictionary of input arguments for :mod:`TraceInv.ComputeTraceOfInverse.ComputeTraceOfInverse` module.
    :type ComputeOptions: dict

    :param NonZeroRatio: The ratio of the number of eigenvalues to be assumed non-zero over all eigenvalues.
        This option is only used for sparse matrices where as assume some of eigenvalues are very small 
        and we are only interested in computing non-zero eigenvalues. In practice, it is not possible to 
        compute all eigenvalues of a large sparse matrix.
        Default is ``0.9`` indicating to compute 90 percent of the eigenvalues with the largest magnitude
        and assume the rest of the eigenvalues are zero.
    :type NonZeroRatio: int

    :param Tolerance: Tolerance of computing eigenvalues. This option is only used for sparse matrices.
        Default value is ``1e-3``.
    :type Tolerance: float

    :param Verbose: If ``True``, prints some information on the computation process. Default is ``False``.
    :type Verbose: bool

    .. note::

        The input matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` can be either sparse or dense.
        In case of a **sparse matrix**, only some of the eigenvalues with the largest magnitude is computed 
        and the rest of its eigenvalues is assumed to be negligible. The ratio of computed eigenvalues over 
        the total number of eigenvalues can be set by ``NonZeroRatio``.
        The tolerance at which the eigenvalues are computed can be set by ``Tolerance``.

    **Example:**

    This class can be invoked from :class:`TraceInv.InterpolateTraceOfInverse.InterpolateTraceOfInverse` module 
    using ``InterpolationMethod='EIG'`` argument.

    .. code-block:: python

        >>> from TraceInv import GenerateMatrix
        >>> from TraceInv import InterpolateTraceOfInverse
        
        >>> # Generate a symmetric positive-definite matrix of the shape (20**2,20**2)
        >>> A = GenerateMatrix(NumPoints=20)
        
        >>> # Create an object that interpolates trace of inverse of A+tI (I is identity matrix)
        >>> TI = InterpolateTraceOfInverse(A,InterpolatiionMethod='EIG')
        
        >>> # Interpolate A+tI at some input point t
        >>> t = 4e-1
        >>> trace = TI.Interpolate(t)

    .. seealso::

        The result of the ``EIG`` method is identical with the exact method ``EXT``, 
        which is given by :class:`TraceInv.InterpolateTraceOfInverse.ExactMethod`.
    """

    # ----
    # Init
    # ----

    def __init__(self,A,B=None,ComputeOptions={},NonZeroRatio=0.9,Tolerance=1e-3,Verbose=False):
        """
        Constructor of the class, which initializes the bases class and computes eigenvalues
        of the input matrices.
        """

        # Base class constructor
        super(EigenvaluesMethod,self).__init__(A,B=B,ComputeOptions=ComputeOptions,Verbose=Verbose)

        # Attributes
        self.NonZeroRatio = NonZeroRatio
        self.Tolerance = Tolerance

        # Initialize Interpolator
        self.A_eigenvalues = None
        self.B_eigenvalues = None
        self.InitializeInterpolator()

    # -----------------------
    # Initialize Interpolator
    # -----------------------

    def InitializeInterpolator(self):
        """
        Initializes the ``A_eigenvalues`` and ``B_eigenvalues``  member data of the class.

        .. note::
        
            If the matrix ``A`` is sparse, it is not possible to find all of its eigenvalues. We only find
            a fraction of the number of its eigenvalues with the larges magnitude and
            we assume the rest of the eigenvalues are close to zero.
        """
       
        if self.Verbose:
            print('Initialize interpolator ...',end='')

        # Find eigenvalues of A
        if self.UseSparse:

            # A is sparse
            self.A_eigenvalues = numpy.zeros(self.n)

            # find 90% of eigenvalues and assume the rest are very close to zero.
            NumNoneZeroEig = int(self.n*self.NonZeroRatio)
            self.A_eigenvalues[:NumNoneZeroEig] = scipy.sparse.linalg.eigsh(
                    A,NumNoneZeroEig,which='LM',tol=self.Tolerance,return_eigenvectors=False)

        else:
            # A is dense matrix
            self.A_eigenvalues = scipy.linalg.eigh(self.A,eigvals_only=True,check_finite=False)

        # Find eigenvalues of B
        if self.BIsIdentity:

            # B is identity
            self.B_eigenvalues = numpy.ones(self.n,dtype=float)

        else:
            # B is not identity
            if self.UseSparse:

                # B is sparse
                self.B_eigenvalues = numpy.zeros(self.n)

                # find 90% of eigenvalues and assume the rest are very close to zero.
                NumNoneZeroEig = int(self.n*self.NonZeroRatio)
                self.B_eigenvalues[:NumNoneZeroEig] = scipy.sparse.linalg.eigsh(
                        B,NumNoneZeroEig,which='LM',tol=self.Tolerance,return_eigenvectors=False)

            else:
                # B is dense matrix
                self.B_eigenvalues = scipy.linalg.eigh(self.B,eigvals_only=True,check_finite=False)

        if self.Verbose:
            print(' Done.')

    # -----------
    # Interpolate
    # -----------

    def Interpolate(self,t,CompareWithExact=False,Plot=False):
        """
        Computes the function :math:`\mathrm{trace}\left( (\mathbf{A} + t \mathbf{B})^{-1} \\right)` 
        at the input point :math:`t` by

        .. math:: 

            \\mathrm{trace}\\left( (\\mathbf{A} + t \\mathbf{B})^{-1} \\right) = \\sum_{i = 1}^n \\frac{1}{\\lambda_i + t \\mu_i}

        where :math:`\\lambda_i` is the eigenvalue of :math:`\\mathbf{A}`.
        and  :math:`\\mu_i` is the eigenvalue of :math:`\\mathbf{B}`.

        :param: t: An inquiry point, which can be a single number, or an array of numbers.
        :type t: float or numpy.array

        :param CompareWithExact: If ``True``, it computes the trace with exact solution, then compares it with the interpolated 
            solution. The return values of the ``Interpolate()`` functions become interpolated trace, exact solution, 
            and relative error. **Note:** When this option is enabled, the exact solution will be computed for all inquiry points, 
            which can take a very long time. Default is ``False``.
        :type CompareWithExact: bool

        :param Plot: If ``True``, it plots the interpolated trace versus the inquiry points. In addition, if the option
            ``CompareWithExact`` is also set to ``True``, the plotted diagram contains both interpolated and exact solutions
            and the relative error of interpolated solution with respect to the exact solution.
        :type Plot: bool

        :return: The exact value of the trace of inverse of ``A + tB``.
        :rtype: float
        """
       
        # Compute trace using eigenvalues
        Trace = numpy.sum(1.0/(self.A_eigenvalues + t * self.B_eigenvalues))

        # Compare with exact solution
        if CompareWithExact:
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
