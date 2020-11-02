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
    Computes the trace of inverse of an invertible matrix :math:`\\mathbf{A} + t \\mathbf{I}` using eigenvalues of 
    :math:`\\mathbf{A}` by

    .. math::

        \\mathrm{trace}\\left( (\\mathbf{A} + t \\mathbf{I})^{-1} \\right) = \\sum_{i = 1}^n \\frac{1}{\\lambda_i + t}

    where :math:`\\lambda_i` is the eigenvalue of :math:`\\mathbf{A}`.

    * The result is an exact value.
    * This class does not accept interpolant points.
    * The input matrix :math:`\\mathbf{A}` can be either sparse or dense.
    * In case of a sparse matrix, only a portion of its eigenvalues with the largest magnitude is computed and the rest 
      of its eigenvalues is assumed to be negligible.
    """

    # ----
    # Init
    # ----

    def __init__(self,A,NonZeroRatio=0.9,Tol=1e-3):
        """
        Cnstructor of the class.

        :param A: Invertible matrix, can be rither dense or sparse matrix.
        :type A: ndarray

        :param NonZeroRatio:
        """

        # Base class constructor
        super(EigenvaluesMethod,self).__init__(A)

        # Attiributes
        self.NonZeroRatio = NonZeroRatio
        self.Tol = Tol

        # Initialize Interpolator
        self.A_eigenvalues = None
        self.InitializeInterpolator()

    # -----------------------
    # Initialize Interpolator
    # -----------------------

    def InitializeInterpolator(self):
        """
        Initializes the ``A_eigenvalues`` member data of the class.

        If the matrix ``A`` is sparse, it is not possible to find all of its eigenvalues. We only find
        90 percent of its eigenvalues with the larges magnitude and we assume the rest of the 
        eigenvalues are close to zero.
        """
        
        print('Initialize interpolator ...',end='')

        # Use Eigenvalues Method
        if self.UseSparse:

            n = A.shape[0]
            self.A_eigenvalues = numpy.zeros(n)

            # find 90% of eigenvalues and assume the rest are very close to zero.
            NumNoneZeroEig = int(n*self.NonZeroRatio)
            self.A_eigenvalues[:NumNoneZeroEig] = scipy.sparse.linalg.eigsh(
                    A,NumNoneZeroEig,which='LM',tol=self.Tol,return_eigenvectors=False)

        else:
            # A is dense matrix
            self.A_eigenvalues = scipy.linalg.eigh(self.A,eigvals_only=True,check_finite=False)

        print(' Done.')

    # -----------
    # Interpolate
    # -----------

    def Interpolate(self,t):
        """
        Interpolates the trace of inverse of ``A + t*I``.

        This function computes the trace of inverse using the eigenvalues by:

        .. math:: 

            \\mathrm{trace}\\left( (\\mathbf{A} + t \\mathbf{I})^{-1} \\right) = \\sum_{i = 1}^n \\frac{1}{\\lambda_i + t}

        where :math:`\\lambda_i` is the eigenvalue of :math:`\\mathbf{A}`.

        :param t: A real variable to form the linear matrix function ``A + tI``.
        :type t: float

        :return: The interpolated value of the trace of inverse of ``A + tI``.
        :rtype: float
        """
        
        T = numpy.sum(1.0/(self.A_eigenvalues + t))

        return T
