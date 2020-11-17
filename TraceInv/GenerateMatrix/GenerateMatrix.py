# =======
# Imports
# =======

from .GeneratePoints import GeneratePoints
from .CorrelationMatrix import CorrelationMatrix
import matplotlib.pyplot as plt

# ===============
# Generate Matrix
# ===============

def GenerateMatrix(
        NumPoints,
        DecorrelationScale=0.1,
        nu=0.5,
        UseSparse=False,
        GridOfPoints=True,
        KernelThreshold=0.03,
        RunInParallel=False,
        PlotFlag=False):
    """
    Generates a symmetric and positive-definite.
    
    The generated matrx is a correlation matrix based on Matern correlation of spatial distance of 
    a list of points in the unit square. The Matern correlation function can be characterized by two 
    parameters:

        * :math:`\\rho`: the discorrelation scale in :math:`\\rho \in (0,1]`. Smaller decorrelation produces correlation matrix that is closer to the identity matrix.
        * :math:`\\nu`: correlation smoothness, :math:`\\rho > 0`. Larger smoothness produces correlation similar to the Gaussian process. Smaller smoothness 
          produces more discrete correlation function.

    Also, the size of the generated matrix is determined by ``NumPoints`` which we refer to as :math:`n`,
    and ``GridOfPoints``.

        * If ``GridOfPoints`` is ``True`` (default value), then, the matrix is of the size :math:`n^2 \\times n^2`.
        * If ``GridOfPoints`` is ``False``, then, the matrix is of the size :math:`n \\times n`.

    The values of the correlation matrix are between :math:`0` and :math:`1`. To sparsify the generated matrix, 
    set ``KernelThreshold``, which makes all correlations below this threshold to be zero. If ``KernelThreshold`` is zet to
    zer, the matrix is not sparsified.

    .. note:: 
        
        Too large ``KernelThreshold`` might eradicate the positive-definiteness of the correlation matrix.

    For very large sparse matrices, use ``RunInParallel`` to generate the rows and columns of the correlation matrix in parallel.

    .. note:: 

        To use parallel processing, the package ``ray`` should be insatalled.

    :param NumPoints: Depending on ``GridOfPoints``, this is either the number of points or the number of grid points along an axis.
    :type NumPoints: int

    :param DecorrelationScale: A parameter of correlation function that scales distance.
    :type DecorrelationScale: float

    :param nu: A parameter of the Matern correlation function. ``nu`` modulates the smoothness of the stochastic process.
    :type nu: float

    :param UseSparse: Flag to indicate the correlation matrix should be sparse or dense matrix.
    :type UseSparse: bool

    :param KernelThreshold: To sparsify the matrix (if ``UseSparse`` is ``True``), the correlation function values below this threshold 
        is set to zero.
    :type Kernel Threshold: float

    :param RunInParallel: Runs the code in parallel. Note that the ``ray`` module should be uncommented.
    :type RunInParallel: bool

    :param PlotFlag: If ``True``, the matrix will be ploted.
    :type PlotFlag: bool

    :return: Correlation matrix. If ``x`` and ``y`` are ``n*1`` arrays, the correlation ``K`` is ``n*n`` matrix.
    :rtype: ndarray or sparse array
    """

    # Generate a set of points in the unit square
    x,y = GeneratePoints(NumPoints,GridOfPoints)

    # Compute the correlation between the set of points
    K = CorrelationMatrix(x,y,DecorrelationScale,nu,UseSparse,KernelThreshold,RunInParallel)

    # Plot Correlation Matrix
    if PlotFlag:
        fig,ax = plt.subplots()
        p = ax.matshow(K)
        fig.colorbar(p,ax=ax)
        plt.title('Correlation Matrix')
        plt.show()

    return K
