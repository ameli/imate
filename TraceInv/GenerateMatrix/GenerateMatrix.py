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
    The main module of this package, which generates a symmetric and positive-definite correlation matrix.

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
