# =======
# Imports
# =======

from .GeneratePoints import GeneratePoints
from .CorrelationMatrix import CorrelationMatrix

try:
    from .._Utilities.PlotUtilities import *
    from .._Utilities.PlotUtilities import LoadPlotSettings
    from .._Utilities.PlotUtilities import SavePlot
    PlotModulesExist = True
except:
    PlotModulesExist = False

__all__ = ['GenerateMatrix']

# ===============
# Generate Matrix
# ===============

def GenerateMatrix(
        NumPoints=20,
        DecorrelationScale=0.1,
        nu=0.5,
        UseSparse=False,
        GridOfPoints=True,
        KernelThreshold=0.03,
        RunInParallel=False,
        Plot=False,
        Verbose=False):
    """
    Generates symmetric and positive-definite matrix for test purposes.
    
    The generated matrix is a correlation matrix based on Matern correlation of spatial distance of 
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
    set ``KernelThreshold``, which makes all correlations below this threshold to be zero. If ``KernelThreshold`` is set to
    zero, the matrix is not sparsified.

    .. note:: 
        
        Too large ``KernelThreshold`` might eradicate the positive-definiteness of the correlation matrix.

    For very large sparse matrices, use ``RunInParallel`` to generate the rows and columns of the correlation matrix in parallel.

    .. note:: 

        To use parallel processing, the package ``ray`` should be installed.

    Plotting:
        If the option ``Plot`` is set to ``True``, it plots the generated matrix.

        * If no graphical backend exists (such as running the code on a remote server or manually disabling the X11 backend), the plot will not be shown, rather, it will ve saved as an ``svg`` file in the current directory. 
        * If the executable ``latex`` is on the path, the plot is rendered using :math:`\rm\LaTeX`, which then, it takes a bit longer to produce the plot. 
        * If :math:`\rm\LaTeX` is not installed, it uses any available San-Serif font to render the plot.

   .. note::

       To manually disable interactive plot display, and save the plot as ``SVG`` instead, add the following in the
       very begining of your code before importing ``TraceInv``:

       .. code-block:: python
         
           >>> import os
           >>> os.environ['TRACEINV_NO_DISPLAY'] = 'True'

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

    :param Plot: If ``True``, the matrix will be plotted.
    :type Plot: bool

    :param Verbose: If ``True``, prints some information during the process. Default is ``False``.
    :type Verbose: bool

    :return: Correlation matrix. If ``x`` and ``y`` are ``n*1`` arrays, the correlation ``K`` is ``n*n`` matrix.
    :rtype: numpy.ndarray or sparse array

    **Example:**

    Generate a matrix of the shape ``(20**2,20**2)`` based on mutual correlation of a grid of 20x20 points on unit square

    .. code-block:: python

       >>> from TraceInv import GenerateMatrix
       >>> A = GenerateMatrix(NumPoints=20)

    Generate a correlation matrix of shape ``(20,20)`` based on 20 random points in unit square. Default for ``GridOfPoints`` is True.

    .. code-block:: python

       >>> A = GenerateMatrix(NumPoints=20,GridOfPoints=False)

    Generate a matrix of shape ``(20**2,20**2)`` with stronger spatial correlation. Default for ``DecorrelationScale`` is ``0.1``.

    .. code-block:: python

       >>> A = GenerateMatrix(NumPoints=20,DecorrelationScale=0.3)

    Generate a correlation matrix with more smoothness.. Default for ``nu`` is ``0.5``.

    .. code-block:: python

       >>> A = GenerateMatrix(NumPoints=20,nu=2.5)

    Sparsify correlation matrix (makes all entries below 0.03 to zero). Default for ``KernelThreshold`` is ``0.03``.

    .. code-block:: python

       >>> A = GenerateMatrix(NumPoints=20,UseSparse=True,KernelThreshold=0.03)

    For very large correlation matrices, generate the rows and columns are generated in parallel.
    To use ``RunInParallel`` option, the package ``ray`` should be installed.

    .. code-block:: python

       >>> A = GenerateMatrix(NumPoints=100,UseSparse=True,RunInParallel=True)

    Plot the matrix by

    .. code-block:: python

        >>> A = GenerateMatrix(NumPoints=30,Plot=True)
    """

    # Generate a set of points in the unit square
    x,y = GeneratePoints(NumPoints,GridOfPoints)

    # Compute the correlation between the set of points
    K = CorrelationMatrix(x,y,DecorrelationScale,nu,UseSparse,KernelThreshold,RunInParallel,Verbose)

    # Plot Correlation Matrix
    if Plot:
        PlotMatrix(K,UseSparse,Verbose)

    return K

# ===========
# Plot Matrix
# ===========

def PlotMatrix(K,UseSparse,Verbose=False):
    """
    Plots the matrix ``K``. 

    If ``K`` is a sparse matrix, it plots all non-zero elements with single color
    regardless of their values, and leaves the zero elements white.

    Whereas, if ``K`` is not a sparse matrix, the colormap of the plot
    correspond to the value of the elements of the matrix.

    If a graphical backend is not provided, the plot is not displayed,
    rather saved as ``SVG`` file in the current directory of user.

    :param K: matrix to plot
    :type K : numpy.ndarray or scipy.sparse.csc_matrix

    :param UseSparse: Determine whether the matrix is dense or sparse
    :type UseSparse: bool

    :param Verbose: If ``True``, prints some information during the process. Default is ``False``.
    :type Verbose: bool
    """

    # Load plot settings
    if PlotModulesExist:
        LoadPlotSettings()
    else:
        raise ImportError("Cannot load plot settings.")

    # Figure
    fig,ax = plt.subplots(figsize=(6,4))

    if UseSparse:
        # Plot sparse matrix
        p = ax.spy(K,markersize=1,color='blue',rasterized=True)
    else:
        # Plot dense matrix
        p = ax.matshow(K,cmap='Blues')
        cbar = fig.colorbar(p,ax=ax)
        cbar.set_label('Correlation')

    ax.set_title('Correlation Matrix',y=1.11)
    ax.set_xlabel('Index $i$')
    ax.set_ylabel('Index $j$')

    plt.tight_layout()
    
    # Check if the graphical backend exists
    if matplotlib.get_backend() != 'agg':
        plt.show()
    else:
        # write the plot as SVG file in the current working directory
        SavePlot(plt,'CorrelationMatrix',TransparentBackground=True)
