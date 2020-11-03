# =======
# Imports
# =======

import numpy
import scipy
from scipy import special
from scipy import sparse
# import multiprocessing
import os
import logging
import warnings

try:
    import ray
    RayInstalled = True
except:
    RayInstalled = False

# ==================
# Correlation Kernel
# ==================

def CorrelationKernel(Distance,DecorrelationScale,nu):
    """
    Matern class correlation function.

    If ``nu`` is half integer, the Matern function has exponential form.
    Otherwise it is represented by Bessel function.

    :param Distance: The distance matrix (``n*n``) that represents the Euclidean distance between mutual points.
    :type Distance: ndarray

    :param DecorrelationScale: A parameter of correlation function that scales distance.
    :type DecorrelationScale: float

    :param nu: A parameter of the Matern correlation function. ``nu`` modulates the smoothness of the stochastic process.
    :type nu: float
    """

    # scaled distance
    ScaledDistance = Distance / DecorrelationScale

    if nu == 0.5:
        Correlation = numpy.exp(-ScaledDistance)

    elif nu == 1.5:
        Correlation = (1.0 + numpy.sqrt(3.0)*ScaledDistance) * numpy.exp(-numpy.sqrt(3.0)*ScaledDistance)

    elif nu == 2.5:
        Correlation = (1.0 + numpy.sqrt(5.0)*ScaledDistance + (5.0/3.0)*(ScaledDistance**2)) * numpy.exp(-numpy.sqrt(5.0)*ScaledDistance)

    elif nu < 100:
        
        # Change zero elements of ScaledDistance to a dummy number, to avoid multiplication of zero by Inf in Bessel function below
        ScaledDistance[0] = 1
        Correlation = ((2.0**(1.0-nu))/scipy.special.gamma(nu)) * ((numpy.sqrt(2.0*nu) * ScaledDistance)**nu) * scipy.special.kv(nu,numpy.sqrt(2.0*nu)*ScaledDistance)

        # Set diagonals of correlation to one, since we altered the diagonals of ScaledDistance
        Correlation[0] = 1

        if numpy.any(numpy.isnan(Correlation)):
            raise ValueError('Correlation has nan element. nu: %f, DecorelationScale: %f'%(nu,DecorrelationScale))
        if numpy.any(numpy.isinf(Correlation)):
            raise ValueError('Correlation has inf element. nu: %f, DecorelationScale: %f'%(nu,DecorrelationScale))

    else:
        # For nu > 100, assume nu is Inf. In this case, Matern function approaches Gaussian kernel
        Correlation = numpy.exp(-0.5*ScaledDistance**2)

    return Correlation

# =================================
# Compute Correlation For A Process
# =================================

# if RayInstalled:
# @ray.remote
def ComputeCorrelationForAProcess(DecorrelationScale,nu,KernelThreshold,x,y,UseSparse,NumCPUs,StartIndex):
    """
    Computes correlation at the ColumnIndex-th column and row of ``K``.

    * ``K`` is updated inplace.
    * This function is used as a partial function for parallel processing.

    * If ``StartIndex`` is ``None``, it fills all columns of correlation matrix ``K``.
    * If ``StartIndex`` is not ``None``, it fills only a sub-rang of columns of ``K`` from ``StartIndex`` to ``n`` by ``NumCPUs`` increment.

    .. note::

        To run the code in parallel, uncomment the ``@ray.remote`` directive before the definition of the function.

    :param DecorrelationScale: A parameter of correlation function that scales distance.
    :type DecorrelationScale: float

    :param nu: A parameter of the Matern correlation function. ``nu`` modulates the smoothness of the stochastic process.
    :type nu: float
    
    :param KernelThreshold: To sparsify the matrix (if ``UseSparse`` is ``True``), the correlation function values below this threshold 
        is set to zero.
    :type Kernel Threshold: float
    
    :param x: x-coordinates of the set of points. 
    :type x: array

    :param y: y-coordinates of the set of points. 
    :type y: array

    :param UseSparse: Flag to indicate the correlation matrix should be sparse or dense matrix.
    :type UseSparse: bool

    :param NumCPUs: Number of processors to employ parallel processing with ``ray`` package.
    :type NumCPUs: int

    :param StartIndex: The start index of the column of ``K`` to be filled.
        If this is ``None``, all columns of ``K`` are filled.
        If this is not ``None``, only a range of columns of ``K`` are filled.

    :type: int
    """

    n = x.size

    if UseSparse:
        K = scipy.sparse.lil_matrix((n,n))
    else:
        K = numpy.zeros((n,n),dtype=float)

    # Range of filling columns of correlation
    if StartIndex is None:
        Range = range(n)
    else:
        Range = range(StartIndex,n,NumCPUs)

    # Fill K only at each NumCPU columns starting from StartIndex
    for i in Range:

        # Euclidean distance of points
        Distance = numpy.sqrt((x[i:]-x[i])**2 + (y[i:] - y[i])**2)
        Correlation = CorrelationKernel(Distance,DecorrelationScale,nu)

        # Sparsify
        if UseSparse:
            Correlation[Correlation < KernelThreshold] = 0

        # Diagonal element
        K[i,i] = Correlation[0] * 0.5

        # Upper-right elements
        if i < n-1:
            K[i,i+1:] = Correlation[1:]

    if UseSparse:
        return K.tocsc()
    else:
        return K

# ==================
# Correlation Matrix
# ==================

def CorrelationMatrix(x,y,DecorrelationScale,nu,UseSparse,KernelThreshold=0.03,RunInParallel=False):
    """
    Generates correlation matrix ``K``.

    .. note::

        If the ``KernelThreshold`` is large, it causes:
            * ``K`` to not be positive-definite.
            * ``trace((K+etaI)**{-1})`` to oscillate.

    :param x: x-coordinates of the set of points. 
    :type x: array

    :param y: y-coordinates of the set of points. 
    :type y: array

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

    :return: Correlation matrix. If ``x`` and ``y`` are ``n*1`` arrays, the correlation ``K`` is ``n*n`` matrix.
    :rtype: ndarray or sparse array
    """

    print('Generate correlation matrix ...')

    # size of Distance matrix
    n = x.size

    # Check if the thresholding is not too much to avoid the correlation matrix becomes identity. Each point should have at least one neighbor point in correlation matrix.
    if UseSparse:
        # Compute Adjacency
        NumPointsAlongAxis = numpy.rint(numpy.sqrt(n))
        GridSize = 1.0 / (NumPointsAlongAxis - 1.0)
        KernelLength = -DecorrelationScale*numpy.log(KernelThreshold)
        Adjacency = KernelLength / GridSize

        # If Adjacency is less that one, the correlation matrix becomes identity since no point will be adjacet to other in the correlation matrix.
        if Adjacency < 1.0:
            raise ValueError('Adjacency: %0.2f. Correlation matrix will become identity since Kernel length is less that grid size. To increase adjacency, consider decreasing KernelThreshold or increase DecorrelationScale.'%(Adjacency))

    # Disable parallel processing if ray is not installed
    if not RayInstalled:
        RunInParallel = False

    # If matrice are sparse, it is better to generate columns of correlation in parallel
    if (RunInParallel == False) and (UseSparse == True):
        warnings.warn('If matrices are sparse, it is better to generate columns of correlation matrix in parallel. Set "RunInParallel" to True.')

    if RunInParallel:

        try:
            # Get number of cpus
            NumCPUs = os.cpu_count()

            # Parallelization with ray
            ray.init(num_cpus=NumCPUs,logging_level=logging.FATAL)

            # Parallel section with ray. This just creates process Ids. It does not do computation
            Process_Ids = [ComputeCorrelationForAProcess.remote(DecorrelationScale,nu,KernelThreshold,x,y,UseSparse,NumCPUs,StartIndex) for StartIndex in range(NumCPUs)]

            # Do the parallel computations
            K_List = ray.get(Process_Ids)

            # Initialize an empty correlation
            if UseSparse:
                K = scipy.sparse.csc_matrix((n,n))
            else:
                K = numpy.zeros((n,n),dtype=float)

            # Sum K in each process to complete the correlation
            for K_InList in K_List:
                K = K + K_InList

            ray.shutdown()

        except:

            warnings.warn('Ray parallel processing to generate correlation failed. Try with a single process ...')

            # Sometimes Ray's communications fail. Compute correlation withput parallel section
            K = ComputeCorrelationForAProcess(DecorrelationScale,nu,KernelThreshold,x,y,UseSparse,None,None)

    else:

        # Compute correlation withput parallel section
        K = ComputeCorrelationForAProcess(DecorrelationScale,nu,KernelThreshold,x,y,UseSparse,None,None)

    # Fill lower left elements using symmetry of matrix
    K = K + K.T

    # Density
    if UseSparse == True:

        Density = K.nnz / numpy.prod(K.shape)
        print('Using sparse correlation matrix with kernel threshold: %0.4f and sparsity: %0.4f'%(KernelThreshold,Density))

    return K
