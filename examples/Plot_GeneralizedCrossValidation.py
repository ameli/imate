#! /usr/bin/env python

# =======
# Imports
# =======

import sys
import numpy
import scipy
from scipy import sparse
from scipy import optimize
from functools import partial
import time
import pickle

# Package Modules
from Utilities.PlotUtilities import *
from Utilities.PlotUtilities import LoadPlotSettings
from Utilities.PlotUtilities import SavePlot
from Utilities.ProcessingTimeUtilities import RestrictComputationToSingleProcessor
from Utilities.ProcessingTimeUtilities import TimeCounterClass
from Utilities.ProcessingTimeUtilities import ProcessTime
from Utilities.DataUtilities import GenerateBasisFunctions
from Utilities.DataUtilities import GenerateNoisyData
from Utilities.DataUtilities import GenerateMatrix
from TraceInv import InterpolateTraceOfInverse
from TraceInv import ComputeTraceOfInverse

# ============================
# Generalized Cross Validation
# ============================

def GeneralizedCrossValidation(X,K,z,TI,Shift,TimeCounter,UseLogLambda,Lambda):
    """
    Computes the CGV function :math:`V(\\theta)`.

    * ``X`` shape is ``(n,m)``.
    * :math:`K = X^{\intercal} X`, which is ``(m,m)`` shape.

    **Reference:** 
    
        Golub, G., & Von Matt, U. (1997). 
        Generalized Cross-Validation for Large-Scale Problems. 
        Journal of Computational and Graphical Statistics, 6(1), 1-34.
        doi: `10.2307-1390722 <https://www.jstor.org/stable/pdf/1390722.pdf?refreqid=excelsior%3Adf48321fdd477aab0ea5dbf2542df01d>`_

        :param X: Matrix if basis functions of the shape ``(n,m)`` with ``m`` basis functions over ``n`` spatial points.
        :type X: numpy.ndarray

        :param K: Correlation matrix of the shape ``(n,n)``.
        :type: numpy.ndarray or scipy.sparse.csc_matrix

        :param z: Column vector of data at ``n`` points.
        :type: numpy.array

        :param TI: TraceInv interpolating object
        :type TI: TraceInv.InterpolateTraceOfInverse

        :param Shift: Shift for the signular matrix ``K0`` to ``K = K0 + Shift * I``.
        :type Shift: float

        :param TimeCounter: A counter object to store the elasped time and to be read outside of this function.
        :type TimeCounteR: examples.Utilities.TimeCounterClass

        :param UseLog: A flag, if ``True``, it assumes ``Lambda`` is in logarithmic scale.
                If ``False``, then ``Lambda`` is not assumed to be in logarithmic scale.
        :type UseLog: bool

        :param Lambda: Parameter of generalized cross validation.
        :type: float

        The generalized cros-validation (GCV) function is:

        .. math::

            V(\theta) = \frac{\frac{1}{n} \| \mathbf{I} - \mathbf{X} (\mathbf{X}^{\intercal} \mathbf{X} + n \theta \mathbf{I})^{-1} \mathbf{X}^{\intercal} \boldsymbol{z} \|_2^2}{\left( \frac{1}{n} \mathrm{trace}\left( (\mathbf{I} - \mathbf{X}(\mathbf{X}^{\intercal} \mathbf{X} + n \theta \mathbf{I})^{-1})\mathbf{X}^{\intercal} \right) \right)^2}

        In the above equation, the term involving trace is implemented differently depending on wtether :math:`n > m` or :math:`n < m` (see details in [GOLUB-1979]_).

        .. note::

            In this function, we use the variable ``Lambda`` for :math:`\\theta`.


        **References:**

        .. [GOLUB-1979] Golub, G., Heath, M., & Wahba, G. (1979). Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter. Technometrics, 21(2), 215-223. doi: `10.2307/1268518 <https://www.jstor.org/stable/1268518?seq=1>`_ 

    """

    # If lambda is in the logarithm scale, convert it to normal scale
    if UseLogLambda == True:
        Lambda = 10**Lambda

    n,m = X.shape
    mu = n*Lambda

    y1 = X.T.dot(z)
    Im = numpy.eye(m)
    A = X.T.dot(X) + mu * Im

    # Compute numerator
    y2 = numpy.linalg.solve(A,y1)
    y3 = X.dot(y2)
    y = z - y3
    Numerator = numpy.linalg.norm(y)**2 / n

    # Compute denominator
    if n > m:
        if TI is not None:
            # Interpolate trace of inverse
            Trace = n - m + mu * TI.Interpolate(mu-Shift)
        else:
            # Compute the exact value of the trace of inverse
            time0 = ProcessTime()
            # Ainv = numpy.linalg.inv(A)
            # Trace = n - m + mu * numpy.trace(Ainv)
            Trace = n - m + mu * ComputeTraceOfInverse(A)
            time1 = ProcessTime()
            if TimeCounter is not None:
                TimeCounter.Add(time1 - time0)
    else:
        if TI is not None:
            # Interpolate trace of inverse
            Trace = mu * TI.Interpolate(mu-Shift)
        else:
            time0 = ProcessTime()
            In = numpy.eye(n)
            B = X.dot(X.T) + mu * In
            # Binv = numpy.linalg.inv(B)
            # Trace = mu * numpy.trace(Binv)
            Trace = mu * ComputeTraceOfInverse(B)
            time1 = ProcessTime()
            if TimeCounter is not None:
                TimeCounter.Add(time1 - time0)

    Denominator = (Trace / n)**2

    GCV = Numerator / Denominator

    return GCV

# ============
# Minimize GCV
# ============

def MinimizeGCV(X,K,z,TI,Shift,LambdaBounds,InitialElapsedTime,TimeCounter):
    """
    Finds the parameter ``lambda`` such that GCV is minimized.

    In this function, ``lambda`` in logarithmic scale.
    """

    print('\nMinimize GCV ...')

    # Use lambda in log scale
    UseLogLambda = True
    Bounds = [(numpy.log10(LambdaBounds[0]),numpy.log10(LambdaBounds[1]))]
    Tolerance = 1e-4
    GuessLogLambda = -4
    
    # Partial function to minimize
    GCV_PartialFunction = partial( \
            GeneralizedCrossValidation, \
            X,K,z,TI,Shift,TimeCounter,UseLogLambda)

    # Optimization methods
    time0 = ProcessTime()

    # Local optimization method (use for both direct and presented method)
    # Method = 'Nelder-Mead'
    # Res = scipy.optimize.minimize(GCV_PartialFunction,GuessLogLambda,method=Method,tol=Tolerance,
            # options={'maxiter':1000,'xatol':Tolerance,'fatol':Tolerance,'disp':True})
            # callback=MinimizeTerminatorObj.__call__,

    # Global optimization methods (use for direct method)
    numpy.random.seed(31)   # for repeatability of results
    Res = scipy.optimize.differential_evolution(GCV_PartialFunction,Bounds,workers=1,tol=Tolerance,atol=Tolerance,
            updating='deferred',polish=True,strategy='best1bin',popsize=40,maxiter=200) # Works well
    # Res = scipy.optimize.dual_annealing(GCV_PartialFunction,Bounds,maxiter=500)
    # Res = scipy.optimize.shgo(GCV_PartialFunction,Bounds,
            # options={'minimize_every_iter': True,'local_iter': True,'minimizer_kwargs':{'method': 'Nelder-Mead'}})
    # Res = scipy.optimize.basinhopping(GCV_PartialFunction,x0=GuessLogLambda)

    print(Res)

    # Brute Force optimization method (use for direct method)
    # rranges = ((0.1,0.3),(0.5,25))
    # Res = scipy.optimize.brute(GCV_PartialFunction,ranges=rranges,full_output=True,finish=scipy.optimize.fmin,workers=-1,Ns=30)
    # Optimal_DecorrelationScale = Res[0][0]
    # Optimal_nu = Res[0][1]
    # max_lp = -Res[1]
    # Iterations = None
    # Message = "Using bute force"
    # Sucess = True

    time1 = ProcessTime()
    ElapsedTime = InitialElapsedTime + time1 - time0
    print('Elapsed time: %f\n'%ElapsedTime)

    Results = \
    {
        'MinGCV': Res.fun,
        'MinLogLambda': Res.x[0],
        'MinLambda': 10.0**Res.x[0],
        'FunEvaluations': Res.nfev,
        'ElapsedTime': ElapsedTime
    }

    return Results

# =================================
# Plot Generalized Cross Validation
# =================================

def PlotGeneralizedCrossValidation(Data,test):
    """
    Plots GCV for a range of Lambda.

    Data is a list of dictionaries, ``Data[0], Data[1], ...``.
    Each dictionary ``Data[i]`` has the fields:

        * ``'Lambda'``: x axis, this is the same for all dictionaries in the list.
        * ``'GCV'``: y axis data.
        * ``'Label'``: the label of the data GCV in the plot.
    """
    
    # Load plot settings
    LoadPlotSettings()

    # Create a list of one item if Data is not a list.
    if not isinstance(Data,list):
        Data = [Data]

    Lambda = Data[0]['Lambda']
    
    fig,ax = plt.subplots(figsize=(7,4.8))
    ColorsList =["#000000","#2ca02c","#d62728"]

    h_list = []
    for i in range(len(Data)):
        h, = ax.semilogx(Lambda,Data[i]['GCV'],label=Data[i]['Label'],color=ColorsList[i])
        h_list.append(h)
        ax.semilogx(Data[i]['MinimizationResult']['MinLambda'],Data[i]['MinimizationResult']['MinGCV'],'o',color=ColorsList[i],markersize=3)

    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$V(\theta)$')
    ax.set_title('Generalized Cross Validation')
    ax.set_xlim([Lambda[0],Lambda[-1]])

    GCV = Data[0]['GCV']
    ax.set_yticks([numpy.min(GCV),numpy.min(GCV[:GCV.size//5])])
    ax.set_ylim([0.1634,numpy.max(GCV)+0.0001])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    ax.grid(True,axis='y')
    ax.legend(frameon=False,fontsize='small',bbox_to_anchor=(0.59,0.21),loc='lower left')

    plt.tight_layout()

    # Save Plot
    Filename = 'GeneralizedCrossValidation'
    if test:
        Filename = "test_" + Filename
    SavePlot(plt,Filename)

    # If no display backend is enabled, do not plot in the interactive mode
    if (not test) and (matplotlib.get_backend() != 'agg'):
        plt.show()

# ====
# Main
# ====

def main(test=False):
    """
    Run the script by

    ::

        python examples/Plot_GeneralizedCrossValidation.py

    The script generates the figure below and prints the processing times of the computations. 
    See more details in Figure 3 and results of Table 2 of [Ameli-2020]_.

    .. image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/GeneralizedCrossValidation.svg
       :width: 550
       :align: center

    .. note::
        To *plot* GCV and trace estimations, compute trace with ``cholesky``, and *with* matrix
        inverse. That is, set ``UseInverseMatrix=True``.

    .. note::
        To properly *measure the elased-time* of minimizing GCV, do the followings:

        1. in the :func:`MinimizeGCV`, use the *Differential Evolution* method, and  set ``worker=1`` (**NOT** ``-1``).
        2. Definitely call the function :func:`Utilities.ProcessingTimeUtilities.RestrictComputationToSingleProcessor()`
            to disable any multi-core processing. By this, all computations are forced to execute on a single thread.
            Otherwise, all measured elapsed times will be wrong due to the parallel processing.
            The only way that seems to measure elased time of multicore process properly is to 
            prevent python to use multi-cores.
        3. Set thebound of search for ``lambda`` to ``10e-16`` to ``10e+16``.
        4. Trace should be computed by either:
            * Hutchinson method
            * Cholesky factorization and without computing Inverse (set ``UseInverseMatrix=False``).

    .. warning::
        To compute the elapsed-time, do not compute trace with *stochastic Lanczos Quadrature* method, since for very
        small ``lambda``, the tri-diagnalization fails.

    .. note::
        In the *rational polynomial functions * method for interpolation (using ``InterpolationMethod='RPF'``), 
        the variable ``p`` in the code represents the number of interpolant points.
        However, in the paper [Ameli-2020], the variable ``p`` represents the degree of the rational polynomial,
        As such, in the code we have ``p = 2``, and ``p = 4``, (num points) but in the plots in this script, 
        they are labeled as ``p = 1`` and ``p = 2`` (degree of the rational polynomial).

    **References**

    .. [Ameli-2020] Ameli, S., and Shadden. S. C. (2020). Interpolating the Trace of the Inverse of Matrix **A** + t **B**. `arXiv:2009.07385 <https://arxiv.org/abs/2009.07385>`__ [math.NA]
    """

    # When measuring elapsed time, restrict number of processors to a single core only to measure time properly
    RestrictComputationToSingleProcessor()

    # Shift to make singular matrix non-signular
    # Shift = 2e-4
    # Shift = 4e-4
    Shift = 1e-3

    # Generate a nearly singular matrix
    if test:
        n = 100
        m = 50
    else:
        n = 1000
        m = 500
    NoiseLevel = 4e-1
    X = GenerateBasisFunctions(n,m)
    z = GenerateNoisyData(X,NoiseLevel)
    K = GenerateMatrix(n,m,Shift)

    # Interpolatng points
    InterpolantPoints_1 = [1e-3,1e-2,1e-1,1]
    InterpolantPoints_2 = [1e-3,1e-1]

    # Interpolating method
    ComputeOptions = {'ComputeMethd':'cholesky','UseInverseMatrix':True}      # Use this for plotting GCV and traces
    # ComputeOptions = {'ComputeMethd':'cholesky','UseInverseMatrix':False}   # use this to measure elapsed time of optimizing GCV
    # ComputeOptions = {'ComputeMethd':'hutchinson','NumIterations':20}       # Use this to measure elapsed time of optimizing GCV
    InterpolationMethod = 'RPF'

    # Interpolation with 4 interpolant points
    time0 = ProcessTime()
    TI_1 = InterpolateTraceOfInverse(K,InterpolantPoints=InterpolantPoints_1,InterpolationMethod=InterpolationMethod,ComputeOptions=ComputeOptions)
    time1 = ProcessTime()
    InitialElapsedTime1 = time1 - time0

    # Interpolation with 2 interpolant points
    time2 = ProcessTime()
    TI_2 = InterpolateTraceOfInverse(K,InterpolantPoints=InterpolantPoints_2,InterpolationMethod=InterpolationMethod,ComputeOptions=ComputeOptions)
    time3 = ProcessTime()
    InitialElapsedTime2 = time3 - time2

    # List of interpolating objects
    TI = [TI_1,TI_2]

    # Minimize GCV
    # LambdaBounds = (1e-4,1e1)
    LambdaBounds = (1e-16,1e16)
    TimeCounter = TimeCounterClass()
    MinimizationResult1 = MinimizeGCV(X,K,z,None,Shift,LambdaBounds,0,TimeCounter)
    MinimizationResult2 = MinimizeGCV(X,K,z,TI_1,Shift,LambdaBounds,InitialElapsedTime1,TimeCounter)
    MinimizationResult3 = MinimizeGCV(X,K,z,TI_2,Shift,LambdaBounds,InitialElapsedTime2,TimeCounter)

    print('Time to compute trace only:')
    print('Exact: %f'%(TimeCounter.ElapsedTime))
    print('Interp 4 points: %f'%InitialElapsedTime1)
    print('Interp 2 points: %f'%InitialElapsedTime2)
    print('')

    # Compute GCV for a range of Lambda
    if test:
        Lambda_Resolution = 50
    else:
        Lambda_Resolution = 500
    Lambda = numpy.logspace(-7,1,Lambda_Resolution)
    GCV1 = numpy.empty(Lambda.size)
    GCV2 = numpy.empty(Lambda.size)
    GCV3 = numpy.empty(Lambda.size)
    UseLogLambda = False
    for i in range(Lambda.size):
        GCV1[i] = GeneralizedCrossValidation(X,K,z,None,Shift,None,UseLogLambda,Lambda[i])
        GCV2[i] = GeneralizedCrossValidation(X,K,z,TI_2,Shift,None,UseLogLambda,Lambda[i])
        GCV3[i] = GeneralizedCrossValidation(X,K,z,TI_1,Shift,None,UseLogLambda,Lambda[i])

    # Make a dictionary list of data for plots
    PlotData1 = \
    {
        'Lambda': Lambda,
        'GCV': GCV1,
        'Label': 'Exact',
        'MinimizationResult': MinimizationResult1
    }

    PlotData2 = \
    {
        'Lambda': Lambda,
        'GCV': GCV2,
        'Label': r'Interpolation, $p = 1$',
        'MinimizationResult': MinimizationResult3
    }

    PlotData3 = \
    {
        'Lambda': Lambda,
        'GCV': GCV3,
        'Label': r'Interpolation, $p = 2$',
        'MinimizationResult': MinimizationResult2
    }

    PlotData = [PlotData1,PlotData2,PlotData3]

    # Plots
    PlotGeneralizedCrossValidation(PlotData,test)

# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(main())
