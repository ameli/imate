#! /usr/bin/env python

"""
Notes:

    To find the CPU rum time of minimizing GCV, I used
    - SHGO global optimization method
    - Bound of search for lambda is 10^{-16} to 10^{+16}
    - Trace is computed with either:
      1. Hutchinson method
      2. Cholesky factorization and without computing Inverse
         (that is,in EstimationTrace.py, set UseInverseMatrix = False)

      But, the trace fails to be computed with Lanczos Quadrature method, since for very
      small lambda, the tri-diagnalization fails.

      To plot GCV and trace estimations, compute trace with Cholesky, and WITH matrix
      inverse. That is, in EstimateTrace.py, set UseInverseMatrix=True.

      NOTE:
      If you want to measure the elased time of minimizing GCV, do the followings:

      1. in the Differential Evolution method, set worker=1 (NOT -1).
      2. In the import os, uncomment all the os.environ(...) variables.

      Otherwise, all measured elapsed times will be wrong due to the parallel processing.
      The only way that seems to measure elased time of multicore process properly is to 
      restrict python to use multi-cores.

      Note:
      In rational polynomial method, p in the code means the number of interpolant points.
      However, in the paper, p means the degree of the rational polynomial.
      In the code we have p = 2, and p = 4, (num points) but in the plots in this script, they are
      labeled as p = 1 and p = 2 (degree of the rational polynomial).
"""

# =======
# Imports
# =======

import numpy
import scipy
from scipy import sparse
from scipy import optimize
from functools import partial
import time
import pickle
import os

# Classes
from TraceEstimation import TraceEstimation
from PlotSettings import *

# =============
# Generate Data
# =============

def GenerateData(n,m):

    numpy.random.seed(31)

    if n > m:
        u = numpy.random.randn(n)
        U = numpy.eye(n) - 2.0 * numpy.outer(u,u) / numpy.linalg.norm(u)**2
        U = U[:,:m]
    else:
        u = numpy.random.randn(m)
        U = numpy.eye(m) - 2.0 * numpy.outer(u,u) / numpy.linalg.norm(u)**2
        U = U[:n,:]

    v = numpy.random.randn(m)
    V = numpy.eye(m) - 2.0 * numpy.outer(v,v.T) / numpy.linalg.norm(v)**2
    
    # sigma = numpy.exp(-20*(numpy.arange(m)/m)**(0.5))
    sigma = numpy.exp(-40*(numpy.arange(m)/m)**(0.75))  # good for n,m = 1000,500
    # sigma = numpy.exp(-20*(numpy.arange(m)/m)**(0.5)) * numpy.sqrt(n/1000) * 1e2
    # sigma = numpy.exp(-10*(numpy.arange(m)/m)**(0.2)) * numpy.sqrt(n/1000) * 1e2
    # sigma = numpy.sqrt(sigma)

    Sigma = numpy.diag(sigma)

    X = numpy.matmul(U,numpy.matmul(Sigma,V.T))
    
    # beta = numpy.random.randn(m)
    beta = numpy.random.randn(m) / numpy.sqrt(n/1000)

    NoiseLevel = 4e-1
    epsilon = NoiseLevel * numpy.random.randn(n)

    # Data
    z = numpy.dot(X,beta) + epsilon

    return X,z

# ============
# Time Counter
# ============

class TimeCounterClass(object):
    """
    This class is used to measure the elapsed time of computing trace only for the
    exact (non-interpolation) method.

    In the interpolation method, the trace is "pre-computed" (in TraceEstimationUtilities), so we can easily find
    how much time did it take to compute trace in the pre-computation.

    However, in the direct method of minimizing GCV, we can only measue the total time of minimization process.
    To measure only the elapsed time of computing trace (which is a part of computing GCV) we pass an object
    of this class to accumulatively measure the elapsed time of a portion related to computing trace.
    """
    def __init__(self):
        self.ElapsedTime = 0
    def Add(self,Time):
        self.ElapsedTime = self.ElapsedTime + Time
    def Reset(self):
        self.ElapsedTime = 0

# ============================
# Generalized Cross Validation
# ============================

def GeneralizedCrossValidation(X,K,z,TraceEstimationUtilities,eta0,TimeCounter,UseLogLambda,Lambda):
    """
    Computes the CGV function, V(Lambda)

    X size is (n,m)
    K = X.T * X, which is (m,m) size

    Reference: https://www.jstor.org/stable/pdf/1390722.pdf?refreqid=excelsior%3Adf48321fdd477aab0ea5dbf2542df01d
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
        if TraceEstimationUtilities is not None:
            Trace = n - m + mu * TraceEstimation.EstimateTrace(TraceEstimationUtilities,mu-eta0)
        else:
            time0 = time.process_time()
            # Ainv = numpy.linalg.inv(A)
            # Trace = n - m + mu * numpy.trace(Ainv)
            Trace = n - m + mu * TraceEstimation.ComputeTraceOfInverse(A)
            time1 = time.process_time()
            if TimeCounter is not None:
                TimeCounter.Add(time1 - time0)
    else:
        if TraceEstimationUtilities is not None:
            Trace = mu * TraceEstimation.EstimateTrace(TraceEstimationUtilities,mu-eta0)
        else:
            time0 = time.process_time()
            In = numpy.eye(n)
            B = X.dot(X.T) + mu * In
            # Binv = numpy.linalg.inv(B)
            # Trace = mu * numpy.trace(Binv)
            Trace = mu * TraceEstimation.ComputeTraceOfInverse(B)
            time1 = time.process_time()
            if TimeCounter is not None:
                TimeCounter.Add(time1 - time0)

    Denominator = (Trace / n)**2

    GCV = Numerator / Denominator

    return GCV

# ============
# Minimize GCV
# ============

def MinimizeGCV(X,K,z,TraceEstimationUtilities,eta0,LambdaBounds,InitialElapsedTime,TimeCounter):
    """
    In this function we use lambda in logarithmic scale.
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
            X,K,z,TraceEstimationUtilities,eta0,TimeCounter,UseLogLambda)

    # Optimization methods
    time0 = time.process_time()
    # time0 = time.time()

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

    time1 = time.process_time()
    # time1 = time.time()
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

def PlotGeneralizedCrossValidation(Data):
    """
    Plots GCV for a range of Lambda.

    Data is a list of dictionaries, Data[0], Data[1], ...
    Each dictionary Data[i] has the fields

        'Lambda': x axis, this is the same for all dictionaries in the list.
        'GCV': y axis data. These are 
        'Label': the label of the data GCV in the plot.
    """

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

    # Save plots
    plt.tight_layout()
    SaveDir = './doc/images/'
    PlotFilename = 'GeneralizedCrossValidation'
    SaveFilename_PDF = SaveDir + PlotFilename + '.pdf'
    SaveFilename_SVG = SaveDir + PlotFilename + '.svg'
    plt.savefig(SaveFilename_PDF,transparent=True,bbox_inches='tight')
    plt.savefig(SaveFilename_SVG,transparent=True,bbox_inches='tight')
    print('Plot saved to %s.'%(SaveFilename_PDF))
    print('Plot saved to %s.'%(SaveFilename_SVG))
    # plt.show()

# =====================================
# Plot Trace Estimate - Ill Conditioned
# =====================================

def PlotTraceEstimate_IllConditioned(TraceEstimationUtilitiesList,K):
    """
    Plots the curve of trace of Kn inverse versus eta.
    """

    # If not a list, embed the object into a list
    if not isinstance(TraceEstimationUtilitiesList,list):
        TraceEstimationUtilitiesList = [TraceEstimationUtilitiesList]

    # Determine to use sparse
    UseSparse = False
    if scipy.sparse.isspmatrix(K):
        UseSparse = True

    # Extract parameters
    T0   = TraceEstimationUtilitiesList[0]['AuxilliaryEstimationMethodUtilities']['T0']
    n    = TraceEstimationUtilitiesList[0]['AuxilliaryEstimationMethodUtilities']['n']
    T1   = TraceEstimationUtilitiesList[0]['AuxilliaryEstimationMethodUtilities']['T1']
    eta1 = TraceEstimationUtilitiesList[0]['AuxilliaryEstimationMethodUtilities']['eta1']

    NumberOfEstimates = len(TraceEstimationUtilitiesList)

    eta = numpy.r_[-numpy.logspace(-9,-3.0001,500)[::-1],0,numpy.logspace(-9,3,500)]
    ZeroIndex = numpy.argmin(numpy.abs(eta))
    trace_upperbound = numpy.zeros(eta.size)
    trace_lowerbound = numpy.zeros(eta.size)
    trace_exact = numpy.zeros(eta.size)
    trace_estimate = numpy.zeros((NumberOfEstimates,eta.size))

    for i in range(eta.size):
        if eta[i] >= 0.0:
            trace_upperbound[i] = 1.0/(1.0/T0 + eta[i]/n)
        else:
            trace_upperbound[i] = 1.0/(1.0/T0 - eta[i]/n)
        trace_lowerbound[i] = n/(1.0+eta[i])

        # Kn
        if UseSparse:
            I = scipy.sparse.eye(K.shape[0],format='csc')
        else:
            I = numpy.eye(K.shape[0])
        Kn = K + eta[i]*I
        trace_exact[i] = TraceEstimation.ComputeTraceOfInverse(Kn)

        for j in range(NumberOfEstimates):
            trace_estimate[j,i] = TraceEstimation.EstimateTrace(TraceEstimationUtilitiesList[j],eta[i])

    # Tau
    tau_upperbound = trace_upperbound / n
    tau_lowerbound = trace_lowerbound / n
    tau_exact = trace_exact / n
    tau_estimate = trace_estimate / n

    # Plots trace
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(12,6))
    ax[0].plot(eta,tau_exact,color='black',label='Exact')
    ax[0].plot(eta[ZeroIndex:],tau_upperbound[ZeroIndex:],'--',color='black',label=r'Upper bound (at $t \geq 0$)')
    ax[0].plot(eta[:ZeroIndex],tau_upperbound[:ZeroIndex],'-.',color='black',label=r'Lower bound (at $t < 0$)')
    # ax[0].plot(eta,tau_lowerbound,'-.',color='black',label='Lower bound')

    ColorsList =["#d62728",
            "#2ca02c",
            "#bcbd22",
            "#ff7f0e",
            "#1f77b4",
            "#9467bd",
            "#8c564b",
            "#17becf",
            "#7f7f7f",
            "#e377c2"]

    for j in reversed(range(NumberOfEstimates)):
        p = TraceEstimationUtilitiesList[j]['AuxilliaryEstimationMethodUtilities']['p']
        q = ax[0].plot(eta,tau_estimate[j,:],label=r'Interpolation, $p=%d$'%(p//2),color=ColorsList[j])
        if j == 0:
            q[0].set_zorder(20)

    ax[0].set_xlim([eta[0],eta[-1]])
    ax[0].set_ylim([1e-3,1e4])
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$\tau(t)$')
    ax[0].set_title(r'(a) Exact, interpolation, and bounds of $\tau(t)$')
    ax[0].grid(True)
    ax[0].legend(fontsize='x-small',loc='lower left')
    ax[0].set_xscale('symlog',linthreshx=1e-8)
    ax[0].set_yscale('log')
    ax[0].set_xticks(numpy.r_[-10**numpy.arange(-3,-7,-3,dtype=float),0,10**numpy.arange(-6,4,3,dtype=float)])
    ax[0].tick_params(axis='x',which='minor',bottom=False)

    # Inset plot
    ax2 = plt.axes([0,0,1,1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax[0],[0.12,0.4,0.45,0.35])
    ax2.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on ax1 and draw lines
    # in grey linking the two axes.

    # Avoid inset mark lines interset the inset axes itself by setting its anchor
    InsetColor = 'oldlace'
    mark_inset(ax[0],ax2,loc1=1,loc2=4,facecolor=InsetColor,edgecolor='0.5')
    ax2.plot(eta,tau_exact,color='black',label='Exact')
    ax2.plot(eta[ZeroIndex:],tau_upperbound[ZeroIndex:],'--',color='black',label=r'Upper bound (at $t \geq 0$)')
    ax2.plot(eta[:ZeroIndex],tau_upperbound[:ZeroIndex],'-.',color='black',label=r'Lower bound (at $t < 0$)')
    for j in reversed(range(NumberOfEstimates)):
        ax2.plot(eta,tau_estimate[j,:],color=ColorsList[j])
    # ax2.set_xlim([1e-3,1.4e-3])
    # ax2.set_ylim(400,500)
    # ax2.set_xticks([1e-3,1.4e-3])
    # ax2.set_yticks([400,500])
    ax2.set_xlim([1e-2,1.15e-2])
    ax2.set_ylim(80,90)
    ax2.set_xticks([1e-2,1.15e-2])
    ax2.set_yticks([80,90])
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xticklabels(['0.01','0.0115'])
    # ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax2.set_facecolor(InsetColor)
    # plt.setp(ax2.get_yticklabels(),backgroundcolor='white')

    # ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    # ax2.grid(True,axis='y')

    # Plot errors
    # ax[1].semilogx(eta,tau_upperbound-tau_exact,'--',color='black',label='Upper bound')  # Absolute error
    ax[1].semilogx(eta[ZeroIndex:],100*(tau_upperbound[ZeroIndex:]/tau_exact[ZeroIndex:]-1),'--',color='black',label=r'Upper bound (at $t \geq 0$)',zorder=15)  # Relative error
    ax[1].semilogx(eta[:ZeroIndex],100*(tau_upperbound[:ZeroIndex]/tau_exact[:ZeroIndex]-1),'-.',color='black',label=r'Lower bound (at $t < 0$)',zorder=15)  # Relative error
    for j in reversed(range(NumberOfEstimates)):
        p = TraceEstimationUtilitiesList[j]['AuxilliaryEstimationMethodUtilities']['p']
        # q = ax[1].semilogx(eta,tau_estimate[j,:]-tau_exact,label=r'Estimation, $p=%d$'%(p),color=ColorsList[j])  # Absolute error
        q = ax[1].semilogx(eta,100*(tau_estimate[j,:]/tau_exact-1),label=r'Interpolation, $p=%d$'%(p//2),color=ColorsList[j])       # Relative error
        if j == 0:
            q[0].set_zorder(20)
    # ax[1].semilogx(eta,tau_estimate_alt-tau_exact,label=r'Alt. estimation',zorder=-20)   # Absolute error
    # ax[1].semilogx(eta,tau_estimate_alt/tau_exact-1,label=r'Alt. estimation',zorder=-20)   # Relative error
    ax[1].set_xlim([eta[0],eta[-1]])
    ax[1].set_ylim([-0.5,2.5])
    ax[1].set_yticks(numpy.arange(-0.5,2.6,0.5))
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$\tau_{\mathrm{approx}}(t)/\tau_{\mathrm{exact}}(t) - 1$')
    ax[1].set_title(r'(b) Relative error of estimation of $\tau(t)$')
    ax[1].grid(True)
    ax[1].legend(fontsize='x-small',loc='upper left')
    ax[1].set_xscale('symlog',linthreshx=1e-8)
    ax[1].set_yscale('linear')
    ax[1].set_xticks(numpy.r_[-10**numpy.arange(-3,-7,-3,dtype=float),0,10**numpy.arange(-6,4,3,dtype=float)])
    ax[1].tick_params(axis='x',which='minor',bottom=False)

    ax[1].yaxis.set_major_formatter(PercentFormatter(decimals=1))

    # Save plots
    plt.tight_layout()
    SaveDir = './doc/images/'
    SaveFilename = 'EstimateTrace-IllConditioned'
    SaveFilename_PDF = SaveDir + SaveFilename + '.pdf'
    SaveFilename_SVG = SaveDir + SaveFilename + '.svg'
    # plt.savefig(SaveFilename_PDF,transparent=True,bbox_inches='tight')
    plt.savefig(SaveFilename_PDF,bbox_inches='tight')
    plt.savefig(SaveFilename_SVG,bbox_inches='tight')
    print('Plot saved to %s.'%(SaveFilename_PDF))
    print('Plot saved to %s.'%(SaveFilename_SVG))

# ================
# Compute New Data
# ================

def ComputeNewData(ResultsFilename):

    # n = 10000
    # m = 5000
    n = 1000
    m = 500
    X,z = GenerateData(n,m)

    # K is used to estimate trace
    # eta0 = 2e-4
    # eta0 = 4e-4
    eta0 = 1e-3
    if n > m:
        K0 = X.T.dot(X)
        K = K0 + eta0 * numpy.eye(m,m)

        Cond_K0 = numpy.linalg.cond(K0)
        Cond_K = numpy.linalg.cond(K)
        print('Cond K0: %0.2e, Cond K: %0.2e'%(Cond_K0,Cond_K))

    else:
        K0 = X.dot(X.T)
        K = K0 + eta0 * numpy.eye(n,n)

        Cond_K0 = numpy.linalg.cond(K0)
        Cond_K = numpy.linalg.cond(K)
        print('Cond K0: %0.2e, Cond K: %0.2e'%(Cond_K0,Cond_K))

    # ComputeAuxilliaryMethod = True
    ComputeAuxilliaryMethod = False
    UseEigenvaluesMethod = False
    # TraceEstimationMethod = 'OrthogonalFunctionsMethod2'
    TraceEstimationMethod = 'RationalPolynomialMethod'
    FunctionType = None

    # Interpolation with 4 interpolant points
    time0 = time.process_time()
    TraceEstimationUtilities_1 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,TraceEstimationMethod,FunctionType,[1e-3,1e-2,1e-1,1],ComputeAuxilliaryMethod)
    time1 = time.process_time()
    InitialElapsedTime1 = time1 - time0

    # Interpolation with 2 interpolant points
    time2 = time.process_time()
    TraceEstimationUtilities_2 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,TraceEstimationMethod,FunctionType,[1e-3,1e-1],ComputeAuxilliaryMethod)
    time3 = time.process_time()
    InitialElapsedTime2 = time3 - time2

    TraceEstimationUtilitiesList = [TraceEstimationUtilities_1,TraceEstimationUtilities_2]

    # Minimize GCV
    # LambdaBounds = (1e-4,1e1)
    LambdaBounds = (1e-16,1e16)
    TimeCounter = TimeCounterClass()
    MinimizationResult1 = MinimizeGCV(X,K,z,None,eta0,LambdaBounds,0,TimeCounter)
    MinimizationResult2 = MinimizeGCV(X,K,z,TraceEstimationUtilities_1,eta0,LambdaBounds,InitialElapsedTime1,TimeCounter)
    MinimizationResult3 = MinimizeGCV(X,K,z,TraceEstimationUtilities_2,eta0,LambdaBounds,InitialElapsedTime2,TimeCounter)

    print('Time to compute trace only:')
    print('Exact: %f'%(TimeCounter.ElapsedTime))
    print('Interp 4 points: %f'%InitialElapsedTime1)
    print('Interp 2 points: %f'%InitialElapsedTime2)
    print('')

    Lambda = numpy.logspace(-7,1,500)
    GCV1 = numpy.empty(Lambda.size)
    GCV2 = numpy.empty(Lambda.size)
    GCV3 = numpy.empty(Lambda.size)
    UseLogLambda = False
    for i in range(Lambda.size):
        GCV1[i] = GeneralizedCrossValidation(X,K,z,None,eta0,None,UseLogLambda,Lambda[i])
        GCV2[i] = GeneralizedCrossValidation(X,K,z,TraceEstimationUtilities_2,eta0,None,UseLogLambda,Lambda[i])
        GCV3[i] = GeneralizedCrossValidation(X,K,z,TraceEstimationUtilities_1,eta0,None,UseLogLambda,Lambda[i])

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

    # Results dictionary
    Results = \
    {
        'K': K,
        'TraceEstimationUtilitiesList': TraceEstimationUtilitiesList,
        'PlotData': PlotData,
    }

    # Save results
    with open(ResultsFilename,'wb') as handle:
        pickle.dump(Results,handle)
    print('Results saved to %s.'%ResultsFilename)

    return Results

# ========================================
# Restrict Computation To Single Processor
# ========================================

def RestrictComputationToSingleProcessor():
    """
    To measure the CPU time of all processors we use time.process_time() which takes into acount 
    of elapsed time of all running threads. However, it seems when I use scipy.optimize.differential_evolution
    method with either worker=-1 or worker=1, the CPU time is not measured properly.

    After all failed trials, the only solution that measures time (for only scipy.optimize.differential_evolution) 
    is to restrict the whole python script to use a single code. This function does that.

    Note, other scipy.optimzie methods (like shgo) do not have this issue. That means, you can still run the code
    in parallel and the time.process_time() measures the CPU time of all cores properly.
    """

    # Uncomment lines below if measureing elapsed time. These will restrict python to only use one processing thread.
    os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

# ====
# Main
# ====

if __name__ == "__main__":


    # When measuring elapsed time, restrict number of processors to a single core only to measure time properly
    RestrictComputationToSingleProcessor()

    UseSavedResults = True
    ResultsFilename = './doc/data/GeneralizedCrossValidation.pickle'

    if UseSavedResults == False:

        # Compute new data
        Results = ComputeNewData(ResultsFilename)

    else:

        # Load previously computed data
        with open(ResultsFilename,'rb') as handle:
            Results = pickle.load(handle)

    # Extract variables
    K = Results['K']
    TraceEstimationUtilitiesList = Results['TraceEstimationUtilitiesList']
    PlotData = Results['PlotData']
        
    # Plots
    PlotGeneralizedCrossValidation(PlotData)
    PlotTraceEstimate_IllConditioned(TraceEstimationUtilitiesList,K)
