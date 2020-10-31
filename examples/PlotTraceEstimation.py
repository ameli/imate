#! /usr/bin/env python

"""
Before running this code, make sure in TraceEstimation.py, the ComputeTraceOfInverse() is set to
Cholsky method, with either UseInverse or without it.
"""

# =======
# Imports
# =======

# Classes
import Data
from LikelihoodEstimation import LikelihoodEstimation
from TraceEstimation import TraceEstimation
from PlotSettings import *

# =============================
# Compute Noise For Single Data
# =============================

def ComputeNoiseForSingleData():
    """
    This function uses three methods
        1. Maximizing log likelihood with parameters sigma and sigma0
        2. Maximizing log likelihood with parameters sigma and eta
        3. Finding zeros of derivative of log likelihood

    This script uses a single data, for which the random noise with a given standard deviation is added to the data once.
    It plots
        1. liklelihood in 3D as function of parameters sigma and eta
        2. Trace estimation using interpolation
        3. Derivative of log likelihood.
    """

    # Generate noisy data
    NumPointsAlongAxis = 50
    NoiseMagnitude = 0.2
    GridOfPoints = True
    x,y,z = Data.GenerateData(NumPointsAlongAxis,NoiseMagnitude,GridOfPoints)

    # Generate Linear Model
    DecorrelationScale = 0.1
    UseSparse = False
    nu = 0.5
    K = Data.GenerateCorrelationMatrix(x,y,z,DecorrelationScale,nu,UseSparse)

    # BasisFunctionsType = 'Polynomial-0'
    # BasisFunctionsType = 'Polynomial-1'
    BasisFunctionsType = 'Polynomial-2'
    # BasisFunctionsType = 'Polynomial-3'
    # BasisFunctionsType = 'Polynomial-4'
    # BasisFunctionsType = 'Polynomial-5'
    # BasisFunctionsType = 'Polynomial-2-Trigonometric-1'
    X = Data.GenerateLinearModelBasisFunctions(x,y,BasisFunctionsType)

    # Trace estimation weights
    UseEigenvaluesMethod = False    # If set to True, it overrides the interpolation estimation methods
    # TraceEstimationMethod = 'NonOrthogonalFunctionsMethod'   # highest condtion number
    # TraceEstimationMethod = 'OrthogonalFunctionsMethod'      # still high condition number
    TraceEstimationMethod = 'OrthogonalFunctionsMethod2'     # best (lowest) condition number
    # TraceEstimationMethod = 'RBFMethod'

    # Precompute trace interpolation function
    ComputeAuxilliaryMethod = True
    TraceEstimationUtilities_1 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,TraceEstimationMethod,None,[1e-4,4e-4,1e-3,1e-2,1e-1,1,1e+1,1e+2,1e+3],ComputeAuxilliaryMethod)
    TraceEstimationUtilities_2 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,TraceEstimationMethod,None,[1e-4,1e-3,1e-2,1e-1,1,1e+1,1e+3])
    TraceEstimationUtilities_3 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,TraceEstimationMethod,None,[1e-3,1e-2,1e-1,1e+1,1e+3])
    TraceEstimationUtilities_4 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,TraceEstimationMethod,None,[1e-3,1e-1,1e+1])
    TraceEstimationUtilities_5 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,TraceEstimationMethod,None,[1e-1])

    # Use rational polynomial method
    # TraceEstimationUtilities_6 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,'RationalPolynomialMethod',None,[1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3])

    # Use radial basis functions (RBF) instead
    # TraceEstimationUtilities_6 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,TraceEstimationMethod,1,[1e-4,1e-3,1e-2,1e-1,1,1e+1,1e+2,1e+3])
    # TraceEstimationUtilities_7 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,TraceEstimationMethod,2,[1e-2,1e-1,1,1e+1,1e+2])
    # TraceEstimationUtilities_8 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,TraceEstimationMethod,3,[1e-2,1e-1,1,1e+1,1e+2])

    TraceEstimationUtilitiesList = [ \
            TraceEstimationUtilities_1,
            TraceEstimationUtilities_2,
            TraceEstimationUtilities_3,
            TraceEstimationUtilities_4,
            TraceEstimationUtilities_5]

    # Plot Trace Estimate
    TraceEstimation.PlotTraceEstimate(TraceEstimationUtilitiesList,K)

    # ====
    # Plot
    # ====

    def Plot(self):
        """
        Plots the curve of trace of An inverse versus eta (we use t instead of eta in the plots).
        """

        # If not a list, embed the object into a list
        if not isinstance(TraceEstimationUtilitiesList,list):
            TraceEstimationUtilitiesList = [TraceEstimationUtilitiesList]

        # Determine to use sparse
        UseSparse = False
        if scipy.sparse.isspmatrix(A):
            UseSparse = True

        # Extract parameters
        T0   = TraceEstimationUtilitiesList[0]['AuxilliaryEstimationMethodUtilities']['T0']
        n    = TraceEstimationUtilitiesList[0]['AuxilliaryEstimationMethodUtilities']['n']
        T1   = TraceEstimationUtilitiesList[0]['AuxilliaryEstimationMethodUtilities']['T1']
        eta1 = TraceEstimationUtilitiesList[0]['AuxilliaryEstimationMethodUtilities']['eta1']

        NumberOfEstimates = len(TraceEstimationUtilitiesList)

        eta = numpy.logspace(-4,3,100)
        trace_upperbound = numpy.zeros(eta.size)
        trace_lowerbound = numpy.zeros(eta.size)
        trace_exact = numpy.zeros(eta.size)
        trace_estimate = numpy.zeros((NumberOfEstimates,eta.size))
        trace_estimate_alt = numpy.zeros(eta.size)

        for i in range(eta.size):
            trace_upperbound[i] = 1.0/(1.0/T0 + eta[i]/n)
            trace_lowerbound[i] = n/(1.0+eta[i])

            # An
            if UseSparse:
                I = scipy.sparse.eye(A.shape[0],format='csc')
            else:
                I = numpy.eye(A.shape[0])
            An = A + eta[i]*I
            trace_exact[i] = InterpolateTraceOfInverse.ComputeTraceOfInverse(An)
            trace_estimate_alt[i] = 1.0 / (numpy.sqrt((eta[i]/n)**2 + ((1.0/T1)**2 - (1.0/T0)**2 - (eta1/n)**2)*(eta[i]/eta1) + (1/T0)**2));

            for j in range(NumberOfEstimates):
                trace_estimate[j,i] = InterpolateTraceOfInverse.EstimateTrace(TraceEstimationUtilitiesList[j],eta[i])

        # Tau
        tau_upperbound = trace_upperbound / n
        tau_lowerbound = trace_lowerbound / n
        tau_exact = trace_exact / n
        tau_estimate = trace_estimate / n
        tau_estimate_alt = trace_estimate_alt / n

        # Plots trace
        fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(12,6))
        ax[0].loglog(eta,tau_exact,color='black',label='Exact')
        ax[0].loglog(eta,tau_upperbound,'--',color='black',label='Upper bound')
        ax[0].loglog(eta,tau_lowerbound,'-.',color='black',label='Lower bound')

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
            q = ax[0].loglog(eta,tau_estimate[j,:],label=r'Interpolation, $p=%d$'%(p),color=ColorsList[j])
            if j == 0:
                q[0].set_zorder(20)

        # ax[0].loglog(eta,tau_estimate_alt,label=r'Alt. estimation',zorder=-20)

        ax[0].set_xlim([eta[0],eta[-1]])
        ax[0].set_ylim([1e-3,1e1])
        ax[0].set_xlabel(r'$t$')
        ax[0].set_ylabel(r'$\tau(t)$')
        ax[0].set_title(r'(a) Exact, interpolation, and bounds of $\tau(t)$')
        ax[0].grid(True)
        ax[0].legend(fontsize='x-small',loc='upper right')

        # Inset plot
        ax2 = plt.axes([0,0,1,1])
        # Manually set the position and relative size of the inset axes within ax1
        ip = InsetPosition(ax[0],[0.14,0.1,0.5,0.4])
        ax2.set_axes_locator(ip)
        # Mark the region corresponding to the inset axes on ax1 and draw lines
        # in grey linking the two axes.

        # Avoid inset mark lines interset the inset axes itself by setting its anchor
        InsetColor = 'oldlace'
        mark_inset(ax[0],ax2,loc1=1,loc2=2,facecolor=InsetColor,edgecolor='0.5')
        ax2.semilogx(eta,tau_exact,color='black',label='Exact')
        ax2.semilogx(eta,tau_upperbound,'--',color='black',label='Upper bound')
        for j in reversed(range(NumberOfEstimates)):
            ax2.semilogx(eta,tau_estimate[j,:],color=ColorsList[j])
        # ax2.semilogx(eta,tau_estimate_alt,label=r'Alt. estimation',zorder=-1)
        ax2.set_xlim([1e-2,1e-1])
        # ax2.set_xlim([0.35,0.4])
        # ax2.set_ylim(2.5,4)
        ax2.set_ylim(4,6)
        # ax2.set_ylim(1.4,1.6)
        # ax2.set_yticks([2.5,3,3.5,4])
        ax2.set_yticks([4,5,6])
        ax2.xaxis.set_minor_formatter(NullFormatter())
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.set_facecolor(InsetColor)
        # plt.setp(ax2.get_yticklabels(),backgroundcolor='white')

        # ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        # ax2.grid(True,axis='y')

        # Plot errors
        # ax[1].semilogx(eta,tau_upperbound-tau_exact,'--',color='black',label='Upper bound')  # Absolute error
        ax[1].semilogx(eta,100*(tau_upperbound/tau_exact-1),'--',color='black',label='Upper bound',zorder=15)  # Relative error
        for j in reversed(range(NumberOfEstimates)):
            p = TraceEstimationUtilitiesList[j]['AuxilliaryEstimationMethodUtilities']['p']
            # q = ax[1].semilogx(eta,tau_estimate[j,:]-tau_exact,label=r'Estimation, $p=%d$'%(p),color=ColorsList[j])  # Absolute error
            q = ax[1].semilogx(eta,100*(tau_estimate[j,:]/tau_exact-1),label=r'Interpolation, $p=%d$'%(p),color=ColorsList[j])       # Relative error
            if j == 0:
                q[0].set_zorder(20)
        # ax[1].semilogx(eta,tau_estimate_alt-tau_exact,label=r'Alt. estimation',zorder=-20)   # Absolute error
        # ax[1].semilogx(eta,tau_estimate_alt/tau_exact-1,label=r'Alt. estimation',zorder=-20)   # Relative error
        ax[1].set_xlim([eta[0],eta[-1]])
        ax[1].set_yticks(numpy.arange(-0.03,0.13,0.03)*100)
        ax[1].set_ylim([-3,12])
        ax[1].set_xlabel(r'$t$')
        ax[1].set_ylabel(r'$\tau_{\mathrm{approx}}(t)/\tau_{\mathrm{exact}}(t) - 1$')
        ax[1].set_title(r'(b) Relative error of interpolation of $\tau(t)$')
        ax[1].grid(True)
        ax[1].legend(fontsize='x-small')

        ax[1].yaxis.set_major_formatter(PercentFormatter(decimals=0))

        # Save plots
        plt.tight_layout()
        SaveDir = './doc/images/'
        SaveFilename = 'EstimateTrace'
        SaveFilename_PDF = SaveDir + SaveFilename + '.pdf'
        SaveFilename_SVG = SaveDir + SaveFilename + '.svg'
        # plt.savefig(SaveFilename_PDF,transparent=True,bbox_inches='tight')
        plt.savefig(SaveFilename_PDF,bbox_inches='tight')
        plt.savefig(SaveFilename_SVG,bbox_inches='tight')
        print('Plot saved to %s.'%(SaveFilename_PDF))
        print('Plot saved to %s.'%(SaveFilename_SVG))

        # plt.show()

# ====
# Main
# ====

if __name__ == "__main__":

    ComputeNoiseForSingleData()
