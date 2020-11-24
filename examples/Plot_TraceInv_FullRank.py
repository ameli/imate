#! /usr/bin/env python

# =======
# Imports
# =======

import os
import sys
import numpy

# Package modules
from TraceInv import GenerateMatrix
from TraceInv import InterpolateTraceOfInverse
from Utilities.PlotUtilities import *
from Utilities.PlotUtilities import LoadPlotSettings
from Utilities.PlotUtilities import SavePlot

# ====
# Plot
# ====

def Plot(TI,test):
    """
    Plots the curve of trace of An inverse versus eta (we use t instead of eta in the plots).
    """

    print('Plotting ... (may take a few minutes!)')

    # Load plot settings
    LoadPlotSettings()

    # If not a list, embed the object into a list
    if not isinstance(TI,list):
        TI = [TI]

    NumberOfPlots = len(TI)

    # Range to plot
    if test:
        eta_Resolution = 20
    else:
        eta_Resolution = 100
    eta = numpy.logspace(-4,3,eta_Resolution)

    # Functions
    trace_exact = TI[0].Compute(eta)
    trace_upperbound = TI[0].UpperBound(eta)
    trace_lowerbound = TI[0].LowerBound(eta)
    trace_estimate = numpy.zeros((NumberOfPlots,eta.size))
    for j in range(NumberOfPlots):
        trace_estimate[j,:] = TI[j].Interpolate(eta)

    # Tau
    n = TI[0].n
    tau_exact = trace_exact / n
    tau_upperbound = trace_upperbound / n
    tau_lowerbound = trace_lowerbound / n
    tau_estimate = trace_estimate / n

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

    for j in reversed(range(NumberOfPlots)):
        p = TI[j].p
        q = ax[0].loglog(eta,tau_estimate[j,:],label=r'Interpolation, $p=%d$'%(p),color=ColorsList[j])
        if j == 0:
            q[0].set_zorder(20)

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
    for j in reversed(range(NumberOfPlots)):
        ax2.semilogx(eta,tau_estimate[j,:],color=ColorsList[j])
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
    for j in reversed(range(NumberOfPlots)):
        p = TI[j].p
        # q = ax[1].semilogx(eta,tau_estimate[j,:]-tau_exact,label=r'Estimation, $p=%d$'%(p),color=ColorsList[j])  # Absolute error
        q = ax[1].semilogx(eta,100*(tau_estimate[j,:]/tau_exact-1),label=r'Interpolation, $p=%d$'%(p),color=ColorsList[j])       # Relative error
        if j == 0:
            q[0].set_zorder(20)
    ax[1].set_xlim([eta[0],eta[-1]])
    ax[1].set_yticks(numpy.arange(-0.03,0.13,0.03)*100)
    ax[1].set_ylim([-3,12])
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$\tau_{\mathrm{approx}}(t)/\tau_{\mathrm{exact}}(t) - 1$')
    ax[1].set_title(r'(b) Relative error of interpolation of $\tau(t)$')
    ax[1].grid(True)
    ax[1].legend(fontsize='x-small')

    ax[1].yaxis.set_major_formatter(PercentFormatter(decimals=0))

    plt.tight_layout()

    # Save plot
    Filename = 'Example1'
    if test:
        Filename = "test_" + Filename
    SavePlot(plt,Filename,TransparentBackground=False)

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

        python examples/Plot_TraceInv_FullRank.py

    The script generates the figure below (see Figure 2 of [Ameli-2020]_).

    .. image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/Example1.svg
       :align: center

    **References**

    .. [Ameli-2020] Ameli, S., and Shadden. S. C. (2020). Interpolating the Trace of the Inverse of Matrix **A** + t **B**. `arXiv:2009.07385 <https://arxiv.org/abs/2009.07385>`__ [math.NA]

    This function uses three methods

        1. Maximizing log likelihood with parameters ``sigma`` and ``sigma0``
        2. Maximizing log likelihood with parameters ``sigma`` and ``eta``
        3. Finding zeros of derivative of log likelihood

    This script uses a single data, for which the random noise with a given standard deviation is added to the data once.
    It plots

        1. liklelihood in 3D as function of parameters sigma and eta
        2. Trace estimation using interpolation
        3. Derivative of log likelihood.
    """

    # Generate noisy data
    if test:
        NumPoints = 20
    else:
        NumPoints = 50

    # Generate matrix
    A = GenerateMatrix(
        NumPoints,
        DecorrelationScale=0.1,
        nu=0.5,
        UseSparse=False,
        GridOfPoints=True)
   
    # List of interpolant points
    InterpolantPoints_1 = [1e-4,4e-4,1e-3,1e-2,1e-1,1,1e+1,1e+2,1e+3]
    InterpolantPoints_2 = [1e-4,1e-3,1e-2,1e-1,1,1e+1,1e+3]
    InterpolantPoints_3 = [1e-3,1e-2,1e-1,1e+1,1e+3]
    InterpolantPoints_4 = [1e-3,1e-1,1e+1]
    InterpolantPoints_5 = [1e-1]

    # Interpolating objects
    ComputeOptions = {'ComputeMethod':'cholesky','UseInverseMatrix':True}
    InterpolationMethod = 'RMBF'
    TI_1 = InterpolateTraceOfInverse(A,InterpolantPoints=InterpolantPoints_1,InterpolationMethod=InterpolationMethod,ComputeOptions=ComputeOptions)
    TI_2 = InterpolateTraceOfInverse(A,InterpolantPoints=InterpolantPoints_2,InterpolationMethod=InterpolationMethod,ComputeOptions=ComputeOptions)
    TI_3 = InterpolateTraceOfInverse(A,InterpolantPoints=InterpolantPoints_3,InterpolationMethod=InterpolationMethod,ComputeOptions=ComputeOptions)
    TI_4 = InterpolateTraceOfInverse(A,InterpolantPoints=InterpolantPoints_4,InterpolationMethod=InterpolationMethod,ComputeOptions=ComputeOptions)
    TI_5 = InterpolateTraceOfInverse(A,InterpolantPoints=InterpolantPoints_5,InterpolationMethod=InterpolationMethod,ComputeOptions=ComputeOptions)

    # List of interpolating objects
    TI = [TI_1,TI_2,TI_3,TI_4,TI_5]

    # Plot interpolations
    Plot(TI,test)

# ====
# Main
# ====

if __name__ == "__main__":
    sys.exit(main())
