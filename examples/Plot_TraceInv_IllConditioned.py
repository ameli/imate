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
from Utilities.DataUtilities import GenerateMatrix
from Utilities.PlotUtilities import *
from Utilities.PlotUtilities import LoadPlotSettings
from Utilities.PlotUtilities import SavePlot

# ====
# Plot
# ====

def Plot(TI,test):
    """
    Plots the curve of trace of Kn inverse versus eta.
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
        eta_Resolution = 500
    eta = numpy.r_[-numpy.logspace(-9,-3.0001,eta_Resolution)[::-1],0,numpy.logspace(-9,3,eta_Resolution)]
    ZeroIndex = numpy.argmin(numpy.abs(eta))

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

    for j in reversed(range(NumberOfPlots)):
        p = TI[j].p
        q = ax[0].plot(eta,tau_estimate[j,:],label=r'Interpolation, $p=%d$'%(p//2),color=ColorsList[j])
        if j == 0:
            q[0].set_zorder(20)

    ax[0].set_xscale('symlog',linthreshx=1e-8)
    ax[0].set_yscale('log')
    ax[0].set_xlim([eta[0],eta[-1]])
    ax[0].set_ylim([1e-3,1e4])
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$\tau(t)$')
    ax[0].set_title(r'(a) Exact, interpolation, and bounds of $\tau(t)$')
    ax[0].grid(True)
    ax[0].legend(fontsize='x-small',loc='lower left')
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
    for j in reversed(range(NumberOfPlots)):
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
    for j in reversed(range(NumberOfPlots)):
        p = TI[j].p
        # q = ax[1].semilogx(eta,tau_estimate[j,:]-tau_exact,label=r'Estimation, $p=%d$'%(p),color=ColorsList[j])  # Absolute error
        q = ax[1].semilogx(eta,100*(tau_estimate[j,:]/tau_exact-1),label=r'Interpolation, $p=%d$'%(p//2),color=ColorsList[j])       # Relative error
        if j == 0:
            q[0].set_zorder(20)
    # ax[1].semilogx(eta,tau_estimate_alt-tau_exact,label=r'Alt. estimation',zorder=-20)   # Absolute error
    # ax[1].semilogx(eta,tau_estimate_alt/tau_exact-1,label=r'Alt. estimation',zorder=-20)   # Relative error
    ax[1].set_xscale('symlog',linthreshx=1e-8)
    ax[1].set_yscale('linear')
    ax[1].set_xlim([eta[0],eta[-1]])
    ax[1].set_ylim([-0.5,2.5])
    ax[1].set_yticks(numpy.arange(-0.5,2.6,0.5))
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$\tau_{\mathrm{approx}}(t)/\tau_{\mathrm{exact}}(t) - 1$')
    ax[1].set_title(r'(b) Relative error of estimation of $\tau(t)$')
    ax[1].grid(True)
    ax[1].legend(fontsize='x-small',loc='upper left')
    ax[1].set_xticks(numpy.r_[-10**numpy.arange(-3,-7,-3,dtype=float),0,10**numpy.arange(-6,4,3,dtype=float)])
    ax[1].tick_params(axis='x',which='minor',bottom=False)

    ax[1].yaxis.set_major_formatter(PercentFormatter(decimals=1))

    plt.tight_layout()

    # Save Plot
    Filename = 'Example2'
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
    """

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
    K = GenerateMatrix(n,m,Shift)

    # Interpolatng points
    InterpolantPoints_1 = [1e-3,1e-2,1e-1,1]
    InterpolantPoints_2 = [1e-3,1e-1]

    # Interpolating objects
    InterpolationMethod = 'RPF'
    TI_1 = InterpolateTraceOfInverse(K,InterpolantPoints=InterpolantPoints_1,InterpolationMethod=InterpolationMethod)
    TI_2 = InterpolateTraceOfInverse(K,InterpolantPoints=InterpolantPoints_2,InterpolationMethod=InterpolationMethod)

    # List of interpolating objects
    TI = [TI_1,TI_2]

    # Plot interpolations
    Plot(TI,test)

# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(main())
