#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import sys
import numpy

# Package modules
from imate.sample_matrices import correlation_matrix
from imate import InterpolateTraceinv
from _utilities.plot_utilities import *                      # noqa: F401, F403
from _utilities.plot_utilities import load_plot_settings, save_plot, plt, \
        matplotlib, InsetPosition, mark_inset, NullFormatter,  \
        FormatStrFormatter, PercentFormatter


# ====
# plot
# ====

def plot(TI, test):
    """
    Plots the curve of trace of An inverse versus eta (we use t instead of eta
    in the plots).
    """

    print('Plotting ... (may take a few minutes!)')

    # Load plot settings
    load_plot_settings()

    # If not a list, embed the object into a list
    if not isinstance(TI, list):
        TI = [TI]

    num_plots = len(TI)

    # Range to plot
    if test:
        eta_resolution = 20
    else:
        eta_resolution = 100
    eta = numpy.logspace(-4, 3, eta_resolution)

    # Functions
    trace_exact = TI[0].compute(eta)
    trace_upperbound = TI[0].upper_bound(eta)
    trace_lowerbound = TI[0].lower_bound(eta)
    trace_estimate = numpy.zeros((num_plots, eta.size))
    for j in range(num_plots):
        trace_estimate[j, :] = TI[j].interpolate(eta)

    # Tau
    n = TI[0].n
    tau_exact = trace_exact / n
    tau_upperbound = trace_upperbound / n
    tau_lowerbound = trace_lowerbound / n
    tau_estimate = trace_estimate / n

    # Plots trace
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax[0].loglog(eta, tau_exact, color='black', label='Exact')
    ax[0].loglog(eta, tau_upperbound, '--', color='black', label='Upper bound')
    ax[0].loglog(eta, tau_lowerbound, '-.', color='black', label='Lower bound')

    ColorsList = ["#d62728",
                  "#2ca02c",
                  "#bcbd22",
                  "#ff7f0e",
                  "#1f77b4",
                  "#9467bd",
                  "#8c564b",
                  "#17becf",
                  "#7f7f7f",
                  "#e377c2"]

    for j in reversed(range(num_plots)):
        p = TI[j].p
        q = ax[0].loglog(eta, tau_estimate[j, :],
                         label=r'Interpolation, $p=%d$' % (p),
                         color=ColorsList[j])
        if j == 0:
            q[0].set_zorder(20)

    ax[0].set_xlim([eta[0], eta[-1]])
    ax[0].set_ylim([1e-3, 1e1])
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$\tau(t)$')
    ax[0].set_title(r'(a) Exact, interpolation, and bounds of $\tau(t)$')
    ax[0].grid(True)
    ax[0].legend(fontsize='x-small', loc='upper right')

    # Inset plot
    ax2 = plt.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax[0], [0.14, 0.1, 0.5, 0.4])
    ax2.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on ax1 and draw lines
    # in grey linking the two axes.

    # Avoid inset mark lines intersect the inset axes itself by setting anchor
    inset_color = 'oldlace'
    mark_inset(ax[0], ax2, loc1=1, loc2=2, facecolor=inset_color,
               edgecolor='0.5')
    ax2.semilogx(eta, tau_exact, color='black', label='Exact')
    ax2.semilogx(eta, tau_upperbound, '--', color='black',
                 label='Upper bound')
    for j in reversed(range(num_plots)):
        ax2.semilogx(eta, tau_estimate[j, :], color=ColorsList[j])
    ax2.set_xlim([1e-2, 1e-1])
    # ax2.set_xlim([0.35, 0.4])
    # ax2.set_ylim(2.5, 4)
    ax2.set_ylim(4, 6)
    # ax2.set_ylim(1.4, 1.6)
    # ax2.set_yticks([2.5, 3, 3.5, 4])
    ax2.set_yticks([4, 5, 6])
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.set_facecolor(inset_color)
    # plt.setp(ax2.get_yticklabels(), backgroundcolor='white')

    # ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # ax2.grid(True, axis='y')

    # Plot errors
    # ax[1].semilogx(eta, tau_upperbound-tau_exact, '--', color='black',
    #                label='Upper bound')  # Absolute error
    ax[1].semilogx(eta, 100*(tau_upperbound/tau_exact-1), '--', color='black',
                   label='Upper bound', zorder=15)  # Relative error
    for j in reversed(range(num_plots)):
        p = TI[j].p

        # Absolute Error
        # q = ax[1].semilogx(eta, tau_estimate[j, :]-tau_exact,
        #                    label=r'Estimation, $p=%d$'%(p),
        #                    color=ColorsList[j])

        # Relative Error
        q = ax[1].semilogx(eta, 100*(tau_estimate[j, :]/tau_exact-1),
                           label=r'Interpolation, $p=%d$' % (p),
                           color=ColorsList[j])
        if j == 0:
            q[0].set_zorder(20)
    ax[1].set_xlim([eta[0], eta[-1]])
    ax[1].set_yticks(numpy.arange(-0.03, 0.13, 0.03)*100)
    ax[1].set_ylim([-3, 12])
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(
            r'$\tau_{\mathrm{approx}}(t)/\tau_{\mathrm{exact}}(t) - 1$')
    ax[1].set_title(r'(b) Relative error of interpolation of $\tau(t)$')
    ax[1].grid(True)
    ax[1].legend(fontsize='x-small')

    ax[1].yaxis.set_major_formatter(PercentFormatter(decimals=0))

    if not test:
        plt.tight_layout()

    # Save plot
    filename = 'Example1'
    if test:
        filename = "test_" + filename
    save_plot(plt, filename, transparent_background=False)

    # If no display backend is enabled, do not plot in the interactive mode
    if (not test) and (matplotlib.get_backend() != 'agg'):
        plt.show()


# ====
# main
# ====

def main(test=False):
    """
    Run the script by

    ::

        python examples/plot_traceinv_full_rank.py

    The script generates the figure below (see Figure 2 of [Ameli-2020]_).

    .. image:: https://raw.githubusercontent.com/ameli/imate/main/docs/images/E
               xample1.svg
       :align: center

    **References**

    .. [Ameli-2020] Ameli, S., and Shadden. S. C. (2020). Interpolating the
    Trace of the Inverse of Matrix **A** + t **B**.
    `arXiv:2009.07385 <https://arxiv.org/abs/2009.07385>`__ [math.NA]

    This function uses three methods

        1. Maximizing log likelihood with parameters ``sigma`` and ``sigma0``
        2. Maximizing log likelihood with parameters ``sigma`` and ``eta``
        3. Finding zeros of derivative of log likelihood

    This script uses a single data, for which the random noise with a given
    standard deviation is added to the data once. It plots

        1. Likelihood in 3D as function of parameters sigma and eta
        2. Trace estimation using interpolation
        3. Derivative of log likelihood.
    """

    # Generate noisy data
    if test:
        size = 20
    else:
        size = 50

    # Generate matrix
    A = correlation_matrix(
        size,
        dimension=2,
        distance_scale=0.1,
        kernel='exponential',
        sparse=False,
        grid=True)

    # List of interpolant points
    interpolant_points_1 = [1e-4, 4e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3]
    interpolant_points_2 = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+3]
    interpolant_points_3 = [1e-3, 1e-2, 1e-1, 1e+1, 1e+3]
    interpolant_points_4 = [1e-3, 1e-1, 1e+1]
    interpolant_points_5 = [1e-1]

    # Interpolating objects
    traceinv_options = {'method': 'cholesky', 'invert_cholesky': True}
    method = 'RMBF'
    TI_1 = InterpolateTraceinv(A, interpolant_points=interpolant_points_1,
                               method=method,
                               traceinv_options=traceinv_options)

    TI_2 = InterpolateTraceinv(A, interpolant_points=interpolant_points_2,
                               method=method,
                               traceinv_options=traceinv_options)

    TI_3 = InterpolateTraceinv(A, interpolant_points=interpolant_points_3,
                               method=method,
                               traceinv_options=traceinv_options)

    TI_4 = InterpolateTraceinv(A, interpolant_points=interpolant_points_4,
                               method=method,
                               traceinv_options=traceinv_options)

    TI_5 = InterpolateTraceinv(A, interpolant_points=interpolant_points_5,
                               method=method,
                               traceinv_options=traceinv_options)

    # List of interpolating objects
    TI = [TI_1, TI_2, TI_3, TI_4, TI_5]

    # Plot interpolations
    plot(TI, test)


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(main())
