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
from imate import InterpolateSchatten
from _utilities.data_utilities import generate_matrix
from _utilities.plot_utilities import *                      # noqa: F401, F403
from _utilities.plot_utilities import load_plot_settings, save_plot, plt, \
        matplotlib, InsetPosition, mark_inset, NullFormatter,  \
        PercentFormatter, ScalarFormatter


# ====
# plot
# ====

def plot_fun_and_error(TI, test):
    """
    Plots the curve of trace of Kn inverse versus eta.
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
        eta_resolution = 500
    eta = numpy.r_[-numpy.logspace(-9, -3.0001, eta_resolution)[::-1], 0,
                   numpy.logspace(-9, 3, eta_resolution)]
    zero_index = numpy.argmin(numpy.abs(eta))

    # Functions
    trace_exact = TI[0].eval(eta)
    trace_lowerbound = TI[0].bound(eta)
    trace_estimate = numpy.zeros((num_plots, eta.size))
    for j in range(num_plots):
        trace_estimate[j, :] = TI[j].interpolate(eta)

    # Tau
    trace_B = 1
    tau_exact = trace_exact / trace_B
    tau_lowerbound = trace_lowerbound / trace_B
    tau_estimate = trace_estimate / trace_B

    # Plots trace
    textwidth = 9.0  # in inches
    # fig, ax = plt.subplots(nrows=1, ncols=2,
    #                        figsize=(textwidth, textwidth/2))
    fig, ax = plt.subplots(nrows=1, ncols=2,
                           figsize=(textwidth, textwidth/2.5))
    ax[0].plot(eta, tau_exact, color='black', label='Exact')
    ax[0].plot(eta[zero_index:], tau_lowerbound[zero_index:], '--',
               color='black', label=r'Lower bound (at $t \geq 0$)')
    ax[0].plot(eta[:zero_index], tau_lowerbound[:zero_index], '-.',
               color='black', label=r'Upper bound (at $t < 0$)')

    colors_list = ["#d62728",
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
        q = TI[j].q
        h = ax[0].plot(eta, tau_estimate[j, :],
                       label=r'Interpolation, $q=%d$' % q,
                       color=colors_list[j])
        if j == 0:
            h[0].set_zorder(20)

    ax[0].set_xscale('symlog', linthresh=1e-8)
    ax[0].set_yscale('log')
    ax[0].set_xlim([eta[0], eta[-1]])
    ax[0].set_ylim([1e-4, 1e3])
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$\tau_p(t)$')
    ax[0].set_title(r'(a) Interpolation of $\tau_p(t), p=-1$')
    ax[0].grid(True)
    ax[0].legend(fontsize='x-small', loc='upper left')
    ax[0].set_xticks(numpy.r_[-10**numpy.arange(-3, -7, -3, dtype=float), 0,
                     10**numpy.arange(-6, 4, 3, dtype=float)])
    ax[0].tick_params(axis='x', which='minor', bottom=False)

    # Inset plot
    ax2 = plt.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax[0], [0.14, 0.25, 0.45, 0.35])
    ax2.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on ax1 and draw lines
    # in grey linking the two axes.

    # Avoid inset mark lines intersect the inset axes itself by setting anchor
    inset_color = 'oldlace'
    mark_inset(ax[0], ax2, loc1=1, loc2=4, facecolor=inset_color,
               edgecolor='0.5')
    ax2.plot(eta, tau_exact, color='black', label='Exact')
    ax2.plot(eta[zero_index:], tau_lowerbound[zero_index:], '--',
             color='black', label=r'Lower bound (at $t \geq 0$)')
    ax2.plot(eta[:zero_index], tau_lowerbound[:zero_index], '-.',
             color='black', label=r'Upper bound (at $t < 0$)')
    for j in reversed(range(num_plots)):
        ax2.plot(eta, tau_estimate[j, :], color=colors_list[j])
    ax2.set_xlim([1e-2, 1.15e-2])
    ax2.set_xticks([1e-2, 1.15e-2])
    ax2.set_ylim(0.0111, 0.0125)
    ax2.set_yticks([0.0111, 0.0125])
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xticklabels(['0.01', '0.0115'])
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.set_facecolor(inset_color)
    ax2.xaxis.set_tick_params(labelsize=8)
    ax2.yaxis.set_tick_params(labelsize=8)
    # plt.setp(ax2.get_yticklabels(), backgroundcolor='white')

    # Plot errors
    ax[1].semilogx(eta[zero_index:],
                   100*(1-tau_lowerbound[zero_index:]/tau_exact[zero_index:]),
                   '--', color='black', label=r'Lower bound (at $t \geq 0$)',
                   zorder=15)  # Relative error
    ax[1].semilogx(eta[:zero_index],
                   100*(1-tau_lowerbound[:zero_index]/tau_exact[:zero_index]),
                   '-.', color='black', label=r'Upper bound (at $t < 0$)',
                   zorder=15)  # Relative error
    for j in reversed(range(num_plots)):
        q = TI[j].q
        h = ax[1].semilogx(eta, 100*(1-tau_estimate[j, :]/tau_exact),
                           label=r'Interpolation, $q=%d$' % q,
                           color=colors_list[j])       # Relative error
        if j == 0:
            h[0].set_zorder(20)
    ax[1].set_xscale('symlog', linthresh=1e-8)
    ax[1].set_yscale('linear')
    ax[1].set_xlim([eta[0], eta[-1]])
    ax[1].set_ylim([-1, 2])
    ax[1].set_yticks(numpy.arange(-1, 2.1, 0.5))
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(
            r'$1-\tau_{\mathrm{approx}}(t)/\tau_{\mathrm{exact}}(t)$')
    ax[1].set_title(r'(b) Relative error of interpolation, $p=-1$')
    ax[1].grid(True)
    ax[1].legend(fontsize='x-small', loc='upper left')
    ax[1].set_xticks(numpy.r_[-10**numpy.arange(-3, -7, -3, dtype=float), 0,
                     10**numpy.arange(-6, 4, 3, dtype=float)])
    ax[1].tick_params(axis='x', which='minor', bottom=False)

    ax[1].yaxis.set_major_formatter(PercentFormatter(decimals=1))

    if not test:
        plt.tight_layout()

    # Save Plot
    filename = 'traceinv_ill_conditioned_cheb'
    if test:
        filename = "test_" + filename
    save_plot(plt, filename, transparent_background=False)

    # If no display backend is enabled, do not plot in the interactive mode
    if (not test) and (matplotlib.get_backend() != 'agg'):
        plt.show()


# ===============
# plot error only
# ===============

def plot_error_only(TI, test):
    """
    Plots the curve of trace of Kn inverse versus eta.
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
        eta_resolution = 500
    eta = numpy.r_[-numpy.logspace(-9, -3.0001, eta_resolution)[::-1], 0,
                   numpy.logspace(-9, 3, eta_resolution)]
    zero_index = numpy.argmin(numpy.abs(eta))

    # Functions
    trace_exact = TI[0].eval(eta)
    trace_lowerbound = TI[0].bound(eta)
    trace_estimate = numpy.zeros((num_plots, eta.size))
    for j in range(num_plots):
        trace_estimate[j, :] = TI[j].interpolate(eta)

    # Tau
    trace_B = 1
    tau_exact = trace_exact / trace_B
    tau_lowerbound = trace_lowerbound / trace_B
    tau_estimate = trace_estimate / trace_B

    # Plots trace
    textwidth = 9.0  # in inches
    # fig, ax = plt.subplots(nrows=1, ncols=2,
    #                        figsize=(textwidth, textwidth/2))
    fig, ax = plt.subplots(figsize=(textwidth/1.946, textwidth/2.5))

    # ax[0].plot(eta, tau_exact, color='black', label='Exact')
    # ax[0].plot(eta[zero_index:], tau_lowerbound[zero_index:], '--',
    #            color='black', label=r'Lower bound (at $t \geq 0$)')
    # ax[0].plot(eta[:zero_index], tau_lowerbound[:zero_index], '-.',
    #            color='black', label=r'Upper bound (at $t < 0$)')

    colors_list = ["#d62728",
                   "#2ca02c",
                   "#bcbd22",
                   "#ff7f0e",
                   "#1f77b4",
                   "#9467bd",
                   "#8c564b",
                   "#17becf",
                   "#7f7f7f",
                   "#e377c2"]

    # for j in reversed(range(num_plots)):
    #     q = TI[j].q
    #     h = ax[0].plot(eta, tau_estimate[j, :],
    #                    label=r'Interpolation, $q=%d$' % q,
    #                    color=colors_list[j])
    #     if j == 0:
    #         h[0].set_zorder(20)
    #
    # ax[0].set_xscale('symlog', linthresh=1e-8)
    # ax[0].set_yscale('log')
    # ax[0].set_xlim([eta[0], eta[-1]])
    # ax[0].set_ylim([1e-4, 1e3])
    # ax[0].set_xlabel(r'$t$')
    # ax[0].set_ylabel(r'$\tau_p(t)$')
    # ax[0].set_title(r'(a) Interpolation of $\tau_p(t), p=-1$')
    # ax[0].grid(True)
    # ax[0].legend(fontsize='x-small', loc='upper left')
    # ax[0].set_xticks(numpy.r_[-10**numpy.arange(-3, -7, -3, dtype=float), 0,
    #                  10**numpy.arange(-6, 4, 3, dtype=float)])
    # ax[0].tick_params(axis='x', which='minor', bottom=False)
    #
    # # Inset plot
    # ax2 = plt.axes([0, 0, 1, 1])
    # # Manually set the position and relative size of inset axes within ax1
    # ip = InsetPosition(ax[0], [0.14, 0.25, 0.45, 0.35])
    # ax2.set_axes_locator(ip)
    # # Mark the region corresponding to the inset axes on ax1 and draw lines
    # # in grey linking the two axes.
    #
    # # Avoid inset mark lines intersect inset axes itself by setting anchor
    # inset_color = 'oldlace'
    # mark_inset(ax[0], ax2, loc1=1, loc2=4, facecolor=inset_color,
    #            edgecolor='0.5')
    # ax2.plot(eta, tau_exact, color='black', label='Exact')
    # ax2.plot(eta[zero_index:], tau_lowerbound[zero_index:], '--',
    #          color='black', label=r'Lower bound (at $t \geq 0$)')
    # ax2.plot(eta[:zero_index], tau_lowerbound[:zero_index], '-.',
    #          color='black', label=r'Upper bound (at $t < 0$)')
    # for j in reversed(range(num_plots)):
    #     ax2.plot(eta, tau_estimate[j, :], color=colors_list[j])
    # ax2.set_xlim([1e-2, 1.15e-2])
    # ax2.set_xticks([1e-2, 1.15e-2])
    # ax2.set_ylim(0.0111, 0.0125)
    # ax2.set_yticks([0.0111, 0.0125])
    # ax2.xaxis.set_minor_formatter(NullFormatter())
    # ax2.set_xticklabels(['0.01', '0.0115'])
    # ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax2.set_facecolor(inset_color)
    # ax2.xaxis.set_tick_params(labelsize=8)
    # ax2.yaxis.set_tick_params(labelsize=8)
    # # plt.setp(ax2.get_yticklabels(), backgroundcolor='white')

    # Plot errors
    ax.semilogx(eta[zero_index:],
                100*(1-tau_lowerbound[zero_index:]/tau_exact[zero_index:]),
                '--', color='black', label=r'Lower bound (at $t \geq 0$)',
                zorder=15)  # Relative error
    ax.semilogx(eta[:zero_index],
                100*(1-tau_lowerbound[:zero_index]/tau_exact[:zero_index]),
                '-.', color='black', label=r'Upper bound (at $t < 0$)',
                zorder=15)  # Relative error
    for j in reversed(range(num_plots)):
        q = TI[j].q
        h = ax.semilogx(eta, 100*(1-tau_estimate[j, :]/tau_exact),
                        label=r'Interpolation, $q=%d$' % q,
                        color=colors_list[j])       # Relative error
        if j == 0:
            h[0].set_zorder(20)
    ax.set_xscale('symlog', linthresh=1e-8)
    ax.set_yscale('linear')
    ax.set_xlim([eta[0], eta[-1]])
    ax.set_ylim([-1, 2])
    ax.set_yticks(numpy.arange(-1, 2.1, 0.5))
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(
         r'$1-\tau_{\mathrm{approx}}(t)/\tau_{\mathrm{exact}}(t)$')
    ax.set_title(r'(b) Relative error of interpolation, $p=-1$')
    ax.grid(True)
    ax.legend(fontsize='x-small', loc='upper left')
    ax.set_xticks(numpy.r_[-10**numpy.arange(-3, -7, -3, dtype=float), 0,
                  10**numpy.arange(-6, 4, 3, dtype=float)])
    ax.tick_params(axis='x', which='minor', bottom=False)

    ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))

    if not test:
        plt.tight_layout()

    # Save Plot
    filename = 'traceinv_ill_conditioned_cheb'
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

        python examples/Plot_imate_IllConditioned.py

    The script generates the figure below (see also  Figure 3 of
    [Ameli-2020]_).

    .. image:: https://raw.githubusercontent.com/ameli/imate/main/docs/images/E
               xample2.svg
       :align: center

    **References**

    .. [Ameli-2020] Ameli, S., and Shadden. S. C. (2020). Interpolating the
    Trace of the Inverse of Matrix **A** + t **B**. `arXiv:2009.07385
    <https://arxiv.org/abs/2009.07385>`__ [math.NA]
    """

    # shift to make singular matrix non-singular
    # shift = 2e-4
    # shift = 4e-4
    shift = 1e-3

    # Generate a nearly singular matrix
    if test:
        n = 100
        m = 50
    else:
        n = 1000
        m = 500
    K = generate_matrix(n, m, shift)

    # Interpolating points
    interpolant_points_1 = [1e-3, 4e-3, 1e-2, 4e-2, 1e-1, 1]
    interpolant_points_2 = [1e-3, 1e-2, 1e-1, 1]
    interpolant_points_3 = [1e-3, 1e-1]

    # Interpolating objects
    options = {'method': 'cholesky', 'invert_cholesky': True}
    # kind = 'RPF'
    kind = 'CRF'
    scale = None
    p = -1
    TI_1 = InterpolateSchatten(K, p=p, ti=interpolant_points_1, kind=kind,
                               scale=scale, options=options)

    TI_2 = InterpolateSchatten(K, p=p, ti=interpolant_points_2, kind=kind,
                               scale=scale, options=options)

    TI_3 = InterpolateSchatten(K, p=p, ti=interpolant_points_3, kind=kind,
                               scale=scale, options=options)

    # List of interpolating objects
    TI = [TI_1, TI_2, TI_3]

    # Plot interpolations
    # plot_func_and_error(TI, test)
    plot_error_only(TI, test)


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(main())
