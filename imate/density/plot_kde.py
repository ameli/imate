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

import numpy
from .kde import _max_bandwidth

try:
    from .._utilities.plot_utilities import matplotlib, plt, get_theme, \
            show_or_save_plot
    plot_modules_exist = True
except ImportError:
    plot_modules_exist = False

__all__ = ['plot_kde']


# ===========
# log log fit
# ===========

def _log_log_fit(xi, yi, x):
    """
    """

    logxi = numpy.log10(xi)
    logyi = numpy.log10(yi)
    logx = numpy.log10(x)
    p = numpy.polyfit(logxi, logyi, 1)
    logy = numpy.polyval(p, logx)
    y = 10**logy

    return y, p[0]


# ====================
# plot density on axis
# ====================

def _plot_density_on_axis(ax, x, density, log_scale, label, title):
    """
    """

    ax.plot(x, density, label=label, color='black')
    # bg_color = 'lightgray'
    bg_color = 'wheat'
    ax.fill_between(x=x, y1=density, color=bg_color, alpha=0.5, zorder=-10)
    ax.set_xlim([x[0], x[-1]])

    if log_scale:
        ax.set_xscale('log')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\hat{p}(\lambda \vert h(\lambda))$')
    ax.set_title(title)


# ============
# plot density
# ============

@matplotlib.rc_context(get_theme())
def plot_kde(
        data,
        x,
        density,
        bw_list,
        min_bw,
        max_bw,
        cumulative,
        log_scale,
        filename=None,
        verbose=False):
    """
    """

    if not plot_modules_exist:
        raise ImportError('Cannot import modules for plotting. Either ' +
                          'install "matplotlib" package or set "plot=False".')

    figsize = [7, 9]
    nrows = 3

    fig, ax = plt.subplots(nrows=nrows, figsize=tuple(figsize))
    if nrows == 1:
        ax = [ax]

    label = 'Estimate'
    if cumulative:
        title = 'Cumulative Distribution Function'
    else:
        title = 'Probability Density Function'

    _plot_density_on_axis(ax[0], x, density, log_scale, label, title)
    ax[0].hlines([0], x[0], x[-1], color='gray', zorder=-1000)

    if cumulative:
        ax[0].set_ylim([0, 1])

    # Rug plot on the first axis
    rug_size = ax[0].get_ylim()[1] * 0.05
    segments = [((z, -rug_size / 100), (z, -rug_size)) for z in data]
    lc = matplotlib.collections.LineCollection(
            segments, linewidths=0.5, colors='black', zorder=-10)
    ax[0].add_collection(lc)
    ax[0].set_ylim(bottom=-rug_size)

    # Create a custom legend item for the rug plot
    ax[0].scatter([], [], color='black', marker='|', linewidths=0.5, s=100,
                  label='Eigenvalues')
    legend_color = (1, 1, 1, 0.8)
    ax[0].legend(fontsize='xx-small', facecolor=legend_color)

    # Plot log-scale (second axis)
    _plot_density_on_axis(ax[1], x, density, log_scale, label, title)
    ax[1].set_yscale('log')
    ax[1].legend(fontsize='xx-small', facecolor=legend_color)

    # Limit the y scale limit
    ylim = ax[1].get_ylim()
    if ylim[0] < 1e-16:
        ylim_buttom = 1e-16
        ylim_top = 10.0**(numpy.ceil(numpy.log10(numpy.max(density)) + 0.5))
        ax[1].set_ylim([ylim_buttom, ylim_top])
    elif ylim[1] > 1e+16:
        ylim_buttom = 10.0**(numpy.ceil(numpy.log10(numpy.min(density)) - 0.5))
        ylim_top = 1e+16
        ax[1].set_ylim([ylim_buttom, ylim_top])

    # Plot bandwidth
    if max_bw is not None:
        # recreate max_bw with better resolution
        data_hires = numpy.logspace(numpy.log10(data[0]),
                                    numpy.log10(data[-1]), 1000)
        max_bw = _max_bandwidth(data_hires, bw_list[-1])
        ax[-1].plot(data_hires, max_bw, '--', color='black',
                    label='Upper bound')

    if min_bw is not None:
        ax[-1].plot(data, min_bw, linestyle='dotted', color='black',
                    label='Lower bound')

    colors = plt.cm.Greys(numpy.linspace(0, 1, len(bw_list)+1))
    for i in range(len(bw_list)):
        ax[-1].plot(data, bw_list[i], color=colors[i+1],
                    label='Iteration %d' % (i+1))

    ax[-1].set_xlim([x[0], x[-1]])
    if log_scale:
        ax[-1].set_xscale('log')
    ax[-1].set_yscale('log')
    ax[-1].set_xlabel(r'$\lambda$')
    ax[-1].set_ylabel(r'$h(\lambda)$')
    ax[-1].set_title('Kernel Bandwidth')
    ax[-1].legend(fontsize='xx-small', facecolor=legend_color)

    plt.tight_layout()

    show_or_save_plot(plt, filename=filename, default_filename='density',
                      transparent_background=True, bbox_extra_artists=None,
                      verbose=verbose)
