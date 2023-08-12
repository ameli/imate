#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

import sys
import numpy
import scipy
import scipy.special

# Package modules
from _utilities.plot_utilities import *                      # noqa: F401, F403
from _utilities.plot_utilities import load_plot_settings, save_plot, plt, \
        matplotlib


# ==============
# Plot Functions
# ==============

def plot_chebyshev_rational(degree=6, test=False):
    """
    Plots Chebyshev rational functions.
    """

    # Load plot settings
    load_plot_settings(font_scale=1.2)

    t = numpy.logspace(-4, 4, 1000)
    x = (t-1.0) / (t+1.0)

    fig, ax = plt.subplots(figsize=(7, 4.8))

    # for deg in range(0, 3):
    for deg in range(1, degree+1):
        y = 0.5 * (1.0 - scipy.special.eval_chebyt(deg, x))
        # y = (scipy.special.eval_chebyt(deg, x))
        ax.plot(t, y, label=r'$i = %d$' % (deg))

    # ax.legend(ncol=3, loc='lower left', borderpad=0.5, frameon=False)
    ax.legend(loc='upper right', borderpad=0.5, frameon=False)
    ax.set_xlim([t[0], t[-1]])
    ax.set_xscale('log')
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\frac{1}{2}(1-r_i(t))$')
    ax.set_title('Chebyshev rational functions')
    ax.grid(axis='y')

    if not test:
        plt.tight_layout()

    # Save plot
    filename = 'chebyshev'
    if test:
        filename = "test_" + filename
    save_plot(plt, filename, transparent_background=True)

    # If no display backend is enabled, do not plot in the interactive mode
    if (not test) and (matplotlib.get_backend() != 'agg'):
        plt.show()


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(plot_chebyshev_rational())
