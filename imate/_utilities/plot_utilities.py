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

import os
import platform
import matplotlib
import matplotlib.ticker
from matplotlib.ticker import PercentFormatter                     # noqa: F401
from mpl_toolkits.mplot3d import Axes3D                            # noqa: F401
from mpl_toolkits.axes_grid1.inset_locator import inset_axes       # noqa: F401
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition    # noqa: F401
from mpl_toolkits.axes_grid1.inset_locator import mark_inset       # noqa: F401
from matplotlib.ticker import ScalarFormatter, NullFormatter       # noqa: F401
from matplotlib.ticker import FormatStrFormatter, FuncFormatter    # noqa: F401
from mpl_toolkits.axes_grid1 import make_axes_locatable            # noqa: F401

from distutils.spawn import find_executable
from .display_utilities import is_notebook
import logging
import warnings

# Check DISPLAY
if ((not bool(os.environ.get('DISPLAY', None))) or
        (bool(os.environ.get('IMATE_NO_DISPLAY', None)))) and \
        (not is_notebook()):

    # No display found (used on servers). Using non-interactive backend
    if platform.system() == 'Darwin':
        # For MacOS, first, use macos backend, "then" import pyplot
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
    else:
        # For Linux and Windows, "first" import pyplot, then use Agg backend.
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
else:
    # Display exists. Import pyplot without changing any backend.
    import matplotlib.pyplot as plt

# Remove plt.tight_layout() warning
logging.captureWarnings(True)
warnings.filterwarnings(
        action='ignore',
        module='matplotlib',
        category=UserWarning,
        message=('This figure includes Axes that are not compatible with ' +
                 'tight_layout, so results might be incorrect.'))


# ==================
# load plot settings
# ==================

def load_plot_settings():
    """
    Specifies general settings for the plots in the example scripts,
    namely, it sets plot themes by ``seaborn``, fonts by LaTeX if available.
    """

    # Color palette
    import seaborn as sns
    # sns.set()

    # LaTeX
    if find_executable('latex'):
        try:
            # plt.rc('text',usetex=True)
            matplotlib.rcParams['text.usetex'] = True
            matplotlib.rcParams['text.latex.preamble'] = \
                r'\usepackage{amsmath}'

            # LaTeX font is a bit small. Increase axes font size
            sns.set(font_scale=1.2)

        except Exception:
            pass

    # Style sheet
    sns.set_style("white")
    sns.set_style("ticks")

    # Font (Note: this should be AFTER the plt.style.use)
    plt.rc('font', family='serif')
    plt.rcParams['svg.fonttype'] = 'none'   # text in svg will be text not path

    # from cycler import cycler
    # matplotlib.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')


# =========
# save plot
# =========

def save_plot(
        plt,
        filename,
        transparent_background=True,
        pdf=True,
        bbox_extra_artists=None,
        verbose=False):
    """
    Saves plot as svg format in the current working directory.

    :param plt: matplotlib.pyplot object for the plots.
    :type plt: matplotlib.pyplot

    :param filename: Name of the file without extension or directory name.
    :type filename: string

    :param transparent_background: Sets the background of svg file to be
        transparent.
    :type transparent_background: bool
    """

    # Write in the current working directory
    save_dir = os.getcwd()

    # Save plot in svg format
    filename_svg = filename + '.svg'
    filename_pdf = filename + '.pdf'
    if os.access(save_dir, os.W_OK):
        save_fullname_svg = os.path.join(save_dir, filename_svg)
        save_fullname_pdf = os.path.join(save_dir, filename_pdf)

        plt.savefig(
                save_fullname_svg,
                transparent=transparent_background,
                bbox_inches='tight')
        if verbose:
            print('Plot saved to "%s".' % (save_fullname_svg))

        if pdf:
            plt.savefig(
                    save_fullname_pdf,
                    transparent=transparent_background,
                    bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
            if verbose:
                print('Plot saved to "%s".' % (save_fullname_pdf))
    else:
        print('Cannot save plot to %s. Directory is not writable.' % save_dir)


# =================
# show or save plot
# =================

def show_or_save_plot(
        plt,
        filename,
        transparent_background=True,
        pdf=True,
        bbox_extra_artists=None,
        verbose=False):
    """
    Shows the plot. If no graphical beckend exists, saves the plot.
    """

    # Check if the graphical back-end exists
    if matplotlib.get_backend() != 'agg' or is_notebook():
        plt.show()
    else:
        # write the plot as SVG file in the current working directory
        save_plot(plt, filename, transparent_background=transparent_background,
                  pdf=pdf, bbox_extra_artists=bbox_extra_artists,
                  verbose=verbose)
