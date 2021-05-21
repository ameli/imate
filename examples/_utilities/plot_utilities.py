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
from matplotlib.ticker import FormatStrFormatter                   # noqa: F401
from matplotlib.ticker import ScalarFormatter, NullFormatter       # noqa: F401
from mpl_toolkits.axes_grid1.inset_locator import inset_axes       # noqa: F401
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition    # noqa: F401
from mpl_toolkits.axes_grid1.inset_locator import mark_inset       # noqa: F401
from distutils.spawn import find_executable
from .display_utilities import is_notebook

# Check DISPLAY
if ((not bool(os.environ.get('DISPLAY', None))) or
    (not bool(os.environ.get('IMATE_DISPLAY', None)))) \
            and (not is_notebook()):

    # No display found (often used during test phase on servers). Using
    # non-interactive backend.
    if platform.system() == 'Darwin':
        # For MacOS, first, use macos baclend, "then" import pyplot
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
import logging
logging.captureWarnings(True)
import warnings                                                    # noqa: E402
warnings.filterwarnings(action='ignore', module='matplotlib',
                        category=UserWarning,
                        message=('This figure includes Axes that are not ' +
                                 'compatible with tight_layout, so results ' +
                                 'might be incorrect.'))


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
            # plt.rc('text', usetex=True)
            matplotlib.rcParams['text.usetex'] = True
            matplotlib.rcParams['text.latex.preamble'] = \
                r'\usepackage{amsmath}'
            matplotlib.font_manager._rebuild()

            # LaTeX font is a bit small. Increaset axes font size
            sns.set(font_scale=1.2)
        except Exception:
            pass

    # Style sheet
    sns.set_style("white")
    sns.set_style("ticks")

    # Font (Note: this should be AFTER the plt.style.use)
    plt.rc('font', family='serif')
    plt.rcParams['svg.fonttype'] = 'none'  # text in svg will be text not path.

    # from cycler import cycler
    # matplotlib.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')


# =========
# Save Plot
# =========

def save_plot(plt, filename, transparent_background=True):
    """
    Saves plots.

    :param plt: matplotlib.pyplot object for the plots.
    :type plt: matplotlib.pyplot

    :param filename: Name of the file without extension or directory name.
    :type filename: string

    Format:
        The file is saved in both ``svg`` and ``pdf`` format.

    Directory:
        The plot is saved in the directory ``/docs/images/`` with respect to
        the package root, if this directory is exists and writable. Otherwise,
        the plot is saved in the *current* directory of the user.
    """

    # Get the root directory of the package (parent directory of this script)
    file_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(file_directory)
    second_parent_directory = os.path.dirname(parent_directory)

    # Try to save in the docs/images directory. Check if exists and writable
    save_dir = os.path.join(second_parent_directory, 'docs', 'images')
    if (not os.path.isdir(save_dir)) or (not os.access(save_dir, os.W_OK)):

        # Write in the current working directory
        save_dir = os.getcwd()

    # Save plot in both svg and pdf format
    filename_PDF = filename + '.pdf'
    filename_SVG = filename + '.svg'
    if os.access(save_dir, os.W_OK):
        save_fullname_SVG = os.path.join(save_dir, filename_SVG)
        save_fullname_PDF = os.path.join(save_dir, filename_PDF)
        plt.savefig(save_fullname_SVG, transparent=transparent_background,
                    bbox_inches='tight')
        plt.savefig(save_fullname_PDF, transparent=transparent_background,
                    bbox_inches='tight')
        print('')
        print('Plot saved to "%s".' % (save_fullname_SVG))
        print('Plot saved to "%s".' % (save_fullname_PDF))
    else:
        print('Cannot save plot to %s. Directory is not writable.' % save_dir)
