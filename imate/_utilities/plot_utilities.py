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
from matplotlib.ticker import ScalarFormatter, NullFormatter       # noqa: F401
from matplotlib.ticker import FormatStrFormatter, FuncFormatter    # noqa: F401

import shutil
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

__all__ = ['get_custom_theme', 'set_custom_theme', 'save_plot',
           'show_or_save_plot']


# =====================
# customize theme style
# =====================

def _customize_theme_style():
    """
    Get the parameters that control the general style of the plots.

    The style parameters control properties like the color of the background
    and whether a grid is enabled by default. This is accomplished using the
    matplotlib rcParams system.
    """

    # Define colors here
    dark_gray = ".15"
    light_gray = ".8"

    # Common parameters
    style_dict = {

        "figure.facecolor": "white",
        "axes.labelcolor": dark_gray,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": dark_gray,
        "ytick.color": dark_gray,

        "axes.axisbelow": True,
        "grid.linestyle": "-",

        "text.color": dark_gray,
        "font.family": ["sans-serif"],
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans",
                            "Bitstream Vera Sans", "sans-serif"],

        "lines.solid_capstyle": "round",
        "patch.edgecolor": "w",
        "patch.force_edgecolor": True,

        "xtick.top": False,
        "ytick.right": False,
    }

    # Set grid
    style_dict.update({
        "axes.grid": False,
    })

    # Set the color of the background, spines, and grids
    style_dict.update({

        "axes.facecolor": "white",
        "axes.edgecolor": dark_gray,
        "grid.color": light_gray,

        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,

    })

    # Show the axes ticks
    style_dict.update({
        "xtick.bottom": True,
        "ytick.left": True,
    })

    return style_dict


# =======================
# customize theme context
# =======================

def _customize_theme_context(context="notebook", font_scale=1):
    """
    Get the parameters that control the scaling of plot elements.

    These parameters correspond to label size, line thickness, etc. For more
    information, see the :doc:`aesthetics tutorial <../tutorial/aesthetics>`.

    The base context is "notebook", and the other contexts are "paper", "talk",
    and "poster", which are version of the notebook parameters scaled by
    different values. Font elements can also be scaled independently of (but
    relative to) the other values.

    Parameters
    ----------

    context : None, dict, or one of {paper, notebook, talk, poster}
        A dictionary of parameters or the name of a preconfigured set.

    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.
    """

    contexts = ["paper", "notebook", "talk", "poster"]
    if context not in contexts:
        raise ValueError(f"context must be in {', '.join(contexts)}")

    # Set up dictionary of default parameters
    texts_base_context = {

        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,

    }

    base_context = {

        "axes.linewidth": 1.25,
        "grid.linewidth": 1,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "patch.linewidth": 1,

        "xtick.major.width": 1.25,
        "ytick.major.width": 1.25,
        "xtick.minor.width": 1,
        "ytick.minor.width": 1,

        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,

    }

    base_context.update(texts_base_context)

    # Scale all the parameters by the same factor depending on the context
    scaling = dict(paper=.8, notebook=1, talk=1.5, poster=2)[context]
    context_dict = {k: v * scaling for k, v in base_context.items()}

    # Now independently scale the fonts
    font_keys = texts_base_context.keys()
    font_dict = {k: context_dict[k] * font_scale for k in font_keys}
    context_dict.update(font_dict)

    return context_dict


# ====================
# customize theme text
# ====================

def _customize_theme_text():
    """
    Returns a dictionary of settings that primarily sets LaTeX, if exists.
    """

    text_dict = {}

    # LaTeX
    if shutil.which('latex'):
        text_dict['text.usetex'] = True
        text_dict['text.latex.preamble'] = r'\usepackage{amsmath}'

    # Font (Note: this should be AFTER the plt.style.use)
    text_dict['font.family'] = 'serif'
    text_dict['svg.fonttype'] = 'none'  # text in svg will be text not path

    return text_dict


# ================
# get custom theme
# ================

def get_custom_theme(
        context="notebook",
        font_scale=1,
        use_latex=True,
        **kwargs):
    """
    Returns a dictionary that can be used to update plt.rcParams.

    Usage:
    Before a function, add this line:

    @matplotlib.rc_context(get_custom_theme(font_scale=1.2))
    def some_plotting_function():
        ...

        plot.show()

    Note that the plot.show() must be within the "context" (meaning the scope)
    of the above rc_context declaration. That is, if plt.show() is postponed
    to be a global plt.show() outside of the above function, the matplotlib
    parameter settings will be set back to their defaults. Hence, make sure to
    plot within the scope of the intended function where the rcParams context
    is customized.

    By setting font_scale=1, a pre-set of axes tick sizes are applied to the
    plot which are different than the default matplotlib sizes. To disable
    these pre-set sizes, set font_scale=None.
    """

    plt_rc_params = {}

    # Set the style (such as the which background, ticks)
    plt_rc_params.update(_customize_theme_style())

    # Set the context (such as scaling font sizes)
    if font_scale is not None:
        plt_rc_params.update(_customize_theme_context(
            context=context, font_scale=font_scale))

    # Set text rendering and font (such as using LaTeX)
    if use_latex is True:
        plt_rc_params.update(_customize_theme_text())

    # Add extra arguments
    plt_rc_params.update(kwargs)

    return plt_rc_params


# ================
# set custom theme
# ================

def set_custom_theme(context="notebook", font_scale=1, use_latex=True):
    """
    Sets a customized theme for plotting.
    """

    plt_rc_params = get_custom_theme(context=context, font_scale=font_scale,
                                     use_latex=use_latex)
    matplotlib.rcParams.update(plt_rc_params)


# =========
# save plot
# =========

def save_plot(
        plt,
        filename,
        save_dir=None,
        transparent_background=True,
        pdf=True,
        bbox_extra_artists=None,
        dpi=200,
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

    # If no directory specified, write in the current working directory
    if save_dir is None:
        save_dir = os.getcwd()

    # Save plot in both svg and pdf format
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
                    save_fullname_pdf, dpi=dpi,
                    transparent=transparent_background,
                    bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
            plt.close()
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
        dpi=200,
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
                  pdf=pdf, bbox_extra_artists=bbox_extra_artists, dpi=dpi,
                  verbose=verbose)
        plt.close()
