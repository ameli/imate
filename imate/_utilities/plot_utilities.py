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

__all__ = ['get_theme', 'set_theme', 'reset_theme', 'show_or_save_plot']


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


# =========
# get theme
# =========

def get_theme(
        context="notebook",
        font_scale=1,
        use_latex=True,
        rc=None):
    """
    Returns a dictionary that can be used to update plt.rcParams.

    Usage:
    Before a function, add this line:

    @matplotlib.rc_context(get_theme(font_scale=1.2))
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
    if rc is not None:
        plt_rc_params.update(rc)

    return plt_rc_params


# =========
# set theme
# =========

def set_theme(
        context="notebook",
        font_scale=1,
        use_latex=True,
        rc=None):
    """
    Sets a customized theme for plotting.
    """

    plt_rc_params = get_theme(context=context, font_scale=font_scale,
                              use_latex=use_latex, rc=rc)
    matplotlib.rcParams.update(plt_rc_params)


# ===========
# reset theme
# ===========

def reset_theme():
    """
    Reset the matplotlib theme back it default.
    """

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


# =========
# save plot
# =========

def _save_plot(
        plt,
        filename,
        transparent_background=True,
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

    # Is filename contain path, use the current path
    if os.path.isabs(filename) or os.path.isabs(os.path.expanduser(filename)):

        # Remove redundant separators
        filename = os.path.normpath(filename)

        # Extract the directory, base filename, and file extension
        directory, base_and_ext = os.path.split(filename)
        base_filename, extension = os.path.splitext(base_and_ext)

        # Initialize a list of extensions
        extensions = [extension]

    else:
        # If no directory specified, write in the current working directory
        directory = os.getcwd()
        base_filename, extension = os.path.splitext(filename)

        # Determine whether a file extension is provided or not.
        if bool(extension):
            # filename contains an extension
            extensions = [extension]
        else:
            # No extension is provided. Save to both svg and pdf
            extensions = ['svg', 'pdf']

    if not os.access(directory, os.W_OK):
        raise RuntimeError(
            'Cannot save plot to %s. Directory is not writable.' % directory)

    # For each extension, save a file
    for extension in extensions:
        fullpath_filename = os.path.join(directory,
                                         base_filename + '.' + extension)

        plt.savefig(
                fullpath_filename, dpi=dpi,
                transparent=transparent_background,
                bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')

        if verbose:
            print('Plot saved to "%s".' % fullpath_filename)


# =================
# show or save plot
# =================

def show_or_save_plot(
        plt,
        filename=None,
        default_filename=None,
        transparent_background=True,
        bbox_extra_artists=None,
        dpi=200,
        show_and_save=False,
        verbose=False):
    """
    Show and/or save plot.

    Parameters
    ----------

    filename : str or bool, default=None
        If `False`, the plot is neither shown nor saved. If `True` or `None`,
        the plot is shown. If string, the plot is saved instead, where the
        filename is the given string. If the filename contains no file
        extension, the plot is saved as both ``svg`` and ``pdf`` format. If the
        filename contains no directory path, the plot is saved in the current
        directory.

    default_filename : str, default=None
        If the plot cannot be shown (such as when no graphical backend exists),
        the plot is saved instead with the default filename.

    transparent_background : bool, default=True
        If `True`, the figure and axes backgrounds will be rendered transparent
        in saved file. This only works for the file formats that support the
        alpha channel, such as ``svg``, ``pdf``, and ``png`` files.

    bbox_extra_artist : list, default=None
        A list of extra artists to pass to the renderer.

    dpi : int, default=200
        Dots per inch for rendering plots.

    show_and_save : bool, default=False
        By default, the plot is either shown xor saved. But when this argument
        is `True`, the plot is forced to both be shown and saved.

    verbose : bool, default=False
        If `True`, the fullpath filename of the saved plot is printed.

    Notes
    -----

    The ``filename`` argument determines that whether the plot should be shown
    of saved. However, there the following exceptions are made:

    1. If ``show_and_save`` is enabled, plot is both shown and saved.

    2. If no graphical beckend exists, it the plot is saved instead of being
       shown, even of it was not intended to be saved.

    2. If the environment is Jupyter notebook, the plot is always shown, even
       if it was not intended to be shown. If a filename is given, it is both
       shown and saved.
    """

    if filename is False:
        # Do nothing
        return
    elif (filename is True) or (filename is None):
        # Show the plot, but do not save it
        if matplotlib.get_backend() != 'agg':
            show = True
            if show_and_save:
                save = True
            else:
                save = False
        else:
            # There is no graphical backend, no other choice but to save it
            save = True
            if show_and_save:
                show = True
            else:
                show = False
    elif isinstance(filename, str):
        # Save the plot, but do not show it, unless it is a Jupyter notebook
        save = True
        if is_notebook():
            show = True
        else:
            show = False
    else:
        raise ValueError('"filename" should be boolean or a string.')

    # Determine filename
    if (save is True) and (not isinstance(filename, str)):
        if isinstance(default_filename, str):
            filename = default_filename
        else:
            raise NotImplementedError('"default_filename" is not set. ' +
                                      'This is a bug. Please report the ' +
                                      'issue.')

    # Save plot to file(s)
    if save:

        # write the plot as SVG file in the current working directory
        _save_plot(plt, filename,
                   transparent_background=transparent_background,
                   bbox_extra_artists=bbox_extra_artists, dpi=dpi,
                   verbose=verbose)

        # Closing is necessary especially if a large number of plots are saved.
        if not show:
            plt.close()

    # Show plot
    if show:
        plt.show()
