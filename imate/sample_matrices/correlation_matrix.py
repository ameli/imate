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
from ._generate_points import generate_points
from ._dense_correlation_matrix import dense_correlation_matrix
from ._sparse_correlation_matrix import sparse_correlation_matrix

try:
    from .._utilities.plot_utilities import matplotlib, plt
    from .._utilities.plot_utilities import load_plot_settings, save_plot
    plot_modules_exist = True
except ImportError:
    plot_modules_exist = False

__all__ = ['correlation_matrix']


# ==================
# correlation matrix
# ==================

def correlation_matrix(
        size=20,
        dimension=1,
        distance_scale=0.1,
        kernel='exponential',
        kernel_param=None,
        grid=True,
        sparse=False,
        density=0.001,
        dtype=r'float64',
        plot=False,
        verbose=False):
    """
    Generates symmetric and positive-definite matrix for test purposes.

    **Correlation Function:**

    The generated matrix is a correlation matrix based on Matern correlation of
    spatial distance of a list of points in the unit hypercube. The Matern
    correlation function accepts the correlation scale parameter
    :math:`\\rho \\in (0,1]`. Smaller decorrelation produces correlation matrix
    that is closer to the identity matrix.

    **Matrix Size:**

    The size of generated matrix is determined by the parameter ``num_points``
    which here we refer to as :math:`n`, the dimension, ``dimension`` which we
    refer to as :math:`d`, and ``grid`` boolean variable.

        * If ``grid`` is ``True`` (default value), then, the size of the square
          matrix is :math:`n^d`.
        * If ``grid`` is ``False``, then, the size of the square matrix is
          :math:`n`.

    **Sparsification:**

    The values of the correlation matrix are between :math:`0` and :math:`1`.
    To sparsify the matrix, the correlation kernel below a certain threshold
    value is set to zero to which tapers the correlation kernel. Such threshold
    can be set through the parameter ``density``, which sets an approximate
    density of the non-zero elements of the sparse matrix.

    .. note::

        Setting a too small ``density`` might eradicate the
        positive-definiteness of the correlation matrix.

    **Plotting:**

    If the option ``plot`` is set to ``True``, it plots the generated matrix.

    * If no graphical backend exists (such as running the code on a remote
      server or manually disabling the X11 backend), the plot will not be
      shown, rather, it will be saved as an ``svg`` file in the current
      directory.
    * If the executable ``latex`` is on the path, the plot is rendered using
      :math:`\\rm\\LaTeX`, which then, it takes longer to produce the plot.
    * If :math:`\\rm\\LaTeX` is not installed, it uses any available San-Serif
      font to render the plot.

   .. note::

       To manually disable interactive plot display, and save the plot as
       ``SVG`` instead, add the following in the very beginning of your code
       before importing ``imate``:

       .. code-block:: python

           >>> import os
           >>> os.environ['IMATE_NO_DISPLAY'] = 'True'

    :param size: The size of the generated matrix is determined as follows:
        * If ``grid`` is ``True``, the size of matrix is ``size**dimension``.
        * If ``grid`` is ``False``, the size of matrix is ``size``.
    :type size: int

    :param dimension: The dimension of the space of points to generate the
        correlation matrix.
    :type dimension: int

    :param distance_scale: A parameter of correlation function that scales
        distance.
    :type distance_scale: float

    :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
    :type nu: float

    :param grid: Determines if the generated set of points are on a structured
        grid or randomly generated.

        * If ``True``, the points are generated on a structured grid in
          a unit hypercube with equal distances. In this case, the size of
          generated matrix (which is equal to the number of points) is
        ``size**dimension``.
        * If ``False``, the spatial points are generated
          randomly. In this case, the size of the generated matrix is ``size``.
    :type grid: bool

    :param sparse: Flag to indicate the correlation matrix should be sparse or
        dense matrix. If set to ``True``, you may also specify ``density``.
    :type parse: bool

    :param density: Specifies an approximate density of the non-zero elements
        of the generated sparse matrix. The actual density of the matrix may
        not be exactly the same as this value.
    :rtype: double

    :param plot: If ``True``, the matrix will be plotted.
    :type Plot: bool

    :param verbose: If ``True``, prints some information during the process.
    :type verbose: bool

    :return: Correlation matrix.
    :rtype: numpy.ndarray or scipy.sparse.csc

    **Example:**

    Generate a matrix of the shape ``(20,20)`` by mutual correlation of a set
    of :math:`20` points in the unit interval:

    .. code-block:: python

       >>> from imate.sample_matrices import correlation_matrix
       >>> A = correlation_matrix(20)

    Generate a matrix of the shape :math:`(20^2, 20^2)` by mutual correlation
    of a grid of :math:`20 \\times 20` points in the unit square:

    .. code-block:: python

       >>> from imate.sample_matrices import correlation_matrix
       >>> A = correlation_matrix(20, dimension=20)

    correlation a correlation matrix of shape ``(20, 20)`` based on 20 random
    points in unit square:

    .. code-block:: python

       >>> A = correlation_matrix(size=20, dimension=20, grid=False)

    correlation a matrix of shape ``(20, 20)`` with spatial :math:`20` points
    that are more correlated:

    .. code-block:: python

       >>> A = correlation_matrix(size=20, distance_scale=0.3)

    Sparsify correlation matrix of size :math:`(20^2, 20^2)` with approximate
    density of :math:`1e-3`

    .. code-block:: python

       >>> A = correlation_matrix(size=20, dimension=2, sparse=True,
       ...                        density=1e-3)

    Plot a dense matrix of size :math:`(30^2, 30^2)` by

    .. code-block:: python

        >>> A = correlation_matrix(size=30, dimension=2, plot=True)
    """

    # Check input arguments
    _check_arguments(size, dimension, distance_scale, kernel,
                     kernel_param, grid, sparse, density, dtype, plot, verbose)

    # Default for kernel parameter
    if kernel_param is None:
        if kernel == 'matern':
            kernel_param = 0.5
        elif kernel == 'rational-quadratic':
            kernel_param = 1.0

    # Convert string to binary
    kernel = kernel.encode('utf-8')

    # correlation a set of points in the unit square
    coords = generate_points(size, dimension, grid)

    # Compute the correlation between the set of points
    if sparse:

        # Generate as sparse matrix
        correlation_matrix = sparse_correlation_matrix(
            coords,
            distance_scale,
            kernel,
            kernel_param,
            density,
            dtype,
            verbose)

    else:

        # Generate a dense matrix
        correlation_matrix = dense_correlation_matrix(
            coords,
            distance_scale,
            kernel,
            kernel_param,
            dtype,
            verbose)

    # Plot Correlation Matrix
    if plot:
        plot_matrix(correlation_matrix, sparse, verbose)

    return correlation_matrix


# ===============
# check arguments
# ===============

def _check_arguments(
        size,
        dimension,
        distance_scale,
        kernel,
        kernel_param,
        grid,
        sparse,
        density,
        dtype,
        plot,
        verbose):
    """
    Checks the type and values of the input arguments.
    """

    # Check size
    if size is None:
        raise TypeError('"size" cannot be None.')
    elif not numpy.isscalar(size):
        raise TypeError('"size" should be a scalar value.')
    elif not isinstance(size, (int, numpy.integer)):
        TypeError('"size" should be an integer.')
    elif size < 1:
        raise ValueError('"size" should be a positive integer.')

    # Check dimension
    if dimension is None:
        raise TypeError('"dimension" cannot be None.')
    elif not numpy.isscalar(dimension):
        raise TypeError('"dimension" should be a scalar value.')
    elif not isinstance(dimension, (int, numpy.integer)):
        TypeError('"dimension" should be an integer.')
    elif dimension < 1:
        raise ValueError('"dimension" should be a positive integer.')

    # Check distance_scale
    if distance_scale is None:
        raise TypeError('"distance_scale" cannot be None.')
    elif not numpy.isscalar(distance_scale):
        raise TypeError('"distance_scale" should be a scalar value.')
    elif isinstance(distance_scale, complex):
        TypeError('"distance_scale" should be a float number.')
    elif distance_scale <= 0.0:
        raise ValueError('"distance_scale" should be a positive number.')

    # Check kernel
    if not isinstance(kernel, str):
        raise TypeError('"kernel" should be a string.')
    elif kernel not in ['matern', 'exponential', 'square_exponential',
                        'rational_quadratic']:
        raise ValueError('"kernel" should be one of "matern", ' +
                         '"exponential", "square-exponential", or ' +
                         '"ratioanl_quadratic".')

    # Check kernel_param
    if kernel_param is not None:
        if not numpy.isscalar(kernel_param):
            raise TypeError('"kernel_param" should be a scalar value.')
        elif isinstance(kernel_param, complex):
            TypeError('"kernel_param" should be an float number.')
        elif kernel == 'exponental' and kernel_param is not None:
            raise ValueError('When "kernel" is "exponential", ' +
                             '"kernel_param" should be "None".')
        elif kernel == 'square-exponental' and kernel_param is not None:
            raise ValueError('When "kernel" is "-square-exponential", ' +
                             '"kernel_param" should be "None".')

    # Check grid
    if grid is None:
        raise TypeError('"grid" cannot be None.')
    elif not numpy.isscalar(grid):
        raise TypeError('"grid" should be a scalar value.')
    elif not isinstance(grid, bool):
        TypeError('"grid" should be boolean.')

    # Check sparse
    if sparse is None:
        raise TypeError('"sparse" cannot be None.')
    elif not numpy.isscalar(sparse):
        raise TypeError('"sparse" should be a scalar value.')
    elif not isinstance(sparse, bool):
        TypeError('"sparse" should be boolean.')

    # Check density
    if density is None:
        raise TypeError('"density" cannot be None.')
    elif not numpy.isscalar(density):
        raise TypeError('"density" should be a scalar value.')
    elif isinstance(density, complex):
        TypeError('"density" should be a float number.')
    elif density <= 0.0 or density >= 1.0:
        raise ValueError('"density" hshould be between "0.0" and "1.0".')

    # Check dtype
    if dtype is None:
        raise TypeError('"dtype" cannot be None.')
    elif not numpy.isscalar(dtype):
        raise TypeError('"dtype" should be a scalar value.')
    elif not isinstance(dtype, str):
        raise TypeError('"dtype" should be a string')
    elif dtype not in [r'float32', r'float64', r'float128']:
        raise TypeError('"dtype" should be either "float32", "float64", or ' +
                        '"float128".')

    # Check plot
    if plot is None:
        raise TypeError('"plot" cannot be None.')
    elif not numpy.isscalar(plot):
        raise TypeError('"plot" should be a scalar value.')
    elif not isinstance(plot, bool):
        TypeError('"plot" should be boolean.')

    # Check if plot modules exist
    if plot is True:
        try:
            from .._utilities.plot_utilities import matplotlib      # noqa F401
            from .._utilities.plot_utilities import load_plot_settings
            load_plot_settings()
        except ImportError:
            raise ImportError('Cannot import modules for plotting. Either ' +
                              'install "matplotlib" and "seaborn" packages, ' +
                              'or set "plot=False".')

    # Check verbose
    if verbose is None:
        raise TypeError('"verbose" cannot be None.')
    elif not numpy.isscalar(verbose):
        raise TypeError('"verbose" should be a scalar value.')
    elif not isinstance(verbose, bool):
        TypeError('"verbose" should be boolean.')


# ===========
# plot Matrix
# ===========

def plot_matrix(matrix, sparse, verbose=False):
    """
    Plots a given matrix.

    If the matrix is a sparse, it plots all non-zero elements with single
    color regardless of their values, and leaves the zero elements white.

    Whereas, if the matrix is not a sparse matrix, the colormap of the plot
    correspond to the value of the elements of the matrix.

    If a graphical backend is not provided, the plot is not displayed,
    rather saved as ``SVG`` file in the current directory of user.

    :param matrix: A 2D array
    :type matrix: numpy.ndarray or scipy.sparse.csc_matrix

    :param sparse: Determine whether the matrix is dense or sparse
    :type sparse: bool

    :param verbose: If ``True``, prints some information during the process.
    :type verbose: bool
    """

    if not plot_modules_exist:
        raise ImportError('Cannot import modules for plotting. Either ' +
                          'install "matplotlib" and "seaborn" packages, ' +
                          'or set "plot=False".')

    # Load plot settings
    try:
        load_plot_settings()
    except ImportError:
        raise ImportError('Cannot import modules for plotting. Either ' +
                          'install "matplotlib" and "seaborn" packages, ' +
                          'or set "plot=False".')

    # Figure
    fig, ax = plt.subplots(figsize=(6, 4))

    if sparse:
        # Plot sparse matrix
        p = ax.spy(matrix, markersize=1, color='blue', rasterized=True)
    else:
        # Plot dense matrix
        p = ax.matshow(matrix, cmap='Blues')
        cbar = fig.colorbar(p, ax=ax)
        cbar.set_label('Correlation')

    ax.set_title('Correlation Matrix', y=1.11)
    ax.set_xlabel('Index $i$')
    ax.set_ylabel('Index $j$')

    plt.tight_layout()

    # Check if the graphical backend exists
    if matplotlib.get_backend() != 'agg':
        plt.show()
    else:
        # write the plot as SVG file in the current working directory
        save_plot(plt, 'CorrelationMatrix', transparent_background=True)
