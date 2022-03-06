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

import numpy
from numbers import Number

# Classes, Files
from ._exact_method import ExactMethod
from ._eigenvalues_method import EigenvaluesMethod
from ._monomial_basis_functions_method import MonomialBasisFunctionsMethod
from ._root_monomial_basis_functions_method import \
        RootMonomialBasisFunctionsMethod
from ._radial_basis_functions_method import RadialBasisFunctionsMethod
from ._rational_polynomial_functions_method import \
        RationalPolynomialFunctionsMethod

try:
    from .._utilities.plot_utilities import *                # noqa: F401, F403
    from .._utilities.plot_utilities import load_plot_settings, matplotlib, \
        show_or_save_plot, plt
    plot_modules_exist = True
except ImportError:
    plot_modules_exist = False


# ====================
# Interpolate Traceinv
# ====================

class InterpolateTraceinv(object):
    """
    Interpolates the trace of inverse of affine matrix functions.

    An affine matrix function is defined by

    .. math::

        t \\mapsto \\mathbf{A} + t \\mathbf{B}

    where :math:`\\mathbf{A}` and :math:`\\mathbf{B}` are invertible matrices
    and :math:`t` is a real parameter.

    This module interpolates the function

    .. math::

        t \\mapsto \\mathrm{trace} \\left( (\\mathbf{A} +
        t \\mathbf{B})^{-1} \\right)

    The interpolator is initialized by providing :math:`p` interpolant points
    :math:`t_i`, :math:`i = 1, \\dots, p`, which are often logarithmically
    spaced in some interval :math:`t_i \\in [t_1, t_p]`.
    The interpolator can interpolate the above function at inquiry points
    :math:`t \\in [t_1, t_p]` using various methods.

    Parameters
    ----------
        A : numpy.ndarray, scipy.sparse matrix
            A positive-definite matrix. Matrix can be dense or sparse.

        B : numpy.ndarray, scipy.sparse matrix, default=None
            A positive-definite matrix. Matrix can be dense or sparse. If
            `None` or not provided, it is assumed that `B` is an identity
            matrix of the shape of `A`.

        interpolant_points : list(float), numpy.array(float), default=None
            A list or an array of points that the interpolator use to
            interpolate. The trace of inverse is computed for the interpolant
            points with exact method. If `None`, a default list of interpolant
            points is used.

        method : str, default=`'RMBF'`
            Algorithm of interpolation. See table below.

            ==========  ==============================  ============
            `method`    Description                     Results
            ==========  ==============================  ============
            ``'EXT'``   Computes without interpolation  exact
            ``'EIG'``   Uses Eigenvalues of matrix      exact
            ``'MBF'``   Monomial Basis Functions        interpolated
            ``'RMBF'``  Root monomial basis functions   interpolated
            ``'RBF'``   Radial basis functions          interpolated
            ``'RPF'``   Rational polynomial functions   interpolated
            ==========  ==============================  ============

        traceinv_options : dict, default={'method': 'cholesky'}
            A dictionary of arguments to pass to :mod:`imate.traceinv` module.

        verbose : bool, default=False
            If `True`, it prints some information on the computation process.

        interpolation_options : \\*\\*kwargs
            Additional options to pass to each specific interpolator class.
            See the sub-classes of :mod:`imate.InterpolantBase`.

    Attributes
    ----------
    method : str
        Method of interpolation
    verbose : bool
        Verbosity of the computation process.
    n : int
        Since of the matrix
    p : int
        number of interpolant points

    Methods
    -------
    __call__
    compute
    compare_with_exact_solution
    interpolate
    lower_bound
    upper_bound
    plot_interpolation

    Examples
    --------

    Interpolate the trace of inverse of the affine matrix function
    :math:`\\mathbf{A} + t \\mathbf{B}`:

    .. code-block:: python

        >>> # Generate two sample matrices (symmetric and positive-definite)
        >>> from imate import generate_matrix
        >>> A = generate_matrix(size=20, correlation_scale=1e-1)
        >>> B = generate_matrix(size=20, correlation_scale=2e-2)

        >>> # Initialize interpolator object
        >>> from imate import InterpolateTraceinv
        >>> interpolant_points = [1e-2, 1e-1, 1, 1e1]
        >>> TI = InterpolateTraceinv(A, B, interpolant_points)

        >>> # Interpolate at an inquiry point t = 0.4
        >>> t = 4e-1
        >>> Trace = TI.interpolate(t)

    Interpolate an array of inquiry points

    .. code-block:: python

        >>> # Interpolate at an array of points
        >>> import numpy
        >>> t_array = numoy.logspace(-2, 1, 10)
        >>> Trace = TI.interpolator(t_array)

    By default, the interpolation method is ``'RMBF'``. Use a different
    interpolation method, such as ``'RBF'`` by

    .. code-block:: python

        >>> TI = InterpolateTraceinv(A, B, interpolant_points, method='RBF')

    By default, the trace is computed with the Cholesky decomposition method as
    the interpolant points. Configure the
    computation method by ``traceinv_options`` as

    .. code-block:: python

        >>> # Specify arguments to imate.omputeTraceOfInverse in a dictionary
        >>> traceinv_options = \
        ... {
        ...     'ComputeMethod': 'hutchinson',
        ...     'NumIterations': 20
        ... }

        >>> # Pass the options to the interpolator
        >>> TI = InterpolateTraceinv(A, B, interpolant_points,
        >>>     traceinv_options=traceinv_options)

    See Also
    --------
        traceinv : Computes trace of inverse of a matrix.

    """

    # ====
    # init
    # ====

    def __init__(self, A, B=None, interpolant_points=None, method='RMBF',
                 traceinv_options={'method': 'cholesky'}, verbose=False,
                 **interpolation_options):
        """
        Initializes the object depending on the method.
        """

        # Attributes
        self.method = method
        self.verbose = verbose
        self.n = A.shape[0]
        if interpolant_points is not None:
            self.p = len(interpolant_points)
        else:
            self.p = 0

        # Define an interpolation object depending on the given method
        if method == 'EXT':
            # Exact computation, not interpolation
            self.interpolator = ExactMethod(
                    A, B, traceinv_options=traceinv_options, verbose=verbose,
                    **interpolation_options)

        elif method == 'EIG':
            # Eigenvalues method
            self.interpolator = EigenvaluesMethod(
                    A, B, traceinv_options=traceinv_options, verbose=verbose,
                    **interpolation_options)

        elif method == 'MBF':
            # Monomial Basis Functions method
            self.interpolator = MonomialBasisFunctionsMethod(
                    A, B, interpolant_points,
                    traceinv_options=traceinv_options, verbose=verbose,
                    **interpolation_options)

        elif method == 'RMBF':
            # Root Monomial Basis Functions method
            self.interpolator = RootMonomialBasisFunctionsMethod(
                    A, B, interpolant_points,
                    traceinv_options=traceinv_options, verbose=verbose,
                    **interpolation_options)

        elif method == 'RBF':
            # Radial Basis Functions method
            self.interpolator = RadialBasisFunctionsMethod(
                    A, B, interpolant_points,
                    traceinv_options=traceinv_options, verbose=verbose,
                    **interpolation_options)

        elif method == 'RPF':
            # Rational Polynomial Functions method
            self.interpolator = RationalPolynomialFunctionsMethod(
                    A, B, interpolant_points,
                    traceinv_options=traceinv_options, verbose=verbose,
                    **interpolation_options)

        else:
            raise ValueError("'method' is invalid. Select one of 'EXT', " +
                             "'EIG', 'MBF', 'RMBF', 'RBF', or 'RPF'.")

    # ========
    # __call__
    # ========

    def __call__(self, t, compare_with_exact=False, plot=False):
        """
        Same as :func:`InterpolateTraceinv.interpolate` method.
        """

        return self.interpolate(t, compare_with_exact, plot)

    # =======
    # compute
    # =======

    def compute(self, t):
        """
        Computes the function :math:`\\mathrm{trace}\\left( (\\mathbf{A} +
        t \\mathbf{B})^{-1} \\right)` at the input point :math:`t` using exact
        method.

        The computation method used in this function is exact (no
        interpolation). This function  is primarily used to compute traceinv on
        the *interpolant points*.

        Parameters
        ----------
            t : float or numpy.array
                An inquiry point, which can be a single number, or an array of
                numbers.

        Returns
        -------
            traceinv : float or numpy.array
                The exact value of the traceinv function.
        """

        if isinstance(t, Number):
            # Single number
            T = self.interpolator.compute(t)

        else:
            # An array of points
            T = numpy.empty((len(t), ), dtype=float)
            for i in range(len(t)):
                T[i] = self.interpolator.compute(t[i])

        return T

    # ===========================
    # compare with exact solution
    # ===========================

    def compare_with_exact_solution(self, t, Trace):
        """
        Computes the trace with exact method (no interpolation), then compares
        it with the interpolated solution.

        Parameters
        ----------
            t : numpy.array
                Inquiry points

            Trace: float or numpy.array
                The interpolated computation of trace.

        Returns
        -------
            exact : float or numpy.array
                Exact solution of trace.

            relative_error : float or numpy.array
                Relative error of interpolated solution compared to the exact
                solution.
        """

        if self.method == 'EXT':

            # The Trace results are already exact. No need to recompute again.
            trace_exact = Trace
            trace_relative_error = numpy.zeros(t.shape)

        else:

            # Compute exact solution
            trace_exact = self.compute(t)
            trace_relative_error = (Trace - trace_exact) / (trace_exact)

        return trace_exact, trace_relative_error

    # ===========
    # interpolate
    # ===========

    def interpolate(self, t, compare_with_exact=False, plot=False):
        """
        Interpolates :math:`\\mathrm{trace} \\left( (\\mathbf{A} +
        t \\mathbf{B})^{-1} \\right)` at :math:`t`.

        Parameters
        ----------
        t : float, list, numpy.array
            The inquiry point(s) to be interpolated.

        compare_with_exact : bool, default=False
            If `True`, it computes the trace with exact solution, then compares
            it with the interpolated solution. The return values of the
            :func:`InterpolateTraceinv.interpolate` function become
            interpolated trace, exact solution, and relative error. **Note:**
            When this option is enabled, the exact solution will be computed
            for all inquiry points, which can take a very long time.

        plot : bool, default=False
            If `True`, it plots the interpolated trace versus the inquiry
            points. In addition, if the option `compare_with_exact` is also set
            to `True`, the plotted diagram contains both interpolated and exact
            solutions and the relative error of interpolated solution with
            respect to the exact solution.

        Returns
        -------
        trace : float or numpy.array
            The interpolated value of the traceinv function.

        Notes
        -----

        **Plotting:**

        Regarding the plotting of the graph of interpolation:

            * If no graphical backend exists (such as running the code on a
              remote server or manually disabling the X11 backend), the plot
              will not be shown, rather, it will ve saved as an ``svg`` file in
              the current directory.
            * If the executable ``latex`` is on the path, the plot is rendered
              using :math:`\rm\\laTeX`, which then, it takes a bit
              longer to produce the plot.
            * If :math:`\rm\\laTeX` is not installed, it uses any available
              San-Serif font to render the plot.

        To manually disable interactive plot display, and save the plot as
        ``SVG`` instead, add the following in the very begining of your code
        before importing ``imate``:

        .. code-block:: python

            >>> import os
            >>> os.environ['IMATE_NO_DISPLAY'] = 'True'
        """

        if isinstance(t, Number):
            # Single number
            trace = self.interpolator.interpolate(t)

        else:
            # An array of points
            trace = numpy.empty((len(t), ), dtype=float)
            for i in range(len(t)):
                trace[i] = self.interpolator.interpolate(t[i])

        # Compare with exact solution
        if compare_with_exact:

            # Since this method is exact, no need to compute exact again.
            trace_exact, trace_relative_error = \
                    self.compare_with_exact_solution(t, trace)

        # Plot
        if plot:
            if compare_with_exact:
                self.plot_interpolation(t, trace, trace_exact,
                                        trace_relative_error)
            else:
                self.plot_interpolation(t, trace)

        # Return
        if compare_with_exact:
            return trace, trace_exact, trace_relative_error
        else:
            return trace

    # ===========
    # lower bound
    # ===========

    def lower_bound(self, t):
        """
        Lower bound of the function :math:`t \\mapsto \\mathrm{trace} \\left(
        (\\mathbf{A} + t \\mathbf{B})^{-1} \\right)`.

        The lower bound is given by

        .. math::

            \\mathrm{trace}((\\mathbf{A}+t\\mathbf{B})^{-1}) \\geq
            \\frac{n^2}{\\mathrm{trace}(\\mathbf{A}) +
            \\mathrm{trace}(t \\mathbf{B})}

        Parameters
        ----------
        t : float or numpy.array
            An inquiry point or an array of inquiry points.

        Returns
        -------
            lb : float or numpy.array
                Lower bound of the affine matrix function.
        """

        if isinstance(t, Number):
            # Single number
            T_lb = self.interpolator.lower_bound(t)

        else:
            # An array of points
            T_lb = numpy.empty((len(t), ), dtype=float)
            for i in range(len(t)):
                T_lb[i] = self.interpolator.lower_bound(t[i])

        return T_lb

    # ===========
    # upper bound
    # ===========

    def upper_bound(self, t):
        """
        Upper bound of the function :math:`t \\mapsto \\mathrm{trace} \\left(
        (\\mathbf{A} + t \\mathbf{B})^{-1} \\right)`.

        The upper bound is given by

        .. math::

                \\frac{1}{\\tau(t)} \\geq \\frac{1}{\\tau_0} + t

        where

        .. math::

                \\tau(t) = \\frac{\\mathrm{trace}\\left( (\\mathbf{A} +
                t \\mathbf{B})^{-1}
                \\right)}{\\mathrm{trace}(\\mathbf{B}^{-1})}

        and :math:`\\tau_0 = \\tau(0)`.

        Parameters
        ----------
        t : float or numpy.array
            An inquiry point or an array of inquiry points.

        Returns
        -------
        ub : float or numpy.array
            bound of the affine matrix function.
        """

        if isinstance(t, Number):
            # Single number
            T_ub = self.interpolator.upper_bound(t)

        else:
            # An array of points
            T_ub = numpy.empty((len(t), ), dtype=float)
            for i in range(len(t)):
                T_ub[i] = self.interpolator.upper_bound(t[i])

        return T_ub

    # ====
    # plot
    # ====

    def plot(
            self,
            inquiry_points,
            trace_interpolated=None,
            compare=False,
            trace_exact=None,
            trace_relative_error=None):
        """
        Plots the interpolation results, together with the comparison with the
        exact solution and the relative error of the interpolation.

        To plot, set ``Plot=True`` argument in
        :mod:`imate.InterpolateTraceinv`.

        Parameters
        ----------
        inquiry_points: numpy.array
            Inquiry points to be interpolated

        trace_interpolated : numpy.array
            Interpolation of the trace at inquiry

        trace_exact : numpy.array, default=None
            Exact solutions of the trace at inquiry points. If this variable is
            not None, it will be plotted together with the interpolated
            results.

        trace_relative_error : numpy.array, default=None
            Relative errors of the interpolation with respect to the exact
            solution. If not None, the relative errors will be plotted on a
            second axis.
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

        # Check t should be an array
        if numpy.isscalar(inquiry_points) or (inquiry_points.size == 1):
            raise ValueError("Argument 'inquiry_points' should be an " +
                             "array of length greater than one to be able " +
                             " to plot results.")

        # If no data is provided, generate interpolation
        if trace_interpolated is None:
            trace_interpolated = self.interpolate(inquiry_points)

        if compare:
            if trace_exact is None:
                trace_exact = self.compute(inquiry_points)

            if trace_relative_error is None:
                trace_relative_error = (trace_interpolated / trace_exact) - 1.0

        # Plot results
        if trace_relative_error is None:
            # One subplot
            fig, ax = plt.subplots(figsize=(5, 4))
            ax = [ax]
        else:
            # Two subplots
            fig, ax = plt.subplots(ncols=2, figsize=(9, 4))

        # Plot settings
        markersize = 4
        exact_color = 'firebrick'
        interp_color = 'black'

        # Normalize traceinv to tau
        tau_i = self.interpolator.trace_i / self.interpolator.trace_Binv
        tau_interpolated = trace_interpolated / self.interpolator.trace_Binv
        if trace_exact is not None:
            tau_exact = trace_exact / self.interpolator.trace_Binv

        # Plot interpolant points with their exact values
        if self.interpolator.p > 0:
            ax[0].loglog(self.interpolator.t_i, tau_i, 'o', color=exact_color,
                         markersize=markersize, label='Interpolant points',
                         zorder=20)

        # Plot exact values
        if trace_exact is not None:
            ax[0].semilogx(inquiry_points, tau_exact, color=exact_color,
                           label='Exact')

        # Plot interpolated results
        ax[0].semilogx(inquiry_points, tau_interpolated, color=interp_color,
                       label='Interpolated')

        ax[0].grid(axis='x')
        ax[0].set_xlim([inquiry_points[0], inquiry_points[-1]])
        ax[0].set_xlabel(r'$t$')
        if self.interpolator.B_is_identity:
            if matplotlib.rcParams['text.usetex'] is True:
                ax[0].set_ylabel(r'$\frac{1}{n}\mathrm{trace}(\mathbf{A} + ' +
                                 r't \mathbf{I})^{-1}$')
            else:
                ax[0].set_ylabel(r'$\frac{1}{n}\mathrm{trace}(\mathbf{A} + ' +
                                 r't \mathbf{I})^{-1}$')
        else:
            if matplotlib.rcParams['text.usetex'] is True:
                ax[0].set_ylabel(r'$\frac{\mathrm{trace}(\mathbf{A} + t ' +
                                 r'\mathbf{B})^{-1}}{\mathrm{trace}(' +
                                 r'\mathbf{B})^{-1}}$')
            else:
                ax[0].set_ylabel(r'$\frac{\mathrm{trace}(\mathbf{A} + t ' +
                                 r'\mathbf{B})^{-1}}{\mathrm{trace}(' +
                                 r'\mathbf{B})^{-1}}$')
        if trace_relative_error is not None:
            ax[0].set_title('(a) Comparison of Interpolated and Exact ' +
                            'Function')
        else:
            ax[0].set_title('Interpolation')
        ax[0].legend(fontsize='small')

        # Plot relative error in percent
        if trace_relative_error is not None:
            if self.interpolator.p > 0:
                ax[1].semilogx(self.interpolator.t_i,
                               numpy.zeros(self.interpolator.p), 'o',
                               color=exact_color, markersize=markersize,
                               label='Interpolant points', zorder=20)
            ax[1].semilogx(inquiry_points, 100.0*trace_relative_error,
                           color=interp_color, label='Interpolated')
            ax[1].grid(axis='x')
            ax[1].semilogx(ax[1].get_xlim(), [0, 0], color='#CCCCCC',
                           linewidth=0.75)
            ax[1].set_xlim([inquiry_points[0], inquiry_points[-1]])
            ax[1].set_xlabel('$t$')
            ax[1].set_ylabel('Relative Error of Interpolation')
            ax[1].set_title('(b) Relative Error')
            ax[1].yaxis.set_major_formatter(
                    matplotlib.ticker.PercentFormatter(decimals=2))
            ax[1].legend(fontsize='small')

        plt.tight_layout()

        # Check if the graphical backend exists
        if matplotlib.get_backend() != 'agg':
            plt.show()
        else:
            # Save the plot as SVG file in the current directory
            show_or_save_plot(plt, 'interpolation',
                      transparent_background=True)
