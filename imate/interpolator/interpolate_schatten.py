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
from ._inverse_monomial_basis_functions_method import \
        InverseMonomialBasisFunctionsMethod
from ._radial_basis_functions_method import RadialBasisFunctionsMethod
from ._rational_polynomial_functions_method import \
        RationalPolynomialFunctionsMethod
from ._chebyshev_rational_functions_method import \
        ChebyshevRationalFunctionsMethod
from ._spline_method import SplineMethod

try:
    from .._utilities.plot_utilities import *                # noqa: F401, F403
    from .._utilities.plot_utilities import load_plot_settings, matplotlib, \
        show_or_save_plot, plt
    plot_modules_exist = True
except ImportError:
    plot_modules_exist = False


# ====================
# Interpolate Schatten
# ====================

class InterpolateSchatten(object):
    """
    Interpolates the Schatten norm, or anti-norm, of an affine matrix function.

    The Shcatten-type operator of order :math:`p \\in \\mathbb{R}` of an affine
    matrix function :math:`t \\mapsto \\mathbf{A} + t \\mathbf{B}` is defined
    by

    .. math::

        t \\mapsto \\| \\mathbf{A} + t \\mathbf{B} \\|_p

    where the matrices :math:`\\mathbf{A}` and :math:`\\mathbf{B}` are
    Hermitian and positive semi-definite (positive-definite if :math:`p < 0`)
    and :math:`t` is a real parameter. The above

    This module interpolates the function

    .. math::

        t \\mapsto \\mathrm{trace} \\left( (\\mathbf{A} +
        t \\mathbf{B})^{-1} \\right)

    The interpolator is initialized by providing :math:`q` interpolant points
    :math:`t_i`, :math:`i = 1, \\dots, q`, which are often logarithmically
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

        p : float , default=0
            Power of matrix.

        ti : list(float), numpy.array(float), default=None
            A list or an array of points that the interpolator use to
            interpolate. The trace of inverse is computed for the interpolant
            points with exact method. If `None`, a default list of interpolant
            points is used.

        method : str, default=`'IMBF'`
            Algorithm of interpolation. See table below.

            ==========  ================================  ============
            `kind`      Description                       Results
            ==========  ================================  ============
            ``'EXT'``   Computes without interpolation    exact
            ``'EIG'``   Uses Eigenvalues of matrix        exact
            ``'MBF'``   Monomial Basis Functions          interpolated
            ``'IMBF'``  Inverse monomial basis functions  interpolated
            ``'RBF'``   Radial basis functions            interpolated
            ``'RPF'``   Rational polynomial functions     interpolated
            ==========  ================================  ============

        options : dict, default={'method': 'cholesky'}
            A dictionary of arguments to pass to :mod:`imate.traceinv` module.

        verbose : bool, default=False
            If `True`, it prints some information on the computation process.

        kwargs : \\*\\*kwargs
            Additional options to pass to each specific interpolator class.
            See the sub-classes of :mod:`imate.InterpolantBase`.

    Attributes
    ----------
    kind : str
        Method of interpolation
    verbose : bool
        Verbosity of the computation process.
    n : int
        Since of the matrix
    q : int
        number of interpolant points

    Methods
    -------
    __call__
    eval
    interpolate
    lower_bound
    upper_bound
    plot

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
        >>> from imate import InterpolateSchatten
        >>> ti = [1e-2, 1e-1, 1, 1e1]
        >>> TI = InterpolateSchatten(A, B, ti)

        >>> # Interpolate at an inquiry point t = 0.4
        >>> t = 4e-1
        >>> Trace = TI.interpolate(t)

    Interpolate an array of inquiry points

    .. code-block:: python

        >>> # Interpolate at an array of points
        >>> import numpy
        >>> t_array = numoy.logspace(-2, 1, 10)
        >>> Trace = TI.interpolator(t_array)

    By default, the interpolation kind is ``'IMBF'``. Use a different
    interpolation kind, such as ``'RBF'`` by

    .. code-block:: python

        >>> TI = InterpolateSchatten(A, B, ti, kind='RBF')

    By default, the trace is computed with the Cholesky decomposition method as
    the interpolant points. Configure the
    computation method by ``options`` as

    .. code-block:: python

        >>> # Specify arguments to imate.omputeTraceOfInverse in a dictionary
        >>> options = \
        ... {
        ...     'method': 'hutchinson',
        ...     'NumIterations': 20
        ... }

        >>> # Pass the options to the interpolator
        >>> TI = InterpolateSchatten(A, B, ti,
        >>>     options=options)

    See Also
    --------
        traceinv : Computes trace of inverse of a matrix.

    """

    # ====
    # init
    # ====

    def __init__(self, A, B=None, p=0, ti=[], kind='IMBF', verbose=False,
                 options={'method': 'cholesky'}, **kwargs):
        """
        Initializes the object depending on the method.
        """

        # Attributes
        self.kind = kind
        self.verbose = verbose
        self.n = A.shape[0]
        if ti is not None and not numpy.isscalar(ti):
            self.q = len(ti)
        else:
            self.q = 0
        self.p = p

        # Define an interpolation object depending on the given method
        if kind == 'EXT':
            # Exact computation, not interpolation
            self.interpolator = ExactMethod(
                    A, B, p=self.p, options=options, verbose=verbose, **kwargs)

        elif kind == 'EIG':
            # Eigenvalues kind
            self.interpolator = EigenvaluesMethod(
                    A, B, p=self.p, options=options, verbose=verbose, **kwargs)

        elif kind == 'MBF':
            # Monomial Basis Functions kind
            self.interpolator = MonomialBasisFunctionsMethod(
                    A, B, p=self.p, ti=ti, options=options, verbose=verbose,
                    **kwargs)

        elif kind == 'IMBF':
            # Inverse Monomial Basis Functions kind
            self.interpolator = InverseMonomialBasisFunctionsMethod(
                    A, B, p=self.p, ti=ti, options=options, verbose=verbose,
                    **kwargs)

        elif kind == 'RBF':
            # Radial Basis Functions kind
            self.interpolator = RadialBasisFunctionsMethod(
                    A, B, p=self.p, ti=ti, options=options, verbose=verbose,
                    **kwargs)

        elif kind == 'CRF':
            # Chebushev Rational Functions
            self.interpolator = ChebyshevRationalFunctionsMethod(
                    A, B, p=self.p, ti=ti, options=options, verbose=verbose,
                    **kwargs)

        elif kind == 'SPL':
            # Spline
            self.interpolator = SplineMethod(
                    A, B, p=self.p, ti=ti, options=options, verbose=verbose,
                    **kwargs)

        elif kind == 'RPF':
            # Rational Polynomial Functions kind
            self.interpolator = RationalPolynomialFunctionsMethod(
                    A, B, p=self.p, ti=ti, options=options, verbose=verbose,
                    **kwargs)

        else:
            raise ValueError("'kind' is invalid. Select one of 'EXT', " +
                             "'EIG', 'MBF', 'IMBF', 'RBF', 'CRF', 'SPL', or " +
                             "'RPF'.")

    # ========
    # __call__
    # ========

    def __call__(self, t):
        """
        Same as :func:`InterpolateSchatten.interpolate` method.
        """

        return self.interpolate(t)

    # ====
    # eval
    # ====

    def eval(self, t):
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
            T = self.interpolator.eval(t)

        else:
            # An array of points
            T = numpy.empty((len(t), ), dtype=float)
            for i in range(len(t)):
                T[i] = self.interpolator.eval(t[i])

        return T

    # ===========================
    # compare with exact solution
    # ===========================

    def _compare_with_exact_solution(self, t, schatten):
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

        if self.kind == 'EXT':

            # The Trace results are already exact. No need to recompute again.
            schatten_exact = schatten
            schatten_relative_error = numpy.zeros(t.shape)

        else:

            # Compute exact solution
            schatten_exact = self.eval(t)
            schatten_relative_error = 1.0 - (schatten / schatten_exact)

        return schatten_exact, schatten_relative_error

    # ===========
    # interpolate
    # ===========

    def interpolate(self, t):
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
            :func:`InterpolateSchatten.interpolate` function become
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

        **Plotting:** Regarding the plotting of the graph of interpolation:

        * If no graphical backend exists (such as running the code on a
          remote server or manually disabling the X11 backend), the plot
          will not be shown, rather, it will ve saved as an ``svg`` file in
          the current directory.
        * If the executable ``latex`` is on the path, the plot is rendered
          using :math:`\\rm\\laTeX`, which then, it takes a bit
          longer to produce the plot.
        * If :math:`\\rm\\laTeX` is not installed, it uses any available
          San-Serif font to render the plot.

        To manually disable interactive plot display, and save the plot as
        ``SVG`` instead, add the following in the very beginning of your code
        before importing ``imate``:

        .. code-block:: python

            >>> import os
            >>> os.environ['IMATE_NO_DISPLAY'] = 'True'
        """

        if isinstance(t, Number):
            # Single number
            schatten = self.interpolator.interpolate(t)

        else:
            # An array of points
            schatten = numpy.empty((len(t), ), dtype=float)
            for i in range(len(t)):
                schatten[i] = self.interpolator.interpolate(t[i])

        return schatten

    # =====
    # bound
    # =====

    def bound(self, t):
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
            schatten_bound = self.interpolator.bound(t)

        else:
            # An array of points
            schatten_bound = numpy.empty((len(t), ), dtype=float)
            for i in range(len(t)):
                schatten_bound[i] = self.interpolator.bound(t[i])

        return schatten_bound

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
            schatten_ub = self.interpolator.upper_bound(t)

        else:
            # An array of points
            schatten_ub = numpy.empty((len(t), ), dtype=float)
            for i in range(len(t)):
                schatten_ub[i] = self.interpolator.upper_bound(t[i])

        return schatten_ub

    # ====
    # plot
    # ====

    def plot(
            self,
            inquiry_points,
            normalize=True,
            compare=False):
        """
        Plots interpolated function.
        """

        if self.kind in ['CRF', 'SPL']:
            # Plots tau where abscissa is the finite domain [-1, 1]
            self._plot_finite(
                    inquiry_points, normalize=normalize, compare=compare)
        else:
            # Plots tau where abscissa is the semi-infinite domain [0, inf)
            self._plot_semi_infinite(
                    inquiry_points, normalize=normalize, compare=compare)

    # ==================
    # plot semi infinite
    # ==================

    def _plot_semi_infinite(
            self,
            inquiry_points,
            normalize=True,
            compare=False):
        """
        Plots the interpolation results, together with the comparison with the
        exact solution and the relative error of the interpolation.

        To plot, set ``Plot=True`` argument in
        :mod:`imate.InterpolateSchatten`.

        Parameters
        ----------
        inquiry_points: numpy.array
            Inquiry points to be interpolated
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

        # Generate interpolation
        schatten_interpolated = self.interpolate(inquiry_points)

        if compare:
            schatten_exact = self.eval(inquiry_points)

        # Normalize schatten to tau
        if normalize:
            schatten_B = self.interpolator.schatten_B

            # EXT and EIG methods do not compute schatten_B by default.
            if schatten_B is None:
                schatten_B = self.interpolator._compute_schatten(
                        self.interpolator.B, self.interpolator.p)

            normal_factor = schatten_B
        else:
            normal_factor = 1.0

        if self.interpolator.schatten_i is not None:
            tau_i = self.interpolator.schatten_i / normal_factor
        tau_interpolated = schatten_interpolated / normal_factor

        if compare:
            tau_exact = schatten_exact / normal_factor
            tau_relative_error = 1.0 - (tau_interpolated / tau_exact)

        # Plot results
        if compare:
            # Two subplots
            fig, ax = plt.subplots(ncols=2, figsize=(9, 4))
        else:
            # One subplot
            fig, ax = plt.subplots(figsize=(5, 4))
            ax = [ax]

        # Plot settings
        markersize = 4
        exact_color = 'firebrick'
        interp_color = 'black'

        # Plot interpolant points with their exact values
        if self.interpolator.q > 0:
            if self.interpolator.schatten_i is not None:
                ax[0].loglog(self.interpolator.t_i, tau_i, 'o',
                             color=exact_color, markersize=markersize,
                             label='Interpolant points', zorder=20)

        # Plot exact values
        if compare:
            ax[0].loglog(inquiry_points, tau_exact, color=exact_color,
                         label='Exact')

        # Plot interpolated results
        ax[0].loglog(inquiry_points, tau_interpolated, color=interp_color,
                     label='Interpolated')

        if compare:
            tau_min = numpy.min([tau_exact, tau_interpolated])
            tau_max = numpy.max([tau_exact, tau_interpolated])
        else:
            tau_min = numpy.min([tau_interpolated])
            tau_max = numpy.max([tau_interpolated])
        tau_min_snap = 10**(numpy.floor(numpy.log10(tau_min)))
        tau_max_snap = 10**(numpy.round(numpy.log10(tau_max)))

        ax[0].grid(axis='x')
        ax[0].set_xlim([inquiry_points[0], inquiry_points[-1]])
        ax[0].set_ylim([tau_min_snap, tau_max_snap])
        ax[0].set_xlabel(r'$t$')

        if normalize:
            ax[0].set_ylabel(r'$\tau_p(t)$')
        else:
            ax[0].set_ylabel(r'$f_p(t)$')

        if normalize:
            ax0_title = r'Interpolation of $\tau_p(t)$, $p = %g$' % self.p
        else:
            ax0_title = r'Interpolation of $f_p(t)$, $p = %g$' % self.p
        if compare:
            ax0_title = r'(a) ' + ax0_title

        ax[0].set_title(ax0_title)
        ax[0].legend(fontsize='small')

        # Plot relative error in percent
        if compare:
            if self.interpolator.q > 0:
                ax[1].semilogx(self.interpolator.t_i,
                               numpy.zeros(self.interpolator.q), 'o',
                               color=exact_color, markersize=markersize,
                               label='Interpolant points', zorder=20)
            ax[1].semilogx(inquiry_points, 100.0*tau_relative_error,
                           color=interp_color, label='Interpolated')
            ax[1].grid(axis='x')
            ax[1].semilogx(ax[1].get_xlim(), [0, 0], color='#CCCCCC',
                           linewidth=0.75)
            ax[1].set_xlim([inquiry_points[0], inquiry_points[-1]])
            ax[1].set_xlabel('$t$')
            if normalize:
                ax[1].set_ylabel(r'$1-\tau_{\mathrm{approx}}(t) / ' +
                                 r'\tau_{\mathrm{exact}}(t)$')
            else:
                ax[1].set_ylabel(r'$1-f_{\mathrm{approx}}(t) / ' +
                                 r'f_{\mathrm{exact}}(t)$')
            ax1_title = r'(b) Relative error of interpolation, $p=%g$' % self.p
            ax[1].set_title(ax1_title)
            tau_range = numpy.max(numpy.abs(100.0 * tau_relative_error))
            if tau_range != 0.0:
                decimals = int(numpy.ceil(-numpy.log10(tau_range))) + 1
            else:
                decimals = 2
            ax[1].yaxis.set_major_formatter(
                    matplotlib.ticker.PercentFormatter(decimals=decimals))
            ax[1].legend(fontsize='small')

        plt.tight_layout()

        # Check if the graphical backend exists
        if matplotlib.get_backend() != 'agg':
            plt.show()
        else:
            # Save the plot as SVG file in the current directory
            show_or_save_plot(plt, 'interpolation',
                              transparent_background=True)

    # ===========
    # plot finite
    # ===========

    def _plot_finite(
            self,
            inquiry_points,
            normalize=True,
            compare=False):
        """
        Plots the interpolation results, together with the comparison with the
        exact solution and the relative error of the interpolation.

        To plot, set ``Plot=True`` argument in
        :mod:`imate.InterpolateSchatten`.

        Parameters
        ----------
        inquiry_points: numpy.array
            Inquiry points to be interpolated

        schatten_interpolated : numpy.array
            Interpolation of the trace at inquiry

        schatten_exact : numpy.array, default=None
            Exact solutions of the trace at inquiry points. If this variable is
            not None, it will be plotted together with the interpolated
            results.

        schatten_relative_error : numpy.array, default=None
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
        schatten_interpolated = self.interpolate(inquiry_points)

        if compare:
            schatten_exact = self.eval(inquiry_points)

        # Normalize schatten to tau
        if normalize:
            schatten_B = self.interpolator.schatten_B

            # EXT and EIG methods do not compute schatten_B by default.
            if schatten_B is None:
                schatten_B = self.interpolator._compute_schatten(
                        self.interpolator.B, self.interpolator.p)

            normal_factor = schatten_B
        else:
            normal_factor = 1.0

        if self.interpolator.schatten_i is not None:
            tau_i = self.interpolator.schatten_i / normal_factor
        tau_interpolated = schatten_interpolated / normal_factor

        if compare:
            tau_exact = schatten_exact / normal_factor
            tau_relative_error = 1.0 - (tau_interpolated / tau_exact)

        tau_0 = self.interpolator.tau0
        t_i = self.interpolator.t_i
        t = inquiry_points

        if self.kind == 'CRF':
            scale = self.interpolator.scale
        else:
            scale = 1.0

        t_ = t / scale
        t_i_ = t_i / scale
        x = (t_-1.0) / (t_+1.0)
        x_i = (t_i_-1.0) / (t_i_+1.0)

        if self.interpolator.func_type == 1:
            if self.interpolator.schatten_i is not None:
                y_i = tau_i / (tau_0 + t_i) - 1.0
            y = tau_interpolated / (tau_0 + t) - 1.0
            if compare:
                y_ex = tau_exact / (tau_0 + t) - 1.0
        elif self.interpolator.func_type == 2:
            if self.interpolator.schatten_i is not None:
                y_i = (tau_i - tau_0) / t_i - 1.0
            y = (tau_interpolated - tau_0) / t - 1.0
            if compare:
                y_ex = (tau_exact - tau_0) / t - 1.0
        else:
            raise ValueError('"func_type" should be either 1 or 2.')

        # Plot results
        if compare:
            # Two subplots
            fig, ax = plt.subplots(ncols=2, figsize=(9, 4))
        else:
            # One subplot
            fig, ax = plt.subplots(figsize=(5, 4))
            ax = [ax]

        # Plot settings
        markersize = 4
        exact_color = 'firebrick'
        interp_color = 'black'

        # Plot interpolant points with their exact values
        if self.interpolator.q > 0:
            if self.interpolator.schatten_i is not None:
                ax[0].plot(x_i, y_i, 'o', color=exact_color,
                           markersize=markersize, label='Interpolant points',
                           zorder=20)

        # Plot exact values
        if compare:
            ax[0].plot(x, y_ex, color=exact_color,
                       label='Exact')

        # Plot interpolated results
        ax[0].plot(x, y, color=interp_color,
                   label='Interpolated')

        ax[0].grid(axis='x')
        ax[0].set_xlim([-1, 1])
        ax[0].set_ylim(bottom=0, top=None)
        if self.kind == 'SPL':
            ax[0].set_xlabel(r'$(t-1) / (t+1)$')
        elif self.kind == 'CRF':
            ax[0].set_xlabel(r'$(t-\alpha) / (t+\alpha)$')
        else:
            raise ValueError('"method" should be "SPL" or "CRF".')

        if normalize:
            if self.interpolator.func_type == 1:
                ax[0].set_ylabel(r'$\tau_p(t) / (\tau_{p, 0} + t) - 1$')
            elif self.interpolator.func_type == 2:
                ax[0].set_ylabel(r'$(\tau_p(t)-\tau_{p, 0}) / t - 1$')
        else:
            if self.interpolator.func_type == 1:
                ax[0].set_ylabel(r'$f_p(t) / (f_{p, 0} + t) - 1$')
            elif self.interpolator.func_type == 2:
                ax[0].set_ylabel(r'$(f_p(t)-f_{p, 0}) / t - 1$')

        if normalize:
            ax0_title = r'Interpolation of $\tau_p(t)$, $p = %g$' % self.p
        else:
            ax0_title = r'Interpolation of $f_p(t)$, $p = %g$' % self.p
        ax0_title = r'(a) ' + ax0_title

        ax[0].set_title(ax0_title)
        ax[0].legend(fontsize='small')

        if compare:
            # Plot relative error in percent
            if self.interpolator.q > 0:
                ax[1].semilogx(self.interpolator.t_i,
                               numpy.zeros(self.interpolator.q), 'o',
                               color=exact_color, markersize=markersize,
                               label='Interpolant points', zorder=20)
            ax[1].semilogx(inquiry_points, 100.0*tau_relative_error,
                           color=interp_color, label='Interpolated')
            ax[1].grid(axis='x')
            ax[1].semilogx(ax[1].get_xlim(), [0, 0], color='#CCCCCC',
                           linewidth=0.75)
            ax[1].set_xlim([inquiry_points[0], inquiry_points[-1]])
            ax[1].set_xlabel('$t$')
            if normalize:
                ax[1].set_ylabel(r'$1-\tau_{\mathrm{approx}}(t) / ' +
                                 r'\tau_{\mathrm{exact}}(t)$')
            else:
                ax[1].set_ylabel(r'$1-f_{\mathrm{approx}}(t) / ' +
                                 r'f_{\mathrm{exact}}(t)$')
            ax1_title = r'(b) Relative error of interpolation, $p=%g$' % self.p
            ax[1].set_title(ax1_title)
            tau_range = numpy.max(numpy.abs(100.0 * tau_relative_error))
            if tau_range != 0.0:
                decimals = int(numpy.ceil(-numpy.log10(tau_range))) + 1
            else:
                decimals = 2
            ax[1].yaxis.set_major_formatter(
                    matplotlib.ticker.PercentFormatter(decimals=decimals))
            ax[1].legend(fontsize='small')

        plt.tight_layout()

        # Check if the graphical backend exists
        if matplotlib.get_backend() != 'agg':
            plt.show()
        else:
            # Save the plot as SVG file in the current directory
            show_or_save_plot(plt, 'interpolation',
                              transparent_background=True)
