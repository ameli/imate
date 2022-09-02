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

    Parameters
    ----------

    A : numpy.ndarray, scipy.sparse matrix
        Symmetric positive-definite matrix. Matrix can be dense or sparse.

    B : numpy.ndarray, scipy.sparse matrix, default=None
        Symmetric positive-definite matrix. Matrix can be dense or sparse.
        The matrix `B` should have the same size and type of the matrix `A`. If
        `B` is `None` (default value), it is assumed that `B` is the identity
        matrix.

    p : float, default=2
        The order :math:`p` in the Schatten :math:`p`-norm, which can be real
        positive, negative or zero.

    options : dict, default={``'method'``: ``'cholesky'``}
        Options to pass to :func:`imate.schatten` function.

    verbose : bool, default=False
        If `True`, it prints some information on the computation process.

    kind : {`'ext'`, `'eig'`, `'mbf'`, `'imbf'`, `'rbf'`, `'crf'`, `'spl'`, \
            `'rpf'`}, default: `'imbf'`
        The algorithm of interpolation. See documentation for each specific
        algorithm below:

        * :ref:`imate.InterpolateSchatten.ext`
        * :ref:`imate.InterpolateSchatten.eig`
        * :ref:`imate.InterpolateSchatten.mbf`
        * :ref:`imate.InterpolateSchatten.imbf`
        * :ref:`imate.InterpolateSchatten.rbf`
        * :ref:`imate.InterpolateSchatten.crf`
        * :ref:`imate.InterpolateSchatten.spl`
        * :ref:`imate.InterpolateSchatten.rpf`

    ti : array_like(float), default=None
        A list or an array of interpolant points that the interpolator use to
        interpolate. If an empty list is give, ``[]``, a default list of
        interpolant points for specific ``kind`` is used.

    kwargs : \\*\\*kwargs
        Additional arguments to pass to each specific method. See documentation
        for each ``kind`` in the above.

    See Also
    --------

    imate.InterpolateTrace
    imate.InterpolateLogdet
    imate.schatten

    Attributes
    ----------
    kind : str
        Method of interpolation

    verbose : bool
        Verbosity of the computation process

    n : int
        Since of the matrix

    q : int
        Number of interpolant points

    p : float
        Order of Schatten :math:`p`-norm

    Methods
    -------

    __call__
    eval
    interpolate
    bound
    upper_bound
    plot

    Notes
    -----

    **Schatten Norm:**

    In this class, the Schatten :math:`p`-norm of the matrix
    :math:`\\mathbf{A}` is defined by

    .. math::
        :label: schatten-eq-3

        \\Vert \\mathbf{A} \\Vert_p =
        \\begin{cases}
            \\left| \\mathrm{det}(\\mathbf{A})
            \\right|^{\\frac{1}{n}}, & p=0, \\\\
            \\left| \\frac{1}{n}
            \\mathrm{trace}(\\mathbf{A}^{\\frac{1}{p}})
            \\right|^{\\frac{1}{p}}, & p \\neq 0,
        \\end{cases}

    where :math:`n` is the size of the matrix. When :math:`p \\geq 0`, the
    above definition is the Schatten **norm**, and when :math:`p < 0`, the
    above is the Schatten **anti-norm**.

    .. note::

        Conventionally, the Schatten norm is defined without the normalizing
        factor :math:`\\frac{1}{n}` in :math:numref:`schatten-eq-3`. However,
        this factor is justified by the continuity granted by

        .. math::
            :label: schatten-continuous-3

            \\lim_{p \\to 0} \\Vert \\mathbf{A} \\Vert_p =
            \\Vert \\mathbf{A} \\Vert_0.

        See [1]_ (Section 2) and the example below for details.

    **Interpolation of Affine Matrix Function:**

    This class interpolates the one-parameter matrix function:

    .. math::

        t \\mapsto \\| \\mathbf{A} + t \\mathbf{B} \\|_p,

    where the matrices :math:`\\mathbf{A}` and :math:`\\mathbf{B}` are
    symmetric and positive semi-definite (positive-definite if :math:`p < 0`)
    and :math:`t \\in [t_{\\inf}, \\infty)` is a real parameter where
    :math:`t_{\\inf}` is the minimum :math:`t` such that
    :math:`\\mathbf{A} + t_{\\inf} \\mathbf{B}` remains positive-definite.

    The interpolator is initialized by providing :math:`q` interpolant points
    :math:`t_i`, :math:`i = 1, \\dots, q`, which are often logarithmically
    spaced in some interval :math:`t_i \\in [t_1, t_p]`. The interpolator can
    interpolate the above function at inquiry points :math:`t \\in [t_1, t_p]`
    using various methods.

    References
    ----------

    .. [1] Ameli, S., and Shadden. S. C. (2022). Interpolating Log-Determinant
           and Trace of the Powers of Matrix :math:`\mathbf{A} + t \mathbf{B}`.
           `arXiv: 2009.07385 <https://arxiv.org/abs/2207.08038>`_ [math.NA].

    Examples
    --------

    **Basic Usage:**

    Interpolate the Schatten `2`-norm of the affine matrix function
    :math:`\\mathbf{A} + t \\mathbf{B}` using ``imbf`` algorithm and the
    interpolating points :math:`t_i = [10^{-2}, 10^{-1}, 1, 10]`.

    .. code-block:: python
        :emphasize-lines: 9, 13

        >>> # Generate two sample matrices (symmetric and positive-definite)
        >>> from imate import generate_matrix
        >>> A = generate_matrix(size=20, correlation_scale=1e-1)
        >>> B = generate_matrix(size=20, correlation_scale=2e-2)

        >>> # Initialize interpolator object
        >>> from imate import InterpolateSchatten
        >>> ti = [1e-2, 1e-1, 1, 1e1]
        >>> isch = InterpolateSchatten(A, B, p=2, kind='imbf', ti=ti)

        >>> # Interpolate at an inquiry point t = 0.4
        >>> t = 4e-1
        >>> isch(t)

    Alternatively, call :meth:`imate.nterpolateSchatten.interpolate` to
    interpolate at points `t`:

    .. code-block:: python

        >>> # This is the same as isch(t)
        >>> isch.interpolate(t)

    To evaluate the exact value of the Schatten norm at point `t` without
    interpolation, call :meth:`imate.InterpolateSchatten.eval` function:

    .. code-block:: python

        >>> # This evaluates the function value at t exactly (no interpolation)
        >>> isch.interpolate(t)

    .. warning::

        Calling :meth:`imate.InterpolateSchatten.eval` may take a longer time
        to compute as it computes the function exactly. Particularly, if `t` is
        a large array, it may take a very long time to return the exact values.

    **Arguments Specific to Algorithms:**

    In the above example, the ``imbf`` algorithm is used. See more arguments
    of this algorithm at :ref:`imate.InterpolateSchatten.imbf`. In the next
    example, we pass ``basis_functions_type`` specific to this algorithm:

    .. code-block:: python
        :emphasize-lines: 3

        >>> # Passing kwatgs specific to imbf algorithm
        >>> isch = InterpolateSchatten(A, B, p=2, kind='imbf', ti=ti,
        ...                            basis_functions_type='Orthogonal2')
        >>> isch(t)

    **Passing Options:**

    At each interpolation point `ti`, the value of the Schatten norm is
    computed using :func:`imate.schatten` function. You can pass arguments to
    this function using ``options``. To do so, create a dictionary with the
    keys with the name of the argument. For instance, to use
    :ref:`imate.schatten.slq` method with ``min_num_samples=20`` and
    ``max_num_samples=100``, create the following dictionary

    .. code-block:: python

        >>> # Specify arguments as a dictionary
        >>> options = \
        ... {
        ...     'method': 'slq',
        ...     'min_num_samples': 20,
        ...     'max_num_samples': 100
        ... }

        >>> # Pass the options to the interpolator
        >>> isch = InterpolateSchatten(A, B, p=2, options=options, kind='imbf',
        ...                            ti=ti)
        >>> isch(t)

    **Interpolate on Range of Points:**

    Once the interpolation object ``isch`` in the above example is
    instantiated, calling :meth:`imate.InterpolateSchatten.interpolate` on
    a list of inquiry points `t` has almost no computational cost. The next
    example inquires interpolation on `1000` points `t`:

    Interpolate an array of inquiry points

    .. code-block:: python

        >>> # Interpolate at an array of points
        >>> import numpy
        >>> t_array = numoy.logspace(-2, 1, 1000)
        >>> norm_array = isch.interpolator(t_array)

    One may plot the above interpolated results as follows:

    .. code-block:: python

        >>> import matplotlib.pyplot as plt
        >>> plt.loglog(t_array, norm_array, color='black')
        >>> plt.xlim([t_array[0], t_array[-1]])
        >>> plt.xlabel('$t$')
        >>> plt.ylabel('$\\Vert \\mathbf{A} + t \\mathbf{B} \\Vert_{2}$')
        >>> plt.title('Interpolation of Schatten 2-Norm')
        >>> plt.show()

    **Plotting Interpolation and Compare with Exact Solution:**

    A more convenient way to plot the interpolation result is to call
    :meth:`imate.InterpolateSchatten.plot` function.

    .. code-block:: python

        >>> isch.plot(t)

    To compare with the true values (without interpolation), pass
    ``compare=True`` to the above function:

    .. warning::

        By setting ``compare`` to `True`, every point in the array `t` is
        evaluated both using interpolation and with the exact method (no
        interpolation). If the size of `t` is large, this may take a very
        long run time.

    .. code-block:: python

        >>> isch.plot(t, compare=True)

    Alternatively, you may set ``normalize`` to `True` to plot the normalized
    function

    .. math::

        \\frac{\\Vert \\mathbf{A} + t \\mathbf{B} \\Vert_{p}}{
        \\Vert \\mathbf{B} \\Vert_p}.

    .. code-block:: python

        >>> isch.plot(t, normalize=True, compare=True)


    **Using Other Algorithms:**

    All the above examples uses ``imbf`` algorithms. You may choose other
    algorithms using ``kind`` argument. For instance, the next example uses
    the Chebyshev rational functions:

    .. code-block:: python

        >>> # Generate two sample matrices (symmetric and positive-definite)
        >>> isch = InterpolateSchatten(A, B, p=2, kind='crf', ti=ti)

        >>> # Interpolate at an inquiry point t = 0.4
        >>> t = 4e-1
        >>> isch(t)

    **Example of Large Matrices:**

    The following is a practical example on a large matrix. 

    """

    # ====
    # init
    # ====

    def __init__(self, A, B=None, p=2, options={'method': 'cholesky'},
                 verbose=False, kind='imbf', ti=[], **kwargs):
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
        if kind.lower() == 'ext':
            # Exact computation, not interpolation
            self.interpolator = ExactMethod(
                    A, B, p=self.p, options=options, verbose=verbose, **kwargs)

        elif kind.lower() == 'eig':
            # Eigenvalues kind
            self.interpolator = EigenvaluesMethod(
                    A, B, p=self.p, options=options, verbose=verbose, **kwargs)

        elif kind.lower() == 'mbf':
            # Monomial Basis Functions kind
            self.interpolator = MonomialBasisFunctionsMethod(
                    A, B, p=self.p, ti=ti, options=options, verbose=verbose,
                    **kwargs)

        elif kind.lower() == 'imbf':
            # Inverse Monomial Basis Functions kind
            self.interpolator = InverseMonomialBasisFunctionsMethod(
                    A, B, p=self.p, ti=ti, options=options, verbose=verbose,
                    **kwargs)

        elif kind.lower() == 'rbf':
            # Radial Basis Functions kind
            self.interpolator = RadialBasisFunctionsMethod(
                    A, B, p=self.p, ti=ti, options=options, verbose=verbose,
                    **kwargs)

        elif kind.lower() == 'crf':
            # Chebushev Rational Functions
            self.interpolator = ChebyshevRationalFunctionsMethod(
                    A, B, p=self.p, ti=ti, options=options, verbose=verbose,
                    **kwargs)

        elif kind.lower() == 'spl':
            # Spline
            self.interpolator = SplineMethod(
                    A, B, p=self.p, ti=ti, options=options, verbose=verbose,
                    **kwargs)

        elif kind.lower() == 'rpf':
            # Rational Polynomial Functions kind
            self.interpolator = RationalPolynomialFunctionsMethod(
                    A, B, p=self.p, ti=ti, options=options, verbose=verbose,
                    **kwargs)

        else:
            raise ValueError("'kind' is invalid. Select one of 'ext', " +
                             "'eig', 'mbf', 'imbf', 'rbf', 'crf', 'spl', or " +
                             "'rpf'.")

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

        if self.kind.lower() == 'ext':

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
        Interpolates at point(s) `t`.

        Parameters
        ----------

        t : float, list, or numpy.array
            The inquiry point(s) to be interpolated.

        Returns
        -------

        norm : float or numpy.array
            The interpolated value of the Schatten norm.
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

        if self.kind.lower() in ['crf', 'spl']:
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

        if self.kind.lower() == 'crf':
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
        if self.kind.lower() == 'spl':
            ax[0].set_xlabel(r'$(t-1) / (t+1)$')
        elif self.kind.lower() == 'crf':
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
