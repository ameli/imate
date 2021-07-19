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
    from .._utilities.plot_utilities import load_plot_settings, save_plot, \
        matplotlib, plt
    plot_modules_exist = True
except ImportError:
    plot_modules_exist = False


# ====================
# Interpolate Traceinv
# ====================

class InterpolateTraceinv(object):
    """
    Interpolates the trace of inverse of affine matrix functions.

    :param A: A positive-definite matrix. Matrix can be dense or sparse.
    :type A: numpy.ndarray or scipy.sparse.csc_matrix

    :param B: A positive-definite matrix. Matrix can be dense or sparse.
        If ``None`` or not provided, it is assumed that ``B`` is an identity
        matrix of the shape of ``A``.
    :type B: numpy.ndarray or scipy.sparse.csc_matrix

    :param interpolant_points: A list or an array of points that the
        interpolator use to interpolate. The trace of inverse is computed for
        the interpolant points with exact method.
    :type interpolant_points: list(float) or numpy.array(float)

    :param ethod: One of the methods ``'EXT'``, ``'EIG'``, ``'MBF'``,
        ``'RMBF'``, ``'RBF'``, and ``'RPF'``. Default is ``'RMBF'``.
    :type method: string

    :param traceinv_options: A dictionary of arguments to pass to
        :mod:`imate.traceinv` module.
    :type traceinv_options: dict

    :param verbose: If ``True``, prints some information on the computation
        process. Default is ``False``.
    :type verbose: bool

    :param interpolation_options: Additional options to pass to each specific
        interpolator class. See the sub-classes of
        :mod:`imate.InterpolantBase`.
    :type interpolation_options: \\*\\*kwargs

    **Methods:**

    ==========  ==============================  ============
    ``method``  Description                     Results
    ==========  ==============================  ============
    ``'EXT'``   Computes without interpolation  exact
    ``'EIG'``   Uses Eigenvalues of matrix      exact
    ``'MBF'``   Monomial Basis Functions        interpolated
    ``'RMBF'``  Root monomial basis functions   interpolated
    ``'RBF'``   Radial basis functions          interpolated
    ``'RPF'``   Rational polynomial functions   interpolated
    ==========  ==============================  ============

    **Details:**

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
    :math:`t \\in [t_1, t_p]`
    using various methods.

    **Examples:**

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

    .. seealso::
        This module calls a derived class of the base class
        :mod:`imate.InterpolateTraceinv.InterpolantBase`.

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

    # =======
    # compute
    # =======

    def compute(self, t):
        """
        Computes the function :math:`\\mathrm{trace}\\left( (\\mathbf{A} +
        t \\mathbf{B})^{-1} \\right)` at the input point :math:`t` using exact
        method.

        The computation method used in this function is exact (no
        interpolation). This function  is primarily used to compute trace on
        the *interpolant points*.

        :param: t: An inquiry point, which can be a single number, or an array
            of numbers.
        :type t: float or numpy.array

        :return: The exact value of the trace
        :rtype: float or numpy.array
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

        :param t: Inquiry points
        :type t: numpy.ndarray

        :param Trace: The interpolated computation of trace.
        :type Trace: float or numpy.ndarray

        :returns:
            - Exact solution of trace.
            - Relative error of interpolated solution compared to the exact
              solution.
        :rtype:
            - float or numpy.ndarray
            - float or numpy.ndarray
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

        This is the main interface function of this module and it is used after
        the interpolation
        object is initialized.

        :param t: The inquiry point(s).
        :type t: float, list, or numpy.array

        :param compare_with_exact: If ``True``, it computes the trace with
            exact solution, then compares it with the interpolated solution.
            The return values of the ``Interpolate()`` functions become
            interpolated trace, exact solution, and relative error. **Note:**
            When this option is enabled, the exact solution will be computed
            for all inquiry points, which can take a very long time. Default is
            ``False``.
        :type compare_with_exact: bool

        :param plot: If ``True``, it plots the interpolated trace versus the
            inquiry points. In addition, if the option ``compare_with_exact``
            is also set to ``True``, the plotted diagram contains both
            interpolated and exact solutions and the relative error of
            interpolated solution with respect to the exact solution.
        :type plot: bool

        :return: The interpolated value of the trace.
        :rtype: float or numpy.array

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

       .. note::

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

        :param t: An inquiry point or an array of inquiry points.
        :type t: float or numpy.array

        :return: Lower bound of the affine matrix function.
        :rtype: float or numpy.array

        .. seealso::

            This function is implemented in
            :meth:`imate.InterpolateTraceinv.InterpolantBase.g`.
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

        :param t: An inquiry point or an array of inquiry points.
        :type t: float or numpy.array

        :return: Lower bound of the affine matrix function.
        :rtype: float or numpy.array

        .. seealso::

            This function is implemented in
            :meth:`imate.InterpolateTraceinv.InterpolantBase.upper_bound`.
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

    # ==================
    # plot interpolation
    # ==================

    def plot_interpolation(self, inquiry_points, trace_interpolated,
                           trace_exact=None, trace_relative_error=None):
        """
        Plots the interpolation results, together with the comparison with the
        exact solution and the relative error of the interpolation.

        To plot, set ``Plot=True`` argument in
        :mod:`imate.InterpolateTraceinv`.

        :param inquiry_points: Inquiry points
        :type inquiry_points: numpy.ndarray

        :param trace_interpolated: Interpolation of the trace at inquiry
            points.
        :type trace_interpolated: numpy.ndarray

        :param trace_exact: Exact solutions of the trace at inquiry points.
            If this variable is not None, it will be plotted together with the
            interpolated results.
        :type trace_exact: numpy.ndarray

        :param trace_relative_error: Relative errors of the interpolation with
            respect to the exact solution. If not None, the relative errors
            will be plotted on a second axis.
        :type trace_relative_error: numpy.ndarray
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

        # Plot results
        if trace_relative_error is None:
            # One subplot
            fig, ax = plt.subplots(figsize=(5, 4))
            ax = [ax]
        else:
            # Two subplots
            fig, ax = plt.subplots(ncols=2, figsize=(9, 4))

        # Plot interpolant points with their exact values
        if self.interpolator.p > 0:
            ax[0].semilogx(self.interpolator.t_i, self.interpolator.trace_i,
                           'o', color='red', label='Interpolant points',
                           zorder=20)

        # Plot exact values
        if trace_exact is not None:
            ax[0].semilogx(inquiry_points, trace_exact, color='red',
                           label='Exact')

        # Plot interpolated results
        ax[0].semilogx(inquiry_points, trace_interpolated, color='black',
                       label='Interpolated')

        ax[0].grid(True)
        ax[0].set_xlim([inquiry_points[0], inquiry_points[-1]])
        ax[0].set_xlabel(r'$t$')
        if self.interpolator.B_is_identity:
            if matplotlib.rcParams['text.usetex'] is True:
                ax[0].set_ylabel(r'trace$(\\mathbf{A} + t \\mathbf{I})^{-1}$')
            else:
                ax[0].set_ylabel(r'trace$(\mathbf{A} + t \mathbf{I})^{-1}$')
        else:
            if matplotlib.rcParams['text.usetex'] is True:
                ax[0].set_ylabel(r'trace$(\\mathbf{A} + t \\mathbf{B})^{-1}$')
            else:
                ax[0].set_ylabel(r'trace$(\mathbf{A} + t \mathbf{B})^{-1}$')
        ax[0].set_title('Trace of Inverse')
        ax[0].legend(fontsize='small')

        # Plot relative error in percent
        if trace_relative_error is not None:
            if self.interpolator.p > 0:
                ax[1].semilogx(self.interpolator.t_i,
                               numpy.zeros(self.interpolator.p), 'o',
                               color='red', label='Interpolant points',
                               zorder=20)
            ax[1].semilogx(inquiry_points, 100.0*trace_relative_error,
                           color='black', label='Interpolated')
            ax[1].grid(True)
            ax[1].set_xlim([inquiry_points[0], inquiry_points[-1]])
            ax[1].set_xlabel('$t$')
            ax[1].set_ylabel('Relative Error (in Percent)')
            ax[1].set_title('Relative Error')
            ax[1].yaxis.set_major_formatter(
                    matplotlib.ticker.PercentFormatter())
            ax[1].legend(fontsize='small')

        plt.tight_layout()

        # Check if the graphical backend exists
        if matplotlib.get_backend() != 'agg':
            plt.show()
        else:
            # Save the plot as SVG file in the current directory
            save_plot(plt, 'interpolation_results',
                      transparent_background=True)
