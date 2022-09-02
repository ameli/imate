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

from .interpolate_schatten import InterpolateSchatten
import numpy

try:
    from .._utilities.plot_utilities import *                # noqa: F401, F403
    from .._utilities.plot_utilities import load_plot_settings, matplotlib, \
        show_or_save_plot, plt
    plot_modules_exist = True
except ImportError:
    plot_modules_exist = False


# ==================
# Interpolate Logdet
# ==================

class InterpolateLogdet(InterpolateSchatten):
    """
    """

    # ====
    # init
    # ====

    def __init__(self, A, B=None, options={'method': 'cholesky'},
                 verbose=False, kind='imbf', ti=[], **kwargs):
        """
        Initializes the object depending on the method.
        """

        # In Schatten operator, p=0 corresponds to determinant**(1/n)
        super(InterpolateLogdet, self).__init__(
                A, B=B, p=0, ti=ti, kind=kind, verbose=verbose,
                options=options, **kwargs)

    # ==================
    # schatten to logdet
    # ==================

    def _schatten_to_logdet(self, schatten):
        """
        Converts Schatten anti-norm to logdet.
        """

        logdet_ = self.n * numpy.log(schatten)
        return logdet_

    # ==================
    # logdet to schatten
    # ==================

    def _logdet_to_schatten(self, logdet_):
        """
        Converts logdet to Schatten anti-norm.
        """

        schatten = numpy.exp(logdet_ / self.n)
        return schatten

    # ========
    # __call__
    # ========

    def __call__(self, t):
        """
        Same as :func:`InterpolateSchatten.interpolate` method.
        """

        schatten = super(InterpolateLogdet, self).__call__(t)
        return self._schatten_to_logdet(schatten)

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

        schatten = super(InterpolateLogdet, self).eval(t)

        return self._schatten_to_logdet(schatten)

    # ===========================
    # compare with exact solution
    # ===========================

    def _compare_with_exact_solution(self, t, logdet_):
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
            logdet_exact = logdet_
            logdet_relative_error = numpy.zeros(t.shape)

        else:

            # Compute exact solution
            logdet_exact = self.eval(t)
            logdet_relative_error = (logdet_ - logdet_exact) / (logdet_exact)

        return logdet_exact, logdet_relative_error

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

        schatten = super(InterpolateLogdet, self).interpolate(t)
        return self._schatten_to_logdet(schatten)

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

        schatten_lb = super(InterpolateLogdet, self).bound(t)

        logdet_lb = self._schatten_to_logdet(schatten_lb)
        return logdet_lb

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

        schatten_ub = super(InterpolateLogdet, self).upper_bound(t)

        logdet_ub = self._schatten_to_logdet(schatten_ub)
        return logdet_ub

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
        logdet_interpolated = self.interpolate(inquiry_points)

        if compare:
            logdet_exact = self.eval(inquiry_points)

        # Normalize logdet to tau
        if normalize:
            schatten_B = self.interpolator.schatten_B

            # EXT and EIG methods do not compute schatten_B by default.
            if schatten_B is None:
                schatten_B = self.interpolator._compute_schatten(
                        self.interpolator.B, self.interpolator.p)

            normal_factor = self._schatten_to_logdet(schatten_B)
        else:
            normal_factor = 0.0

        if self.interpolator.schatten_i is not None:
            logdet_i = self._schatten_to_logdet(self.interpolator.schatten_i)
            tau_i = logdet_i - normal_factor
        tau_interpolated = logdet_interpolated - normal_factor
        if compare:
            tau_exact = logdet_exact - normal_factor
            tau_absolute_error = tau_interpolated - tau_exact

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
                ax[0].semilogx(self.interpolator.t_i, tau_i, 'o',
                               color=exact_color, markersize=markersize,
                               label='Interpolant points', zorder=20)

        # Plot exact values
        if compare:
            ax[0].semilogx(inquiry_points, tau_exact, color=exact_color,
                           label='Exact')

        # Plot interpolated results
        ax[0].semilogx(inquiry_points, tau_interpolated, color=interp_color,
                       label='Interpolated')

        ax[0].grid(axis='x')
        ax[0].set_xlim([inquiry_points[0], inquiry_points[-1]])
        ax[0].set_xlabel(r'$t$')

        if normalize:
            ax[0].set_ylabel(r'$g_0(t)$')
        else:
            ax[0].set_ylabel(r'$f_0(t)$')

        if normalize:
            ax0_title = r'Interpolation of $g_0(t)$'
        else:
            ax0_title = r'Interpolation of $f_0(t)$'
        if compare:
            ax0_title = r'(a) ' + ax0_title

        ax[0].set_title(ax0_title)
        ax[0].legend(fontsize='small')

        # Plot absolute error in percent
        if compare:
            if self.interpolator.q > 0:
                ax[1].semilogx(self.interpolator.t_i,
                               numpy.zeros(self.interpolator.q), 'o',
                               color=exact_color, markersize=markersize,
                               label='Interpolant points', zorder=20)
            ax[1].semilogx(inquiry_points, tau_absolute_error,
                           color=interp_color, label='Interpolated')
            ax[1].grid(axis='x')
            ax[1].semilogx(ax[1].get_xlim(), [0, 0], color='#CCCCCC',
                           linewidth=0.75)
            ax[1].set_xlim([inquiry_points[0], inquiry_points[-1]])
            ax[1].set_xlabel('$t$')
            if normalize:
                ax[1].set_ylabel(r'$g_{\mathrm{approx}}(t) - ' +
                                 r'g_{\mathrm{exact}}(t)$')
            else:
                ax[1].set_ylabel(r'$f_{\mathrm{approx}}(t) - ' +
                                 r'f_{\mathrm{exact}}(t)$')
            ax1_title = r'(b) Absolute error of interpolation'
            ax[1].set_title(ax1_title)
            ax[1].legend(fontsize='small')

        plt.tight_layout()

        # Check if the graphical backend exists
        if matplotlib.get_backend() != 'agg':
            plt.show()
        else:
            # Save the plot as SVG file in the current directory
            show_or_save_plot(plt, 'interpolation',
                              transparent_background=True)
