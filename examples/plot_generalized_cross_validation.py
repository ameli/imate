#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# imports
# =======

import sys
import numpy
import scipy.optimize
from functools import partial

# Package Modules
from _utilities.plot_utilities import *                      # noqa: F401, F403
from _utilities.plot_utilities import load_plot_settings, save_plot, plt, \
        matplotlib, FormatStrFormatter
from _utilities.processing_time_utilities import TimeCounter, process_time, \
        restrict_computation_to_single_processor
from _utilities.data_utilities import generate_matrix, generate_noisy_data, \
        generate_basis_functions
from imate import InterpolateSchatten
from imate import traceinv


# ============================
# generalized cross validation
# ============================

def generalized_cross_validation(X, K, z, TI, shift, time_counter,
                                 use_log_theta, theta_):
    """
    Computes the CGV function :math:`V(\\theta)`.

    * ``X`` shape is ``(n, m)``.
    * :math:`K = X^{\\intercal} X`, which is ``(m, m)`` shape.

    **Reference:**

        Golub, G., & Von Matt, U. (1997).
        Generalized Cross-Validation for Large-Scale Problems.
        Journal of Computational and Graphical Statistics, 6(1), 1-34.
        doi: `10.2307-1390722 <https://www.jstor.org/stable/pdf/1390722.pdf?ref
        reqid=excelsior%3Adf48321fdd477aab0ea5dbf2542df01d>`_

        :param X: Matrix if basis functions of the shape ``(n, m)`` with ``m``
            basis functions over ``n`` spatial points.
        :type X: numpy.ndarray

        :param K: Correlation matrix of the shape ``(n, n)``.
        :type: numpy.ndarray or scipy.sparse.csc_matrix

        :param z: Column vector of data at ``n`` points.
        :type: numpy.array

        :param TI: imate interpolating object
        :type TI: imate.interpolateTraceInv

        :param shift: shift for the signular matrix ``K0`` to
            ``K = K0 + shift * I``.
        :type shift: float

        :param time_counter: A counter object to store the elapsed time and to
            be read outside of this function.
        :type TimeCounteR: examples._utilities.TimeCounter

        :param UseLog: A flag, if ``True``, it assumes ``theta_`` is in
            logarithmic scale. If ``False``, then ``theta_`` is not assumed to
            be in logarithmic scale.
        :type UseLog: bool

        :param theta_: Parameter of generalized cross validation.
        :type: float

        The generalized cross-validation (GCV) function is:

        .. math::

            V(\\theta) = \\frac{\\frac{1}{n} \\| \\mathbf{I} -
            \\mathbf{X} (\\mathbf{X}^{\\intercal} \\mathbf{X} +
            n \\theta \\mathbf{I})^{-1} \\mathbf{X}^{\\intercal}
            \\boldsymbol{z} \\|_2^2}{\\left( \\frac{1}{n} \\mathrm{trace}
            \\left( (\\mathbf{I} - \\mathbf{X}(\\mathbf{X}^{\\intercal}
            \\mathbf{X} + n \\theta \\mathbf{I})^{-1})\\mathbf{X}^{\\intercal}
            \\right) \\right)^2}

        In the above equation, the term involving trace is implemented
        differently depending on wtether :math:`n > m` or :math:`n < m` (see
        details in [GOLUB-1979]_).

        .. note::

            In this function, we use the variable ``theta_`` for
            :math:`\\theta`.


        **References:**

        .. [GOLUB-1979] Golub, G., Heath, M., & Wahba, G. (1979). Generalized
        Cross-Validation as a Method for Choosing a Good Ridge Parameter.
        Technometrics, 21(2), 215-223. doi: `10.2307/1268518
        <https://www.jstor.org/stable/1268518?seq=1>`_

    """

    # If theta is in the logarithm scale, convert it to normal scale
    if use_log_theta is True:
        theta_ = 10**theta_

    n, m = X.shape
    mu = n*theta_

    y1 = X.T.dot(z)
    Im = numpy.eye(m)
    A = X.T.dot(X) + mu * Im

    # Compute numerator
    y2 = numpy.linalg.solve(A, y1)
    y3 = X.dot(y2)
    y = z - y3
    numerator = numpy.linalg.norm(y)**2 / n

    # Compute denominator
    if n > m:
        if TI is not None:
            # Interpolate trace of inverse
            schatten_mu = TI.interpolate(mu-shift)
            trace_mu = m / schatten_mu
            trace = n - m + mu * trace_mu
        else:
            # Compute the exact value of the trace of inverse
            time0 = process_time()
            # Ainv = numpy.linalg.inv(A)
            # trace = n - m + mu * numpy.trace(Ainv)
            trace = n - m + mu * traceinv(A, method='cholesky')
            time1 = process_time()
            if time_counter is not None:
                time_counter.add(time1 - time0)
    else:
        if TI is not None:
            # Interpolate trace of inverse
            schatten_mu = TI.interpolate(mu-shift)
            trace_mu = n / schatten_mu
            trace = mu * trace_mu
        else:
            time0 = process_time()
            In = numpy.eye(n)
            B = X.dot(X.T) + mu * In
            # Binv = numpy.linalg.inv(B)
            # trace = mu * numpy.trace(Binv)
            trace = mu * traceinv(B, method='cholesky')
            time1 = process_time()
            if time_counter is not None:
                time_counter.add(time1 - time0)

    denominator = (trace / n)**2

    GCV = numerator / denominator

    return GCV


# ============
# minimize gcv
# ============

def minimize_gcv(X, K, z, TI, shift, theta_bounds, initial_elapsed_time,
                 time_counter):
    """
    Finds the parameter ``theta`` such that GCV is minimized.

    In this function, ``theta`` in logarithmic scale.
    """

    print('\nMinimize GCV ...')

    # Use theta in log scale
    use_log_theta = True
    bounds = [(numpy.log10(theta_bounds[0]), numpy.log10(theta_bounds[1]))]
    tolerance = 1e-4
    # guess_log_theta_ = -4

    # Partial function to minimize
    gcv_partial_function = partial(generalized_cross_validation,
                                   X, K, z, TI, shift, time_counter,
                                   use_log_theta)

    # Optimization methods
    time0 = process_time()

    # Local optimization method (use for both direct and presented method)
    # Method = 'Nelder-Mead'
    # results = scipy.optimize.minimize(gcv_partial_function,
    #       guess_log_theta_, method=Method, tol=tolerance,
    #       options={'maxiter':1000, 'xatol':tolerance, 'fatol':tolerance,
    #                'disp':True})
    #       callback=MinimizeTerminatorObj.__call__,

    # Global optimization methods (use for direct method)
    numpy.random.seed(31)   # for repeatability of results
    results = scipy.optimize.differential_evolution(
            gcv_partial_function, bounds, workers=1,
            tol=tolerance, atol=tolerance, updating='deferred', polish=True,
            strategy='best1bin', popsize=40, maxiter=200)  # Works well
    # results = scipy.optimize.dual_annealing(gcv_partial_function, bounds,
    #                                     maxiter=500)
    # results = scipy.optimize.shgo(gcv_partial_function, bounds,
    #             options={'minimize_every_iter': True, 'local_iter': True,
    #                      'minimizer_kwargs':{'method': 'Nelder-Mead'}})
    # results = scipy.optimize.basinhopping(gcv_partial_function,
    #                                   x0=guess_log_theta_)

    print(results)

    # Brute Force optimization method (use for direct method)
    # rranges = ((0.1, 0.3), (0.5, 25))
    # results = scipy.optimize.brute(gcv_partial_function, ranges=rranges,
    #                            full_output=True, finish=scipy.optimize.fmin,
    #                            workers=-1, Ns=30)
    # Optimal_DecorrelationScale = results[0][0]
    # Optimal_nu = results[0][1]
    # max_lp = -results[1]
    # Iterations = None
    # Message = "Using brute force"
    # Success = True

    time1 = process_time()
    elapsed_time = initial_elapsed_time + time1 - time0
    print('Elapsed time: %f\n' % elapsed_time)

    results = {
        'min_gcv': results.fun,
        'min_log_theta_': results.x[0],
        'min_theta_': 10.0**results.x[0],
        'fun_evaluations': results.nfev,
        'elapsed_time': elapsed_time
    }

    return results


# =================================
# plot generalized cross validation
# =================================

def plot_generalized_cross_validation(data, test):
    """
    Plots GCV for a range of theta_.

    data is a list of dictionaries, ``data[0], data[1], ...``.
    Each dictionary ``data[i]`` has the fields:

        * ``'theta_'``: x axis, this is the same for all dictionaries in the
          list.
        * ``'GCV'``: y axis data.
        * ``'label'``: the label of the data GCV in the plot.
    """

    # Load plot settings
    load_plot_settings()

    # Create a list of one item if data is not a list.
    if not isinstance(data, list):
        data = [data]

    theta_ = data[0]['theta_']

    textwidth = 5.25  # in inches
    fig, ax = plt.subplots(figsize=(textwidth, textwidth/1.458))
    colors_list = ["#000000",
                   "#1f77b4",
                   "#2ca02c",
                   "#d62728"]

    h_list = []
    for i in range(len(data)):
        h, = ax.semilogx(theta_, data[i]['GCV'], label=data[i]['label'],
                         color=colors_list[i])
        h_list.append(h)
        ax.semilogx(data[i]['minimization_result']['min_theta_'],
                    data[i]['minimization_result']['min_gcv'], 'o',
                    color=colors_list[i], markersize=3)

    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$V(\theta)$')
    ax.set_title('Generalized Cross Validation')
    ax.set_xlim([theta_[0], theta_[-1]])

    GCV = data[0]['GCV']
    ax.set_yticks([numpy.min(GCV), numpy.min(GCV[:GCV.size//5])])
    ax.set_ylim([0.1634, numpy.max(GCV)+0.0001])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    ax.grid(True, axis='y')
    ax.legend(frameon=False, fontsize='small', bbox_to_anchor=(0.56, 0.24),
              loc='lower left')

    plt.tight_layout()

    # Save Plot
    filename = 'generalized_cross_validation'
    if test:
        filename = "test_" + filename
    save_plot(plt, filename)

    # If no display backend is enabled, do not plot in the interactive mode
    if (not test) and (matplotlib.get_backend() != 'agg'):
        plt.show()


# ====
# main
# ====

def main(test=False):
    """
    Run the script by

    ::

        python examples/Plot_generalized_cross_validation.py

    The script generates the figure below and prints the processing times of
    the computations. See more details in Figure 3 and results of Table 2 of
    [Ameli-2020]_.

    .. image:: https://raw.githubusercontent.com/ameli/imate/main/docs/images/g
               eneralized_cross_validation.svg
       :width: 550
       :align: center

    .. note::
        To *plot* GCV and trace estimations, compute trace with ``cholesky``,
        and *with* matrix inverse. That is, set ``invert_cholesky=True``.

    .. note::
        To properly *measure the elapsed-time* of minimizing GCV, do the
        followings:

        1. in the :func:`minimize_gcv`, use the *Differential Evolution*
           method, and  set ``worker=1`` (**NOT** ``-1``).
        2. Definitely call the function
            :func:`restrict_computation_to_single_processor()`
            to disable any multi-core processing. By this, all computations are
            forced to execute on a single thread. Otherwise, all measured
            elapsed times will be wrong due to the parallel processing.
            The only way that seems to measure elapsed time of multicore
            process properly is to prevent python to use multi-cores.
        3. Set the bound of search for ``theta`` to ``10e-16`` to ``10e+16``.
        4. Trace should be computed by either:
            * Hutchinson method
            * Cholesky factorization and without computing Inverse (set
            ``invert_cholesky=False``).

    .. warning::
        To compute the elapsed-time, do not compute trace with *stochastic
        Lanczos Quadrature* method, since for very small ``theta``, the
        tri-diagonalization fails.

    .. note::
        In the *rational polynomial functions * method for interpolation (using
        ``method='RPF'``), the variable ``q`` in the code represents the number
        of interpolant points. However, in the paper [Ameli-2020], the variable
        ``q`` represents the degree of the rational polynomial, As such, in the
        code we have ``q = 2``, and ``q = 4``, (num points) but in the plots in
        this script, they are labeled as ``q = 1`` and ``q = 2`` (degree of the
        rational polynomial).

    **References**

    .. [Ameli-2020] Ameli, S., and Shadden. S. C. (2020). Interpolating the
       Trace of the Inverse of Matrix **A** + t **B**. `arXiv:2009.07385
       <https://arxiv.org/abs/2009.07385>`__ [math.NA]
    """

    # When measuring elapsed time, restrict number of processors to a single
    # core only to measure time properly
    restrict_computation_to_single_processor()

    # shift to make singular matrix non-singular
    # shift = 2e-4
    # shift = 4e-4
    shift = 1e-3

    # Generate a nearly singular matrix
    if test:
        n = 100
        m = 50
    else:
        n = 1000
        m = 500
    noise_level = 4e-1
    X = generate_basis_functions(n, m)
    z = generate_noisy_data(X, noise_level)
    K = generate_matrix(X, n, m, shift)

    # Interpolating points
    scale = 5.0
    interpolant_points = [
            None,
            [1e-3, 1e-1],
            scale * numpy.logspace(-3, 0, 4),
            scale * numpy.logspace(-3, 0, 6)
    ]

    # Interpolating method
    # For plotting GCV and traces, use Cholesky with setting
    # invert_cholesky=True. However, to measure elapsed time, set
    # invert_cholesky=False.
    options = {'method': 'cholesky', 'invert_cholesky': True}
    kind = 'RPF'
    p = -1

    # Iterate over different set of interpolation points
    TI = []
    initial_elapsed_time = []

    for i in range(len(interpolant_points)):

        if interpolant_points[i] is not None:
            time0 = process_time()
            TI_ = InterpolateSchatten(K, p=p, ti=interpolant_points[i],
                                      kind=kind, options=options)
            time1 = process_time()
            initial_elapsed_time_ = time1 - time0
        else:
            TI_ = None
            initial_elapsed_time_ = 0.0

        TI.append(TI_)
        initial_elapsed_time.append(initial_elapsed_time_)

    # Minimize GCV
    # theta_bounds = (1e-4, 1e1)
    theta_bounds = (1e-16, 1e16)

    res = []
    for i in range(len(interpolant_points)):
        if TI[i] is None:
            time_counter = TimeCounter()
            res_ = minimize_gcv(X, K, z, None, shift, theta_bounds,
                                initial_elapsed_time[i], time_counter)
        else:
            time_counter = None
            res_ = minimize_gcv(X, K, z, TI[i], shift, theta_bounds,
                                initial_elapsed_time[i], time_counter)
        res.append(res_)

    # Compute GCV for a range of theta_
    if test:
        theta_Resolution = 50
    else:
        theta_Resolution = 500
    theta_ = numpy.logspace(-7, 1, theta_Resolution)

    plot_data = []
    for i in range(len(interpolant_points)):
        GCV = numpy.empty(theta_.size)
        use_log_theta = False
        for j in range(theta_.size):
            GCV[j] = generalized_cross_validation(X, K, z, TI[i], shift, None,
                                                  use_log_theta, theta_[j])

        if interpolant_points[i] is None:
            label = 'No interpolation (Exact)'
        else:
            label = 'Interpolation, $q = %d$' \
                    % (len(interpolant_points[i]) // 2)

        # Make a dictionary list of data for plots
        plot_data_ = {
            'theta_': theta_,
            'GCV': GCV,
            'label': label,
            'minimization_result': res[i]
        }

        # plot_data = [plot_data0, plot_data1, plot_data2]
        plot_data.append(plot_data_)

    # Plots
    plot_generalized_cross_validation(plot_data, test)


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(main())
