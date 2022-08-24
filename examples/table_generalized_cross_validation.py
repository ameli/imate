#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# ===========================================
# Restricting computation to single processor
# ===========================================

# VERY IMPORTANT: Without restricting the processor to one core, the results of
# measuring the process time in output table will not be correct. Also, the
# function below must be called before loading other python packages.

from _utilities.processing_time_utilities import \
        restrict_computation_to_single_processor
restrict_computation_to_single_processor()


# =======
# imports
# =======

import sys                                                         # noqa: E402
import numpy                                                       # noqa: E402
import scipy.optimize                                              # noqa: E402
from functools import partial                                      # noqa: E402

# Package Modules
from _utilities.plot_utilities import *                # noqa: F401, F403, E402
from _utilities.processing_time_utilities import TimeCounter       # noqa: E402
from _utilities.processing_time_utilities import process_time      # noqa: E402
from _utilities.data_utilities import generate_matrix              # noqa: E402
from _utilities.data_utilities import generate_noisy_data          # noqa: E402
from _utilities.data_utilities import generate_basis_functions     # noqa: E402
from imate import InterpolateSchatten                              # noqa: E402
from imate import traceinv                                         # noqa: E402


# ===
# fit
# ===

def fit(X, theta, z):
    """
    Computes the regression parameter beta and the fitted values, z_hat.
    """

    n, m = X.shape
    I = numpy.eye(m, m)                                            # noqa: E741
    A = X.T @ X + n*theta * I
    Xtz = numpy.dot(X.T, z)
    beta = numpy.linalg.solve(A, Xtz)
    z_hat = numpy.dot(X, beta)

    return beta, z_hat


# ===================
# relative error beta
# ===================

def relative_error_vector(vector, benchmark_vector):
    """
    This function computes the difference of a vector with the benchmark, and
    normalizes its L2 norm with the norm of the benchmark.
    The result is in percent.
    """

    diff = vector - benchmark_vector

    ord = 2
    p = 1.0
    norm_diff = numpy.linalg.norm(diff, ord=ord)**p
    norm_vector = numpy.linalg.norm(vector, ord=ord)**p
    norm_benchmark_vector = numpy.linalg.norm(
            benchmark_vector, ord=ord)**p

    factor = numpy.max([norm_vector, norm_benchmark_vector])
    relative_error = 100.0 * norm_diff / factor
    # n = vector.size
    # relative_error = numpy.sqrt(numpy.sum(diff**2) / n)

    return relative_error


# ============================
# generalized cross validation
# ============================

def generalized_cross_validation(X, K, z, TI, shift, time_counter, options,
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

        :param shift: shift for the singular matrix ``K0`` to
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
            # trace = n - m + mu * traceinv(A, method='cholesky')[0]
            trace = n - m + mu * traceinv(A, **options)
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
            trace = mu * traceinv(B, **options)
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
                 time_counter, options):
    """
    Finds the parameter ``theta`` such that GCV is minimized.

    In this function, ``theta`` in logarithmic scale.
    """

    # print('\nMinimize GCV ...')

    # Use theta in log scale
    use_log_theta = True
    bounds = [(numpy.log10(theta_bounds[0]), numpy.log10(theta_bounds[1]))]
    tolerance = 1e-4
    # guess_log_theta_ = -4

    # Partial function to minimize
    gcv_partial_function = partial(generalized_cross_validation,
                                   X, K, z, TI, shift, time_counter, options,
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

    # print(results)

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
    # print('Elapsed time: %f\n' % elapsed_time)

    results = {
        'min_gcv': results.fun,
        'min_log_theta_': results.x[0],
        'min_theta_': 10.0**results.x[0],
        'fun_evaluations': results.nfev,
        'total_elapsed_time': elapsed_time
    }

    return results


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

    # Interpolation setting
    kind = 'RPF'
    p = -1

    # Options (cholesky, hutchinson, slq methods)
    num_samples = 30
    lanczos_degree = 30
    all_options = [
            {'method': 'cholesky', 'invert_cholesky': False, 'cholmod': False},
            {'method': 'hutchinson', 'min_num_samples': num_samples,
             'max_num_samples': num_samples, 'orthogonalize': False,
             'assume_matrix': 'sym_pos', 'solver_tol': 1e-6},
            {'method': 'slq', 'min_num_samples': num_samples,
             'max_num_samples': num_samples, 'lanczos_degree': lanczos_degree}
    ]

    # Comparing error with the result of Cholesky method without interpolation
    repeat = 1

    print('-----------------------------------------------------------------' +
          '------------------------')
    print('                Iterations   Process Time          Results       ' +
          '     Relative Error     ')
    print('                -----------  --------------  ------------------- ' +
          ' -----------------------')
    print('Method      q   N_tr  N_tot  T_tr    T_tot   V         log_theta ' +
          ' log_theta  beta   yhat ')
    print('----------  --  ----  -----  ------  ------  -------   --------- ' +
          ' ---------  -----  -----')

    # Initialize benchmark
    benchmark_log_theta = None
    benchmark_beta = None
    benchmark_z_hat = None

    for options in all_options:

        num_interp = len(interpolant_points)
        num_fun_eval = [numpy.zeros((repeat,))] * num_interp
        trace_elapsed_time = [numpy.zeros((repeat,))] * num_interp
        tot_elapsed_time = [numpy.zeros((repeat,))] * num_interp
        min_gcv = [numpy.zeros((repeat,))] * num_interp
        log_theta = [numpy.zeros((repeat,))] * num_interp
        error_log_theta = [numpy.zeros((repeat,))] * num_interp
        error_beta = [numpy.zeros((repeat,))] * num_interp
        error_z_hat = [numpy.zeros((repeat,))] * num_interp

        # Iterate over different set of interpolation points
        for j in range(len(interpolant_points)):

            for i in range(repeat):

                if interpolant_points[j] is not None:
                    time0 = process_time()
                    TI = InterpolateSchatten(K, p=p, ti=interpolant_points[j],
                                             kind=kind, options=options)
                    time1 = process_time()
                    initial_elapsed_time = time1 - time0
                else:
                    TI = None
                    initial_elapsed_time = 0.0

                # Minimize GCV
                # theta_bounds = (1e-4, 1e1)
                theta_bounds = (1e-16, 1e16)

                if TI is None:
                    time_counter = TimeCounter()
                    res = minimize_gcv(X, K, z, TI, shift, theta_bounds,
                                       initial_elapsed_time, time_counter,
                                       options)
                    initial_elapsed_time = time_counter.elapsed_time
                else:
                    time_counter = None
                    res = minimize_gcv(X, K, z, TI, shift, theta_bounds,
                                       initial_elapsed_time, time_counter,
                                       options)

                # Extract results
                num_fun_eval[j][i] = res['fun_evaluations']
                trace_elapsed_time[j][i] = initial_elapsed_time
                tot_elapsed_time[j][i] = res['total_elapsed_time']
                min_gcv[j][i] = res['min_gcv']
                log_theta[j][i] = res['min_log_theta_']

                beta, z_hat = fit(X, res['min_theta_'], z)

                # Benchmark result of theta and beta using Cholesky method
                if ((options['method'] == 'cholesky') and
                   (interpolant_points[j] is None)):
                    benchmark_log_theta = log_theta[0][0]
                    benchmark_beta = beta
                    benchmark_z_hat = z_hat

                # Error of log_theta
                error_log_theta[j][i] = 100.0 * numpy.abs(
                        1.0 - log_theta[j][i] / benchmark_log_theta)

                # Error of beta
                error_beta[j][i] = relative_error_vector(
                        beta, benchmark_beta)

                # Error of z hat
                error_z_hat[j][i] = relative_error_vector(
                        z_hat, benchmark_z_hat)

            # Averaging
            num_fun_eval[j] = int(numpy.mean(num_fun_eval[j]))
            trace_elapsed_time[j] = numpy.mean(trace_elapsed_time[j])
            tot_elapsed_time[j] = numpy.mean(tot_elapsed_time[j])
            min_gcv[j] = numpy.mean(min_gcv[j])
            log_theta[j] = numpy.mean(log_theta[j])
            error_log_theta[j] = numpy.mean(error_log_theta[j])
            error_beta[j] = numpy.mean(error_beta[j])
            error_z_hat[j] = numpy.mean(error_z_hat[j])

            if interpolant_points[j] is None:
                q = 0
                num_trace_eval = num_fun_eval[j]
            else:
                q = len(interpolant_points[j]) // 2
                num_trace_eval = 2*q+1

            # No interpolation
            print('%10s  %2d  %4d  %5d  %6.2f  %6.2f  %0.5f  %+10.5f  '
                  % ('{:<10s}'.format(options['method']), q,
                     num_trace_eval, num_fun_eval[j],
                     trace_elapsed_time[j], tot_elapsed_time[j],
                     min_gcv[j], log_theta[j]), end='')
            print('%9.2f  %5.2f  %5.2f'
                  % (error_log_theta[j], error_beta[j], error_z_hat[j]))

        print('')


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(main())
