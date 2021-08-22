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
from scipy.sparse import isspmatrix


# ===============
# check arguments
# ===============

def check_arguments(
        A,
        gram,
        exponent,
        assume_matrix,
        min_num_samples,
        max_num_samples,
        error_atol,
        error_rtol,
        confidence_level,
        outlier_significance_level,
        solver_tol,
        orthogonalize,
        num_threads,
        verbose,
        plot):
    """
    Checks the type and value of the parameters.
    """

    # Check A
    if (not isinstance(A, numpy.ndarray)) and (not isspmatrix(A)):
        raise TypeError('Input matrix should be either a "numpy.ndarray" or ' +
                        'a "scipy.sparse" matrix.')
    elif A.shape[0] != A.shape[1]:
        raise ValueError('Input matrix should be a square matrix.')

    # Check gram
    if gram is None:
        raise TypeError('"gram" cannot be None.')
    elif not numpy.isscalar(gram):
        raise TypeError('"gram" should be a scalar value.')
    elif not isinstance(gram, bool):
        raise TypeError('"gram" should be boolean.')

    # Check exponent
    if exponent is None:
        raise TypeError('"exponent" cannot be None.')
    elif not numpy.isscalar(exponent):
        raise TypeError('"exponent" should be a scalar value.')
    elif not isinstance(exponent, (int, numpy.integer)):
        TypeError('"exponent" cannot be an integer.')

    # Check assume_matrix
    if assume_matrix is None:
        raise ValueError('"assume_matrix" cannot be None.')
    elif not isinstance(assume_matrix, str):
        raise TypeError('"assume_matrix" must be a string.')
    elif assume_matrix != 'gen' and assume_matrix != "pos" and \
            assume_matrix != "sym" and assume_matrix != "sym_pos":
        raise ValueError('"assume_matrix" should be either "gen", "pos", ' +
                         '"sym, or "sym_pos".')

    # Check min_num_samples
    if min_num_samples is None:
        raise ValueError('"min_num_samples" cannot be None.')
    elif not numpy.isscalar(min_num_samples):
        raise TypeError('"min_num_samples" should be a scalar value.')
    elif not isinstance(min_num_samples, (int, numpy.integer)):
        raise TypeError('"min_num_samples" should be an integer.')
    elif min_num_samples < 1:
        raise ValueError('"min_num_samples" should be at least one.')

    # Check max_num_samples
    if max_num_samples is None:
        raise ValueError('"max_num_samples" cannot be None.')
    elif not numpy.isscalar(max_num_samples):
        raise TypeError('"max_num_samples" should be a scalar value.')
    elif not isinstance(max_num_samples, (int, numpy.integer)):
        raise TypeError('"max_num_samples" should be an integer.')
    elif max_num_samples < 1:
        raise ValueError('"max_num_samples" should be at least one.')

    # Check min and max num samples
    if min_num_samples > max_num_samples:
        raise ValueError('"min_num_samples" cannot be greater than ' +
                         '"max_num_samples".')

    # Check convergence absolute tolerance
    if error_atol is None:
        error_atol = 0.0
    elif not numpy.isscalar(error_atol):
        raise TypeError('"error_atol" should be a scalar value.')
    elif not isinstance(error_atol, (int, numpy.integer, float)):
        raise TypeError('"error_atol" should be a float number.')
    elif error_atol < 0.0:
        raise ValueError('"error_atol" cannot be negative.')

    # Check error relative tolerance
    if error_rtol is None:
        error_rtol = 0.0
    elif not numpy.isscalar(error_rtol):
        raise TypeError('"error_rtol" should be a scalar value.')
    elif not isinstance(error_rtol, (int, numpy.integer, float)):
        raise TypeError('"error_rtol" should be a float number.')
    elif error_rtol < 0.0:
        raise ValueError('"error_rtol" cannot be negative.')

    # Check confidence level
    if confidence_level is None:
        raise TypeError('"confidence_level" cannot be None.')
    elif not numpy.isscalar(confidence_level):
        raise TypeError('"confidence_level" should be a scalar.')
    elif not isinstance(confidence_level, (int, numpy.integer, float)):
        raise TypeError('"confidence_level" should be a float number.')
    elif confidence_level < 0.0 or confidence_level > 1.0:
        raise ValueError('"confidence_level" should be between 0 and 1.')

    # Check outlier significance level
    if outlier_significance_level is None:
        raise TypeError('"outlier_significance_level" cannot be None.')
    elif not numpy.isscalar(outlier_significance_level):
        raise TypeError('"outlier_significance_level" should be a scalar.')
    elif not isinstance(outlier_significance_level,
                        (int, numpy.integer, float)):
        raise TypeError('"outlier_significance_level" must be a float number.')
    elif outlier_significance_level < 0.0 or outlier_significance_level > 1.0:
        raise ValueError(
                '"outlier_significance_level" must be in [0, 1] interval.')

    # Compare outlier significance level and confidence level
    if outlier_significance_level > 1.0 - confidence_level:
        raise ValueError('The sum of "confidence_level" and ' +
                         '"outlier_significance_level" should be less than 1.')

    # Check solver tol
    if solver_tol is not None and not numpy.isscalar(solver_tol):
        raise TypeError('"solver_tol" should be a scalar value.')
    elif solver_tol is not None and \
            not isinstance(solver_tol, (int, numpy.integer, float)):
        raise TypeError('"solver_tol" should be a float number.')
    elif solver_tol is not None and solver_tol < 0.0:
        raise ValueError('"lancozs_tol" cannot be negative.')

    # Check orthogonalize
    if orthogonalize is None:
        raise TypeError('"orthogonalize" cannot be None.')
    elif not numpy.isscalar(orthogonalize):
        raise TypeError('"orthogonalize" should be a scalar value.')
    elif not isinstance(orthogonalize, bool):
        raise TypeError('"orthogonalize" should be boolean.')

    # Check num_threads
    if num_threads is None:
        raise TypeError('"num_threads" cannot be None.')
    elif not numpy.isscalar(num_threads):
        raise TypeError('"num_threads" should be a scalar value.')
    elif not isinstance(num_threads, (int, numpy.integer)):
        raise TypeError('"num_threads" should be an integer.')
    elif num_threads < 0:
        raise ValueError('"num_threads" should be a non-negative integer.')

    # Check verbose
    if verbose is None:
        raise TypeError('"verbose" cannot be None.')
    elif not numpy.isscalar(verbose):
        raise TypeError('"verbose" should be a scalar value.')
    elif not isinstance(verbose, bool):
        raise TypeError('"verbose" should be boolean.')

    # Check plot
    if plot is None:
        raise TypeError('"plot" cannot be None.')
    elif not numpy.isscalar(plot):
        raise TypeError('"plot" should be a scalar value.')
    elif not isinstance(plot, bool):
        raise TypeError('"plot" should be boolean.')

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

    return error_atol, error_rtol


# =============
# print summary
# =============

def print_summary(info):
    """
    """

    # Matrix info
    data_type = info['matrix']['data_type'].decode("utf-8")
    exponent = info['matrix']['exponent']
    assume_matrix = info['matrix']['assume_matrix']
    num_inquiries = info['matrix']['num_inquiries']

    # Error info
    absolute_error = info['error']['absolute_error']
    relative_error = info['error']['relative_error'] * 100.0
    error_atol = info['error']['error_atol']
    error_rtol = info['error']['error_rtol'] * 100.0
    confidence_level = info['error']['confidence_level'] * 100.0
    outlier_significance_level = info['error']['outlier_significance_level']

    # Convergence info
    num_samples_used = info['convergence']['num_samples_used']
    num_outliers = info['convergence']['num_outliers']
    min_num_samples = info['convergence']['min_num_samples']
    max_num_samples = info['convergence']['max_num_samples']
    converged = info['convergence']['converged']
    trace = info['convergence']['samples_mean']

    # time
    tot_wall_time = info['time']['tot_wall_time']
    alg_wall_time = info['time']['alg_wall_time']
    cpu_proc_time = info['time']['cpu_proc_time']

    # device
    num_cpu_threads = info['device']['num_cpu_threads']
    num_gpu_devices = info['device']['num_gpu_devices']
    num_gpu_multiprocessors = info['device']['num_gpu_multiprocessors']
    num_gpu_threads_per_multiprocessor = \
        info['device']['num_gpu_threads_per_multiprocessor']

    # Solver info
    solver_tol = info['solver']['solver_tol']
    orthogonalize = info['solver']['orthogonalize']

    # Makes assume_matrix string more readable
    if assume_matrix == "gen":
        assume_matrix = "generic"
    elif assume_matrix == "sym":
        assume_matrix = "symmetric"
    elif assume_matrix == "sym_pos":
        assume_matrix = "symmetric-positive"
    else:
        raise ValueError('"assume_matrix" is invalid.')

    # Print results
    print('                                    results                      ' +
          '             ')
    print('=================================================================' +
          '=============')
    print('     inquiries                            error            sample' +
          's            ')
    print('--------------------              ---------------------   -------' +
          '--           ')
    print('i         parameters       trace    absolute   relative   num   o' +
          'ut  converged')
    print('=================================================================' +
          '=============')

    # Print rows of results
    for i in range(num_inquiries):

        # Column "i"
        print('%-3d  ' % (i+1), end="")

        # Column "parameters"
        print(' %8s  none  ' % "", end="")

        # Data columns, depend whether has one or multiple rows (num_inquiries)
        if num_inquiries == 1:

            # will print one row
            if numpy.isnan(trace) or numpy.isinf(trace):
                print('%10e   ' % trace, end="")
            else:
                print('%+7.3e   ' % trace, end="")
            if numpy.isnan(absolute_error) or numpy.isinf(absolute_error):
                print('%9e  ' % absolute_error, end="")
            else:
                print('%7.3e  ' % absolute_error, end="")
            if numpy.isnan(relative_error) or numpy.isinf(relative_error):
                print('%9f  ' % relative_error, end="")
            else:
                print('%8.3f%%  ' % relative_error, end="")
            print('%4d  %4d' % (num_samples_used, num_outliers), end="")
            print('%11s' % converged)

        else:

            # will print multiple rows
            if numpy.isnan(trace[i]) or numpy.isinf(trace[i]):
                print('%10e  ' % trace[i], end="")
            else:
                print('%+7.3e   ' % trace[i], end="")
            if numpy.isnan(absolute_error[i]) or \
               numpy.isinf(absolute_error[i]):
                print('%9e  ' % absolute_error[i], end="")
            else:
                print('%7.3e  ' % absolute_error[i], end="")
            if numpy.isnan(relative_error[i]) or \
               numpy.isinf(relative_error[i]):
                print('%9f  ' % relative_error[i], end="")
            else:
                print('%8.3f%%  ' % relative_error[i], end="")
            print('%4d  %4d' % (num_samples_used[i], num_outliers[i]), end="")
            print('%11s' % converged[i])

    print('')

    # Print user configurations
    print('                                    config                       ' +
          '             ')
    print('=================================================================' +
          '=============')

    # Prints matrx and stochastic process
    print('                matrix                            ' +
          'stochastic estimator        ')
    print('-------------------------------------    ------------------------' +
          '-------------')
    print('assumption:        %18s' % assume_matrix, end="    ")
    print('method:                    hutchinson')
    print('float precision:             %8s' % data_type, end="    ")
    print('solver tol:                 %8.3e' % solver_tol)
    if int(exponent) == exponent:
        print('exponent:                         %3d' % int(exponent),
              end="    ")
    else:
        print('exponent:                      %6.2f' % exponent, end="    ")
    print('orthogonalization:               %4s' % orthogonalize)
    print('')

    # Prints convergence and error
    print('             convergence                                 ' +
          'error     ')
    print('-------------------------------------    ------------------------' +
          '-------------')
    print('min num samples:                %5d' % min_num_samples, end="    ")
    print('abs error tol:              %8.3e' % error_atol)
    print('max num samples:                %5d' % max_num_samples, end="    ")
    print('rel error tol:                  %4.2f%%' % error_rtol)
    print('outlier significance level:     %4.2f%%'
          % outlier_significance_level, end="    ")
    print('confidence level:              %4.2f%%' % confidence_level)
    print('')

    # Print information about CPU and GPU
    print('                                   process                       ' +
          '             ')
    print('=================================================================' +
          '=============')
    print('                 time                                   device   ' +
          '               ')
    print('-------------------------------------    ------------------------' +
          '-------------')
    print('tot wall time (sec):        %8.3e' % tot_wall_time, end="    ")
    print('num cpu threads:                  %3d' % num_cpu_threads)
    print('alg wall time (sec):        %8.3e' % alg_wall_time, end="    ")
    print('num gpu devices, multiproc:    %2d, %2d'
          % (num_gpu_devices, num_gpu_multiprocessors))
    print('cpu proc time (sec):        %8.3e' % cpu_proc_time, end="    ")
    print('num gpu threads per multiproc:   %4d'
          % num_gpu_threads_per_multiprocessor)
    print('')
