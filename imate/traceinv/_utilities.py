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


# ==================
# get data type name
# ==================

def get_data_type_name(A):
    """
    Returns the dtype as string.
    """

    if A.dtype == b'float32':
        data_type_name = b'float32'

    elif A.dtype == b'float64':
        data_type_name = b'float64'

    elif A.dtype == b'float128':
        data_type_name = b'float128'

    else:
        raise TypeError('Data type should be "float32", "float64", or ' +
                        '"float128".')

    return data_type_name


# =======
# get nnz
# =======

def get_nnz(A):
    """
    Returns the number of non-zero elements of a matrix.
    """

    if isspmatrix(A):
        return A.nnz
    else:
        return A.shape[0] * A.shape[1]


# ===========
# get density
# ===========

def get_density(A):
    """
    Returns the density of non-zero elements of a matrix.
    """

    if isspmatrix(A):
        return get_nnz(A) / (A.shape[0] * A.shape[1])
    else:
        return 1.0


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
