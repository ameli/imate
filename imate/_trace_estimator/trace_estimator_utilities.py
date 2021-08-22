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

import re
import numpy
import scipy.sparse
from ..linear_operator import LinearOperator, Matrix


# ============
# get operator
# ============

def get_operator(A):
    """
    Check the input operator (or matrix) has proper type and shape. If A is a
    numpy dense matrix or a scipy sparse matrix, it will be converted to an
    instance of :class:`ConstantMatrix` class.
    """

    if isinstance(A, numpy.ndarray) or scipy.sparse.issparse(A):

        # Check matrix dimension and shape
        if A.ndim != 2:
            raise ValueError('Input matrix should be a 2-dimensional array.')
        elif A.shape[0] != A.shape[1]:
            raise ValueError('Input matrix should be a square matrix.')

        # Convert matrix A to a linear operator A
        return Matrix(A)

    elif isinstance(A, LinearOperator):
        if A.get_num_rows() != A.get_num_columns():
            raise ValueError('Input operator should have the same number ' +
                             'of rows and columns.')

        # A is already an object of LinearOperator
        return A

    else:

        raise TypeError('The linear operator "A" should be either a ' +
                        'numpy.ndarray, a scipy.sparse array, or an ' +
                        'instance of a class derived from the ' +
                        '"LinearOperator" class.')


# =======================
# get operator parameters
# =======================

def get_operator_parameters(parameters, data_type_name):
    """
    Checks the type of parameters and returns parameters in a proper type if a
    type conversion is needed (such as list to numpy array). It also returns
    the parameters size as follows:

    * If the parameters is None, or a scalar, the parameters size is 1.
    * If the parameters is a list or array, it returns its size.
    """

    # Check parameters
    if parameters is None:
        parameters_size = 0

    elif numpy.isscalar(parameters):
        parameters_size = 1

    elif isinstance(parameters, list):
        parameters = numpy.array(parameters, dtype=data_type_name)
        if parameters.ndim != 1:
            raise ValueError('"parameters", if given as an array, should be ' +
                             'a one-dimensional array. Current dimension of ' +
                             'parameters is %d.' % parameters.ndim)
        else:
            parameters_size = parameters.size

    elif isinstance(parameters, numpy.ndarray):
        if parameters.ndim != 1:
            raise ValueError('"parameters", if given as an array, should be ' +
                             'a one-dimensional array. Current dimension of ' +
                             'parameters is %d.' % parameters.ndim)
        else:
            parameters_size = parameters.size

    else:
        raise TypeError('"parameters" type should be either "None", a ' +
                        'scalar, a flat list, or one-dimensional ' +
                        '"numpy.ndarray.')

    return parameters, parameters_size


# ===============
# check arguments
# ===============

def check_arguments(
        gram,
        exponent,
        min_num_samples,
        max_num_samples,
        error_atol,
        error_rtol,
        confidence_level,
        outlier_significance_level,
        lanczos_degree,
        lanczos_tol,
        orthogonalize,
        num_threads,
        num_gpu_devices,
        verbose,
        plot,
        gpu):
    """
    Checks if the input arguments have proper type and values.


    :return: If ``error_atol`` or ``error_rtol`` are ``None``, it
        converts them to ``0`` and returns them.
    :rtype: data_type
    """

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
    elif isinstance(exponent, complex):
        TypeError('"exponent" cannot be a complex number.')

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
    elif not isinstance(error_atol, (int, float)):
        raise TypeError('"error_atol" should be a float number.')
    elif error_atol < 0.0:
        raise ValueError('"error_atol" cannot be negative.')

    # Check error relative tolerance
    if error_rtol is None:
        error_rtol = 0.0
    elif not numpy.isscalar(error_rtol):
        raise TypeError('"error_rtol" should be a scalar value.')
    elif not isinstance(error_rtol, (int, float)):
        raise TypeError('"error_rtol" should be a float number.')
    elif error_rtol < 0.0:
        raise ValueError('"error_rtol" cannot be negative.')

    # Check confidence level
    if confidence_level is None:
        raise TypeError('"confidence_level" cannot be None.')
    elif not numpy.isscalar(confidence_level):
        raise TypeError('"confidence_level" should be a scalar.')
    elif not isinstance(confidence_level, (int, float)):
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

    # Check lanczos_degree
    if lanczos_degree is None:
        raise TypeError('"lanczos_degree" cannot be None.')
    elif not numpy.isscalar(lanczos_degree):
        raise TypeError('"lanczos_degree" should be a scalar value.')
    elif not isinstance(lanczos_degree, (int, numpy.integer)):
        raise TypeError('"lanczos_degree" should be an integer.')
    elif lanczos_degree is None:
        raise ValueError('"lanczos_degree" cannot be None.')
    elif lanczos_degree < 1:
        raise ValueError('"lanczos_degree" should be at least one.')

    # Check lanczos tol
    if lanczos_tol is not None and not numpy.isscalar(lanczos_tol):
        raise TypeError('"lanczos_tol" should be a scalar value.')
    elif lanczos_tol is not None and not isinstance(
            lanczos_tol, (int, numpy.integer, float)):
        raise TypeError('"lanczos_tol" should be a float number.')
    elif lanczos_tol is not None and lanczos_tol < 0.0:
        raise ValueError('"lancozs_tol" cannot be negative.')

    # Check orthogonalize
    if orthogonalize is None:
        raise TypeError('"orthogonalize" cannot be None.')
    elif not numpy.isscalar(orthogonalize):
        raise TypeError('"orthogonalize" should be a scalar value.')
    elif not isinstance(orthogonalize, (int, numpy.integer)):
        raise TypeError('"orthogonalize" should be an integer.')
    elif orthogonalize > lanczos_degree:
        raise ValueError('"orthogonalize", if positive, should be at most ' +
                         'equal to "lanczos_degree".')

    # Check num_threads
    if num_threads is None:
        raise TypeError('"num_threads" cannot be None.')
    elif not numpy.isscalar(num_threads):
        raise TypeError('"num_threads" should be a scalar value.')
    elif not isinstance(num_threads, (int, numpy.integer)):
        raise TypeError('"num_threads" should be an integer.')
    elif num_threads < 0:
        raise ValueError('"num_threads" should be a non-negative integer.')

    # Check num_gpu_devices
    if num_gpu_devices is None:
        raise TypeError('"num_gpu_devices" cannot be None.')
    elif not numpy.isscalar(num_gpu_devices):
        raise TypeError('"num_gpu_devices" should be a scalar value.')
    elif not isinstance(num_gpu_devices, (int, numpy.integer)):
        raise TypeError('"num_gpu_devices" should be an integer.')
    elif num_gpu_devices < 0:
        raise ValueError('"num_gpu_devices" should be a non-negative integer.')

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

    # Check gpu
    if gpu is None:
        raise TypeError('"gpu" cannot be None.')
    elif not numpy.isscalar(gpu):
        raise TypeError('"gpu" should be a scalar value.')
    elif not isinstance(gpu, bool):
        raise TypeError('"gpu" should be boolean.')

    return error_atol, error_rtol


# =====================
# get machine precision
# =====================

def get_machine_precision(data_type_name):
    """
    Returns the machine precision used for the "epsilon" value of the
    stochastic quadrature method.

    The machine's epsilon precision:

    * for 32-bit precision data is 2**(-23) = 1.1920929e-07.
    * for 64-bit precision data is 2**(-52) = 2.220446049250313e-16.
    * for 128-bit precision data is 2**(-63) = -1.084202172485504434e-19.

    :param data_type_name: A string indicating the data type, such as
        ``b'float32'``, ``b'float64'``, or ``b'float128'``.
    :type data_type_name: string

    :return: Machine precision for the specified data type.
    :rtype: float
    """

    return numpy.finfo(data_type_name).eps


# ==================
# find num inquiries
# ==================

def find_num_inquiries(Aop, parameters_size):
    """
    Find the number of parameter inquiries. Parameter inquiries may not be the
    same as the length of parameters array. The "parameter size" is the product
    of the "number of the parameter inquiries" and the "number of the
    parameters of the operator".
    """

    # Get number of parameters that the linear operator Aop defined with
    num_operator_parameters = Aop.get_num_parameters()

    # Find the number of inquiries based on the number of parameters
    # and the length (size) of the parameters array (or scalar)
    if num_operator_parameters == 0:

        # No parameters given. There is only one inquiry to be computed
        num_inquiries = 1

    elif parameters_size < num_operator_parameters:
        message = """Not enough parameters are provided. The given parameter
                     size is %d, which should be larger than %d, the number of
                     parameters of the linear operator.""" % (
                        parameters_size, num_operator_parameters)
        raise ValueError(re.sub(' +', ' ', message.replace('\n', '')))

    elif parameters_size % num_operator_parameters != 0:
        message = """Parameters size (given %d) is not an integer-multiple of
                     %d, the number of the parameters of the linear operator.
                     """ % (parameters_size, num_operator_parameters)
        raise ValueError(re.sub(' +', ' ', message.replace('\n', '')))

    else:
        # parameters size should be an integer multiple of num parameters
        num_inquiries = int(parameters_size / num_operator_parameters)

    return num_inquiries


# =============
# print summary
# =============

def print_summary(info):
    """
    """

    # Matrix info
    data_type = info['matrix']['data_type'].decode("utf-8")
    gram = info['matrix']['gram']
    exponent = info['matrix']['exponent']
    num_inquiries = info['matrix']['num_inquiries']
    num_operator_parameters = info['matrix']['num_operator_parameters']
    parameters = info['matrix']['parameters']

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
    lanczos_degree = info['solver']['lanczos_degree']
    lanczos_tol = info['solver']['lanczos_tol']
    orthogonalize = info['solver']['orthogonalize']

    # Convert orthogonalize to "full" or "none" strings
    if orthogonalize < 0 or orthogonalize >= lanczos_degree:
        orthogonalize = 'full'
    elif orthogonalize == 0:
        orthogonalize = 'none'
    else:
        orthogonalize = "%4d" % orthogonalize

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
        if num_operator_parameters == 0:
            print(' %8s  none  ' % "", end="")
        elif num_operator_parameters == 1:
            print('%15.4e  ' % parameters[i], end="")
        else:
            print('[%7.2e, ...]  ' %
                  (parameters[i*num_operator_parameters]), end="")

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
    print('gram:                           %5s' % gram, end="    ")
    print('method:                           slq')
    print('float precision:             %8s' % data_type, end="    ")
    print('lanczos degree:                  %4d' % lanczos_degree)
    print('num matrix parameters:             %2d' % num_operator_parameters,
          end="    ")
    print('lanczos tol:                %8.3e' % lanczos_tol)
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
