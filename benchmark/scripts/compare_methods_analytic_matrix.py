#! /usr/bin/env python

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

import sys
from os.path import join
import getopt
import numpy
import pickle
from datetime import datetime
import imate
from imate import traceinv, logdet
from imate.sample_matrices import toeplitz


# ===============
# parse arguments
# ===============

def parse_arguments(argv):
    """
    Parses the argument.
    """

    # -----------
    # print usage
    # -----------

    def print_usage(exec_name):
        usage_string = "Usage: " + exec_name + " <arguments>"
        options_string = """
Required arguments:

    -f --function=string  Function can be 'logdet' or 'traceinv' (default).

Required arguments (choose at least one, or more):

    -s --32-bit    Uses single-precision matrices. Default is not to use.
    -d --64-bit    Uses double-precision matrices. Default is not to use,
    -l --128-bit   Uses long-double-precision matrices. Default is not to use.
    -a --all       Uses all 32-bit, 64-bit, and 128-bit precision matrices.
        """

        print(usage_string)
        print(options_string)

    # -----------------

    # Initialize variables (defaults)
    arguments = {
        '32-bit': False,
        '64-bit': False,
        '128-bit': False,
        'function': 'traceinv'
    }

    # Get options
    try:
        opts, args = getopt.getopt(
                argv[1:], "sdlaf:", ["32-bit", "64-bit", "128-bit", "all",
                                     "function="])
    except getopt.GetoptError:
        print_usage(argv[0])
        sys.exit(2)

    # Assign options
    for opt, arg in opts:
        if opt in ('-s', '--32-bit'):
            arguments['32-bit'] = True
        elif opt in ('-d', '--64-bit'):
            arguments['64-bit'] = True
        elif opt in ('-l', '--128-bit'):
            arguments['128-bit'] = True
        elif opt in ('-a', '--all'):
            arguments['32-bit'] = True
            arguments['64-bit'] = True
            arguments['128-bit'] = True
        elif opt in ('-f', '--function'):
            arguments['function'] = arg

    if len(argv) < 2:
        print_usage(argv[0])
        sys.exit()

    return arguments


# ===============
# compare methods
# ===============

def compare_methods(M, config, matrix, arguments):
    """
    Compares speed of slq, hutchinson, and cholesky methods on band matrix.
    """

    if arguments['function'] == 'traceinv':
        function = traceinv
    elif arguments['function'] == 'logdet':
        function = logdet
    else:
        raise ValueError("'function' should be either 'traceinv' or 'logdet'.")

    # SLQ method
    trace_s = numpy.zeros((config['num_repeats'], ), dtype=float)
    absolute_error_s = numpy.zeros((config['num_repeats'], ), dtype=float)
    alg_wall_time_s = numpy.zeros((config['num_repeats'], ), dtype=float)

    for i in range(config['num_repeats']):
        print('\tslq, repeat %d ...' % (i+1), end="")
        trace_s[i], info_s = function(
                M,
                gram=config['gram'],
                p=config['exponent'],
                return_info=True,
                method='slq',
                min_num_samples=config['min_num_samples'],
                max_num_samples=config['max_num_samples'],
                error_rtol=config['error_rtol'],
                error_atol=config['error_atol'],
                confidence_level=config['confidence_level'],
                outlier_significance_level=config[
                    'outlier_significance_level'],
                lanczos_degree=config['lanczos_degree'],
                lanczos_tol=config['lanczos_tol'],
                orthogonalize=config['orthogonalize'],
                num_threads=config['num_threads'],
                verbose=config['verbose'],
                plot=config['plot'],
                gpu=False)
        print(' done.')

        absolute_error_s[i] = info_s['error']['absolute_error']
        alg_wall_time_s[i] = info_s['time']['alg_wall_time']

    # Taking average of repeated values
    # trace_s = numpy.mean(trace_s)
    # trace_s = trace_s[-1]
    # absolute_error_s = numpy.mean(absolute_error_s)
    # absolute_error_s = absolute_error_s[-1]
    # alg_wall_time_s = numpy.mean(alg_wall_time_s)

    # Reset values with array of repeated experiment
    info_s['error']['absolute_error'] = absolute_error_s
    info_s['time']['alg_wall_time'] = alg_wall_time_s

    # Hutchinson method (only for traceinv, 32-bit, and 64-bit)
    if M.shape[0] <= matrix['max_hutchinson_size'] and \
            M.dtype != 'float128' and \
            arguments['function'] == 'traceinv':
        trace_h = numpy.zeros((config['num_repeats'], ), dtype=float)
        absolute_error_h = numpy.zeros((config['num_repeats'], ), dtype=float)
        alg_wall_time_h = numpy.zeros((config['num_repeats'], ), dtype=float)

        for i in range(config['num_repeats']):
            print('\thutchinson, repeat %d ...' % (i+1), end="")
            trace_h[i], info_h = function(
                    M,
                    p=config['exponent'],
                    return_info=True,
                    method='hutchinson',
                    assume_matrix='sym',
                    min_num_samples=config['min_num_samples'],
                    max_num_samples=config['max_num_samples'],
                    error_atol=config['error_atol'],
                    error_rtol=config['error_rtol'],
                    confidence_level=config['confidence_level'],
                    outlier_significance_level=config[
                        'outlier_significance_level'],
                    solver_tol=config['solver_tol'],
                    orthogonalize=bool(config['orthogonalize']),
                    num_threads=config['num_threads'],
                    verbose=False,
                    plot=False)
            print(' done.')

            absolute_error_h[i] = info_h['error']['absolute_error']
            alg_wall_time_h[i] = info_h['time']['alg_wall_time']

        # Taking average of repeated values
        # trace_h = numpy.mean(trace_h)
        # trace_h = trace_h[-1]
        # absolute_error_h = numpy.mean(absolute_error_h)
        # absolute_error_h = absolute_error_h[-1]
        # alg_wall_time_h = numpy.mean(alg_wall_time_h)

        # Reset values with array of repeated experiment
        info_h['error']['absolute_error'] = absolute_error_h
        info_h['time']['alg_wall_time'] = alg_wall_time_h
    else:
        # Takes a long time, do not compute
        trace_h = numpy.nan
        info_h = {}

    # Cholesky method (only for 64-bit)
    if M.shape[0] <= matrix['max_cholesky_size'] and M.dtype == 'float64':
        print('\tcholesky ...', end="")

        if arguments['function'] == 'traceinv':
            trace_c, info_c = function(
                    M,
                    p=config['exponent'],
                    return_info=True,
                    method='cholesky',
                    cholmod=None,
                    invert_cholesky=False)

            trace_c2 = numpy.nan
            info_c2 = {}

        elif arguments['function'] == 'logdet':

            # This uses cholmod (if scikit-sparse is installed), otherwise
            # it only uses scipy.sparse.cholesky
            trace_c, info_c = function(
                    M,
                    p=config['exponent'],
                    return_info=True,
                    method='cholesky',
                    cholmod=None)

            # Is cholmod is used, also compute once more without cholmod
            if info_c['solver']['cholmod_used'] is True and \
                    M.shape[0] <= matrix['max_cholesky_size_2']:
                trace_c2, info_c2 = function(
                        M,
                        p=config['exponent'],
                        return_info=True,
                        method='cholesky',
                        cholmod=False)
            else:
                trace_c2 = numpy.nan
                info_c2 = {}
        print(' done.')

    else:
        # Takes a long time, do not compute
        trace_c2 = numpy.nan
        trace_c = numpy.nan
        info_c = {}
        info_c2 = {}

    # Save all results in a dictionary
    result = {
        'trace_s': trace_s,
        'trace_h': trace_h,
        'trace_c': trace_c,
        'trace_c2': trace_c2,
        'info_s': info_s,
        'info_h': info_h,
        'info_c': info_c,
        'info_c2': info_c2
    }

    return result


# ====
# main
# ====

def main(argv):
    """
    benchmark test for speed and accuracy of slq, hutchinson, and cholesky
    methods.
    """

    # Settings
    config = {
        'num_repeats': 10,
        'gram': False,
        'exponent': 1,
        'min_num_samples': 200,
        'max_num_samples': 200,
        'lanczos_degree': 30,
        'lanczos_tol':  None,
        'solver_tol': 1e-6,
        'orthogonalize': 0,
        'error_rtol': 1e-3,
        'error_atol': 0,
        'confidence_level': 0.95,
        'outlier_significance_level': 0.01,
        'verbose': False,
        'plot': False,
        'num_threads': 0
    }

    matrix = {
        'sizes': 2**numpy.arange(5, 26),
        'max_hutchinson_size': 2**22,
        'max_cholesky_size': 2**13,      # for using cholmod
        'max_cholesky_size_2': 2**16,    # for not using cholmod (logdet only)
        'band_alpha': 2.0,
        'band_beta': 1.0,
        'gram': True,
        'format': 'csr',
    }

    devices = {
        'cpu_name': imate.device.get_processor_name(),
        'num_all_cpu_threads': imate.device.get_num_cpu_threads(),
    }

    data_results_32bit = []
    data_results_64bit = []
    data_results_128bit = []
    arguments = parse_arguments(argv)

    # Computing logdet with cholesky method is very efficient. So, do not limit
    # the matrix size for cholesky method of function is logdet.
    if arguments['function'] == 'logdet':
        matrix['max_cholesky_size'] = 2**26
        matrix['max_cholesky_size_2'] = 2**23

    # 32-bit
    if arguments['32-bit']:
        for size in matrix['sizes']:
            print('Processing %s matrix size: %d ...' % ('32-bit', size))

            # Generate matrix
            M = toeplitz(matrix['band_alpha'], matrix['band_beta'], size,
                         gram=matrix['gram'], format=matrix['format'],
                         dtype=r'float32')

            # Run a benchmark for all algorithms
            result = compare_methods(M, config, matrix, arguments)

            # Store results
            data_results_32bit.append(result)

            print('')

    # 64-bit
    if arguments['64-bit']:
        for size in matrix['sizes']:
            print('Processing %s matrix size: %d ...' % ('64-bit', size))

            # Generate matrix
            M = toeplitz(matrix['band_alpha'], matrix['band_beta'], size,
                         gram=matrix['gram'], format=matrix['format'],
                         dtype=r'float64')

            # Run a benchmark for all algorithms
            result = compare_methods(M, config, matrix, arguments)

            # Store results
            data_results_64bit.append(result)

            print('')

    # 128-bit
    if arguments['128-bit']:
        for size in matrix['sizes']:
            print('Processing %s matrix size: %d ...' % ('128-bit', size))

            # Generate matrix
            M = toeplitz(matrix['band_alpha'], matrix['band_beta'], size,
                         gram=matrix['gram'],
                         format=matrix['format'], dtype=r'float128')

            # Run a benchmark for all algorithms
            result = compare_methods(M, config, matrix, arguments)

            # Store results
            data_results_128bit.append(result)

            print('')

    now = datetime.now()

    # Final object of all results
    benchmark_results = {
        'config': config,
        'matrix': matrix,
        'devices': devices,
        'data_results-32bit': data_results_32bit,
        'data_results-64bit': data_results_64bit,
        'data_results-128bit': data_results_128bit,
        'date': now.strftime("%d/%m/%Y %H:%M:%S")
    }

    # Save to file
    benchmark_dir = '..'
    pickle_dir = 'pickle_results'
    if arguments['function'] == 'traceinv':
        output_filename = 'compare_methods_analytic_matrix_traceinv'
    elif arguments['function'] == 'logdet':
        output_filename = 'compare_methods_analytic_matrix_logdet'
    else:
        raise ValueError("'function' should be either 'traceinv' or 'logdet'.")

    output_filename += '.pickle'
    output_full_filename = join(benchmark_dir, pickle_dir, output_filename)
    with open(output_full_filename, 'wb') as file:
        pickle.dump(benchmark_results, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Results saved to %s.' % output_full_filename)


# ===========
# script main
# ===========

if __name__ == "__main__":
    main(sys.argv)
