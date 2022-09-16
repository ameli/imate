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

from os.path import join
import sys
import pickle
import getopt
import numpy
import scipy
import scipy.sparse
import imate
from imate import traceinv, logdet
from imate import AffineMatrixFunction                             # noqa: F401
from imate.sample_matrices import toeplitz
from time import process_time
from datetime import datetime


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
    -g --gram             Uses Gramian matrix instead of the matrix itself.
        """

        print(usage_string)
        print(options_string)

    # -----------------

    # Initialize variables (defaults)
    arguments = {
        'function': 'traceinv',
        'gram': False,
    }

    # Get options
    try:
        opts, args = getopt.getopt(
                argv[1:], "f:g", ["function=", "gram"])
    except getopt.GetoptError:
        print_usage(argv[0])
        sys.exit(2)

    # Assign options
    for opt, arg in opts:
        if opt in ('-f', '--function'):
            arguments['function'] = arg
        elif opt in ('-g', '--gram'):
            arguments['gram'] = True

    if len(argv) < 3:
        print_usage(argv[0])
        sys.exit()

    return arguments


# =========
# benchmark
# =========

def benchmark(argv):
    """
    Test for :mod:`imate.traceinv` sub-package.
    """

    arguments = parse_arguments(argv)

    # Settings
    config = {
        'gram': arguments['gram'],
        'exponent': 1,
        'min_num_samples': 200,
        'max_num_samples': 200,
        'lanczos_degree': 50,
        'lanczos_tol':  None,
        'orthogonalize': -1,
        'solver_tol': 1e-6,
        'error_rtol': 1e-3,
        'error_atol': 0,
        'confidence_level': 0.95,
        'outlier_significance_level': 0.01,
        'verbose': False,
        'plot': False,
        'num_threads': 0,
        'invert_cholesky': False
    }

    matrix = {
        'size': 2**16,
        't': numpy.logspace(-3, 3, 1000),
        'band_alpha': 2.0,
        'band_beta': 1.0,
        'gram': not config['gram'],
        'format': 'csr',
        'dtype': r'float64'
    }

    devices = {
        'cpu_name': imate.device.get_processor_name(),
        'num_all_cpu_threads': imate.device.get_num_cpu_threads(),
    }

    # Generate matrix
    M = toeplitz(matrix['band_alpha'], matrix['band_beta'], matrix['size'],
                 gram=matrix['gram'], format=matrix['format'],
                 dtype=matrix['dtype'])

    Mop = AffineMatrixFunction(M)

    if arguments['function'] == 'traceinv':
        function = traceinv
    elif arguments['function'] == 'logdet':
        function = logdet
    else:
        raise ValueError("'function' should be either 'traceinv' or 'logdet'.")

    print('SLQ method ...', end='')
    trace, info = function(
            Mop,
            gram=config['gram'],
            p=config['exponent'],
            return_info=True,
            method='slq',
            parameters=matrix['t'],
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

    data_result = {
        'trace': trace,
        'info': info
    }

    # Exact solution using Cholesky method
    trace_exact = numpy.zeros((matrix['t'].size, ), dtype=float)
    Identity = scipy.sparse.eye(matrix['size'], format='csr')

    initial_cholesky_cpu_proc_time = process_time()
    for i in range(trace_exact.size):
        print('Processing cholesky %d/%d ...'
              % (i+1, trace_exact.size), end='')
        if config['gram']:
            M_ = M.T @ M
        else:
            M_ = M
        Mt = M_ + matrix['t'][i] * Identity
        if arguments['function'] == 'traceinv':
            trace_exact[i] = function(
                    Mt.tocsr(),
                    gram=False,
                    p=config['exponent'],
                    return_info=False,
                    method='cholesky',
                    cholmod=None,
                    invert_cholesky=config['invert_cholesky'])

        elif arguments['function'] == 'logdet':
            trace_exact[i] = function(
                    Mt.tocsr(),
                    gram=False,
                    p=config['exponent'],
                    return_info=False,
                    method='cholesky',
                    cholmod=None)

        print(' done.')

    # Process time of cholesky method
    cholesky_cpu_proc_time = process_time() - initial_cholesky_cpu_proc_time

    now = datetime.now()

    # Final object of all results
    benchmark_results = {
        'config': config,
        'matrix': matrix,
        'devices': devices,
        'data_result': data_result,
        'trace_exact': trace_exact,
        'date': now.strftime("%d/%m/%Y %H:%M:%S"),
        'cholesky_cpu_proc_time': cholesky_cpu_proc_time,
        'function': arguments['function']
    }

    # Save to file (orth)
    benchmark_dir = '..'
    pickle_dir = 'pickle_results'
    output_filename = 'affine_matrix_function'
    if arguments['function'] == 'traceinv':
        output_filename += '_traceinv'
    else:
        output_filename += '_logdet'
    if arguments['gram']:
        output_filename += '_gram'
    output_filename += '.pickle'
    output_full_filename = join(benchmark_dir, pickle_dir, output_filename)
    with open(output_full_filename, 'wb') as file_:
        pickle.dump(benchmark_results, file_,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print('Results saved to %s.' % output_full_filename)


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(benchmark(sys.argv))
