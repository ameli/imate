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
import getopt
import pickle
import numpy
import imate
from imate import traceinv
from imate import Matrix
from imate import AffineMatrixFunction                             # noqa: F401
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
At last one (or both) of the followings should be provided:

    -o --orthogonalize      Computes Lanczos iterations with orthogonalization.
    -n --not-orthogonalize  Computes Lanczos iterations without
                            orthogonalization.
        """

        print(usage_string)
        print(options_string)

    # -----------------

    # Initialize variables (defaults)
    arguments = {
        'orthogonalize': False,
        'not-orthogonalize': False
    }

    # Get options
    try:
        opts, args = getopt.getopt(
            argv[1:], "on", ["northogonalize", "not-orthognalize"])
    except getopt.GetoptError:
        print_usage(argv[0])
        sys.exit(2)

    # Assign options
    for opt, arg in opts:
        if opt in ('-o', '--orthogonalize'):
            arguments['orthogonalize'] = True
        elif opt in ('-n', '--not-orthogonalize'):
            arguments['not-orthogonalize'] = True

    if len(argv) < 2:
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

    # Settings
    config = {
        'num_repeats': 20,
        'gram': False,
        'exponent': 1,
        'min_num_samples': 200,
        'max_num_samples': 200,
        'lanczos_degree': numpy.logspace(1, 3, 30).astype(int),
        'lanczos_tol':  None,
        'orthogonalize': [],
        'solver_tol': 1e-6,
        'error_rtol': 1e-3,
        'error_atol': 0,
        'confidence_level': 0.95,
        'outlier_significance_level': 0.01,
        'verbose': False,
        'plot': False,
        'num_threads': 0
    }

    devices = {
        'cpu_name': imate.device.get_processor_name(),
        'num_all_cpu_threads': imate.device.get_num_cpu_threads(),
    }

    # Parse arguments
    arguments = parse_arguments(argv)
    if arguments['not-orthogonalize']:
        config['orthogonalize'].append(0)
    if arguments['orthogonalize']:
        config['orthogonalize'].append(-1)

    benchmark_dir = '..'
    # data_name = 'nos5'         # Matrix size: about 500
    # data_name = 'mhd4800b'     # matrix size: about 5K
    # data_name = 'bodyy6'       # matrix size: about 20K
    data_name = 'jnlbrng1'       # matrix size: about 40K
    # data_name = 'G2_circuit'   # matrix size: about 150K

    data_type = '64'
    filename = data_name + '_float' + data_type + '.pickle'
    filepath = join(benchmark_dir, 'matrices', filename)
    with open(filepath, 'rb') as h:
        M = pickle.load(h)
    print('loaded %s.' % filename)

    Mop = Matrix(M)

    matrix = {
        'data_name': data_name,
        'data_type': data_type,
        'matrix_size': M.shape[0],
        'matrix_nnz': M.nnz
    }

    # Exact solution using Cholesky method
    print('Exact solution ...', end='')
    trace_c, info_c = traceinv(
            M,
            return_info=True,
            method='cholesky',
            invert_cholesky=False)
    print(' done.')

    exact_results = {
        'trace': trace_c,
        'info': info_c
    }

    data_results = {
        'not-orthogonalize': [],
        'orthogonalize': []
    }

    for orthogonalize in config['orthogonalize']:

        print('orthogonalize: %d' % orthogonalize)

        # Loop over data filenames
        for lanczos_degree in config['lanczos_degree']:

            print('\tlancos_degree: %d ...' % lanczos_degree)

            trace = numpy.zeros((config['num_repeats'], ), dtype=float)
            absolute_error = numpy.zeros((config['num_repeats'], ),
                                         dtype=float)
            tot_wall_time = numpy.zeros((config['num_repeats'], ), dtype=float)
            alg_wall_time = numpy.zeros((config['num_repeats'], ), dtype=float)
            cpu_proc_time = numpy.zeros((config['num_repeats'], ), dtype=float)

            for i in range(config['num_repeats']):
                print('\t\trepeat %d ...' % (i+1), end="")
                trace[i], info = traceinv(
                        Mop,
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
                        lanczos_degree=int(lanczos_degree),
                        lanczos_tol=config['lanczos_tol'],
                        orthogonalize=orthogonalize,
                        num_threads=config['num_threads'],
                        verbose=config['verbose'],
                        plot=config['plot'],
                        gpu=False)
                print(' done.')

                absolute_error[i] = info['error']['absolute_error']
                tot_wall_time[i] = info['time']['tot_wall_time']
                alg_wall_time[i] = info['time']['alg_wall_time']
                cpu_proc_time[i] = info['time']['cpu_proc_time']

            # Taking average of repeated values
            # trace = numpy.mean(trace)
            # trace = trace[-1]
            # absolute_error = numpy.mean(absolute_error)
            # absolute_error = absolute_error[-1]
            # tot_wall_time = numpy.mean(tot_wall_time)
            # alg_wall_time = numpy.mean(alg_wall_time)
            # cpu_proc_time = numpy.mean(cpu_proc_time)

            # Reset values with array of repeated experiment
            info['error']['absolute_error'] = absolute_error
            info['time']['tot_wall_time'] = tot_wall_time
            info['time']['alg_wall_time'] = alg_wall_time
            info['time']['cpu_proc_time'] = cpu_proc_time

            result = {
                'trace': trace,
                'info': info
            }

            print('')
            if orthogonalize:
                data_results['orthogonalize'].append(result)
            else:
                data_results['not-orthogonalize'].append(result)

    now = datetime.now()

    # Final object of all results
    benchmark_results = {
        'config': config,
        'matrix': matrix,
        'devices': devices,
        'data_results': data_results,
        'exact_results': exact_results,
        'date': now.strftime("%d/%m/%Y %H:%M:%S")
    }

    # Save to file (orth)
    benchmark_dir = '..'
    pickle_dir = 'pickle_results'
    output_filename_base = 'vary_lanczos_degree_practical_matrix'
    if arguments['orthogonalize']:
        output_filename = output_filename_base + '_ortho'
        output_filename += '.pickle'
        output_full_filename = join(benchmark_dir, pickle_dir, output_filename)
        with open(output_full_filename, 'wb') as file_:
            pickle.dump(benchmark_results, file_,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print('Results saved to %s.' % output_full_filename)

    # Save to file (not orth)
    if arguments['not-orthogonalize']:
        output_filename = output_filename_base + '_not_ortho'
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
