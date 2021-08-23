#! /usr/bin/env python

# =======
# Imports
# =======

import os
from os.path import join
import sys
import pickle
import numpy
import getopt
import re
from imate import traceinv
from imate import Matrix
from imate import AffineMatrixFunction                             # noqa: F401
from imate.sample_matrices import band_matrix
import subprocess
import multiprocessing
import platform
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

    -o --openblas=[bool]  Can be 'True' or 'False' to use openblas or not.
        """

        print(usage_string)
        print(options_string)

    # -----------------

    # Initialize variables (defaults)
    arguments = {
        'openblas': False,
    }

    # Get options
    try:
        opts, args = getopt.getopt(
                argv[1:], "o:", ["openblas="])
    except getopt.GetoptError:
        print_usage(argv[0])
        sys.exit(2)

    # Assign options
    for opt, arg in opts:
        if opt in ('-o', '--openblas'):
            if arg == 'True':
                arguments['openblas'] = True
            else:
                arguments['openblas'] = False

    if len(argv) < 2:
        print_usage(argv[0])
        sys.exit()

    return arguments


# ==================
# get processor name
# ==================

def get_processor_name():
    """
    Gets the name of CPU.

    For windows operating system, this function still does not get the full
    brand name of the cpu.
    """

    if platform.system() == "Windows":
        return platform.processor()

    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.getoutput(command).strip()

    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.getoutput(command).strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)[1:]

    return ""


# ============
# get gpu name
# ============

def get_gpu_name():
    """
    Gets the name of gpu device.
    """

    command = 'nvidia-smi -a | grep -i "Product Name" -m 1 | grep -o ":.*"' + \
              ' | cut -c 3-'
    return subprocess.getoutput(command).strip()


# =======================
# get num all gpu devices
# =======================

def get_num_all_gpu_devices():
    """
    Get number of all gpu devices
    """

    command = 'nvidia-smi --list-gpus | wc -l'
    subprocess_result = subprocess.getoutput(command)
    return int(subprocess_result)


# =========
# benchmark
# =========

def benchmark(argv):
    """
    Test for :mod:`imate.traceinv` sub-package.
    """

    data_types = ['32', '64', '128']  # OpenBlas does not support 128-bit

    # Settings
    config = {
        'num_repeats': 10,
        'gram': False,
        'exponent': 1,
        'min_num_samples': 200,
        'max_num_samples': 200,
        'lanczos_degree': 50,
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
        # 'sizes': 2**numpy.arange(8, 17, 2),
        'sizes': 2**numpy.arange(8, 15),
        'band_alpha': 2.0,
        'band_beta': 1.0,
        'gram': True,
        'format': 'csr',
    }

    devices = {
        'cpu_name': get_processor_name(),
        'gpu_name': get_gpu_name(),
        'num_all_cpu_threads': multiprocessing.cpu_count(),
        'num_all_gpu_devices': get_num_all_gpu_devices()
    }

    data_results = []
    arguments = parse_arguments(argv)

    # Loop over data filenames
    for size in matrix['sizes']:

        data_result = {
            'size': size,
            'type_results': [],
        }

        print('matrix size: %d ...' % size)

        # For each data, loop over float type, such as 32-bit, 64-bit, 128-bit
        for data_type in data_types:

            print('\tdata type: %s-bit ...' % data_type)

            if data_type == '32':
                dtype = r'float32'
            elif data_type == '64':
                dtype = r'float64'
            elif data_type == '128':
                dtype = r'float128'

                # openblas does not use 128-bit
                if arguments['openblas']:
                    continue

            else:
                raise ValueError('Invalid data_type: %s.' % data_type)

            # Generate matrix
            M = band_matrix(matrix['band_alpha'], matrix['band_beta'], size,
                            gram=matrix['gram'],
                            format=matrix['format'], dtype=dtype)

            Mop = Matrix(M.toarray())
            # Mop = AffineMatrixFunction(M)

            trace = numpy.zeros((config['num_repeats'], ), dtype=dtype)
            absolute_error = numpy.zeros((config['num_repeats'], ),
                                         dtype=dtype)
            tot_wall_time = numpy.zeros((config['num_repeats'], ), dtype=float)
            alg_wall_time = numpy.zeros((config['num_repeats'], ), dtype=float)
            cpu_proc_time = numpy.zeros((config['num_repeats'], ), dtype=float)

            for i in range(config['num_repeats']):
                print('\t\trepeat %d ...' % (i+1), end="")
                trace[i], info = traceinv(
                        Mop,
                        method='slq',
                        exponent=config['exponent'],
                        gram=config['gram'],
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

                absolute_error[i] = info['error']['absolute_error']
                tot_wall_time[i] = info['time']['tot_wall_time']
                alg_wall_time[i] = info['time']['alg_wall_time']
                cpu_proc_time[i] = info['time']['cpu_proc_time']

            # Taking average of repeated values
            trace = numpy.mean(trace)
            # trace = trace[-1]
            absolute_error = numpy.mean(absolute_error)
            # absolute_error = absolute_error[-1]
            tot_wall_time = numpy.mean(tot_wall_time)
            alg_wall_time = numpy.mean(alg_wall_time)
            cpu_proc_time = numpy.mean(cpu_proc_time)

            # Reset values with array of repeated experiment
            info['error']['absolute_error'] = absolute_error
            info['time']['alg_wall_time'] = alg_wall_time

            result = {
                'trace': trace,
                'info': info
            }
            type_result = {
                'data_type': data_type,
                'result': result,
            }

            print('')
            data_result['type_results'].append(type_result)

        data_results.append(data_result)

    now = datetime.now()

    # Final object of all results
    benchmark_results = {
        'config': config,
        'matrix': matrix,
        'devices': devices,
        'data_results': data_results,
        'date': now.strftime("%d/%m/%Y %H:%M:%S")
    }

    # Save to file
    benchmark_dir = '..'
    pickle_dir = 'pickle_results'
    if arguments['openblas']:
        output_filename = 'benchmark_with_openblas_dense'
    else:
        output_filename = 'benchmark_without_openblas_dense'
    output_filename += '.pickle'
    output_full_filename = join(benchmark_dir, pickle_dir, output_filename)
    with open(output_full_filename, 'wb') as file:
        pickle.dump(benchmark_results, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Results saved to %s.' % output_full_filename)


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(benchmark(sys.argv))
