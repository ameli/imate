#! /usr/bin/env python

# =======
# Imports
# =======

import os
from os.path import join
import sys
import pickle
import numpy
import scipy
import scipy.sparse
from imate import traceinv
from imate import AffineMatrixFunction                             # noqa: F401
from imate.sample_matrices import band_matrix
import subprocess
import multiprocessing
import platform
import re
from datetime import datetime


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


# =========
# benchmark
# =========

def benchmark(argv):
    """
    Test for :mod:`imate.traceinv` sub-package.
    """

    # Settings
    config = {
        'num_repeats': 10,
        'symmetric': False,
        'exponent': 1,
        'min_num_samples': 200,
        'max_num_samples': 200,
        'lanczos_degree': 80,
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
        'invert_cholesky': True
    }

    matrix = {
        'size': 2**8,
        't': numpy.logspace(-3, 3, 50),
        'band_alpha': 2.0,
        'band_beta': 1.0,
        'symmetric': False,
        'format': 'csr',
        'dtype': r'float64'
    }

    devices = {
        'cpu_name': get_processor_name(),
        'num_all_cpu_threads': multiprocessing.cpu_count(),
    }

    # Generate matrix
    M = band_matrix(matrix['band_alpha'], matrix['band_beta'], matrix['size'],
                    symmetric=matrix['symmetric'],
                    format=matrix['format'], dtype=matrix['dtype'])

    # filename = '/home/sia/Downloads/imate-results/matrices/'
    Mop = AffineMatrixFunction(M)

    print('SLQ method ...', end='')
    trace, info = traceinv(
            Mop,
            parameters=matrix['t'],
            method='slq',
            exponent=config['exponent'],
            symmetric=config['symmetric'],
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
    for i in range(trace_exact.size):
        print('Processing cholesky %d/%d ...'
              % (i+1, trace_exact.size), end='')
        Mt = M + matrix['t'][i] * Identity
        trace_exact[i], _ = traceinv(
                Mt.tocsr(),
                method='cholesky',
                cholmod=False,
                invert_cholesky=config['invert_cholesky'])
                # invert_cholesky=False)
        print(' done.')

    now = datetime.now()

    # Final object of all results
    benchmark_results = {
        'config': config,
        'matrix': matrix,
        'devices': devices,
        'data_result': data_result,
        'trace_exact': trace_exact,
        'date': now.strftime("%d/%m/%Y %H:%M:%S")
    }

    # Save to file (orth)
    benchmark_dir = '..'
    pickle_dir = 'pickle_results'
    output_filename = 'affine_matrix_function'
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
