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

import time
import numpy
import numpy.linalg
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import isspmatrix
import multiprocessing
from .._linear_algebra.matrix_utilities import get_data_type_name, get_nnz, \
        get_density
from ..__version__ import __version__


# =================
# eigenvalue method
# =================

def eigenvalue_method(
        A,
        eigenvalues=None,
        gram=False,
        exponent=1.0,
        assume_matrix='gen',
        non_zero_eig_fraction=0.9):
    """
    """

    # Checking input arguments
    check_arguments(A, eigenvalues, gram, exponent, assume_matrix,
                    non_zero_eig_fraction)

    init_tot_wall_time = time.perf_counter()
    init_cpu_proc_time = time.process_time()

    if eigenvalues is None:
        eigenvalues = compute_eigenvalues(A, assume_matrix,
                                          non_zero_eig_fraction)

    # Compute trace of inverse of matrix
    not_nan = numpy.logical_not(numpy.isnan(eigenvalues))
    trace = numpy.sum(numpy.log((eigenvalues[not_nan]**exponent)))

    # Gramian matrix
    if gram:
        trace = 2.0 * trace

    tot_wall_time = time.perf_counter() - init_tot_wall_time
    cpu_proc_time = time.process_time() - init_cpu_proc_time

    # Dictionary of output info
    info = {
        'matrix':
        {
            'data_type': get_data_type_name(A),
            'gram': gram,
            'exponent': exponent,
            'assume_matrix': assume_matrix,
            'size': A.shape[0],
            'sparse': isspmatrix(A),
            'nnz': get_nnz(A),
            'density': get_density(A),
            'num_inquiries': 1
        },
        'device':
        {
            'num_cpu_threads': multiprocessing.cpu_count()
        },
        'time':
        {
            'tot_wall_time': tot_wall_time,
            'alg_wall_time': tot_wall_time,
            'cpu_proc_time': cpu_proc_time,
        },
        'solver':
        {
            'version': __version__,
            'method': 'eigenvalue',
        }
    }

    return trace, info


# ===============
# check arguments
# ===============

def check_arguments(
        A,
        eigenvalues,
        gram,
        exponent,
        assume_matrix,
        non_zero_eig_fraction):
    """
    Checks the type and value of the parameters.
    """

    # Check A
    if (not isinstance(A, numpy.ndarray)) and (not scipy.sparse.issparse(A)):
        raise TypeError('Input matrix should be either a "numpy.ndarray" or ' +
                        'a "scipy.sparse" matrix.')
    elif A.shape[0] != A.shape[1]:
        raise ValueError('Input matrix should be a square matrix.')

    # Check eigenvalues
    if eigenvalues is not None:
        if not isinstance(eigenvalues, numpy.ndarray):
            raise TypeError('"eigenvalues" should be a numpy.ndarray.')
        if eigenvalues.size != A.shape[0]:
            raise ValueError('The length of "eigenvalues" does not match ' +
                             'the size of matrix "A".')

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
    elif not isinstance(exponent, numpy.integer):
        TypeError('"exponent" cannot be an integer.')

    # Check assume_matrix
    if assume_matrix is None:
        raise TypeError('"assume_matrix" cannot be None.')
    elif not isinstance(assume_matrix, str):
        raise TypeError('"assume_matrix" should be a string.')
    elif assume_matrix not in ['gen', 'sym']:
        raise ValueError('"assume_matrix" should be either "gen" or "sym".')

    # Check non_zero_eig_fraction
    if non_zero_eig_fraction is None:
        raise TypeError('"non_zero_eig_fraction" cannot be None.')
    elif not numpy.isscalar(non_zero_eig_fraction):
        raise TypeError('"non_zero_eig_fraction" should be a scalar value.')
    elif not isinstance(non_zero_eig_fraction, float):
        raise TypeError('"non_zero_eig_fraction" should be a float type.')
    elif non_zero_eig_fraction <= 0 or non_zero_eig_fraction >= 1.0:
        raise ValueError('"non_zero_eig_fraction" should be greater then 0.0' +
                         'and smaller than 1.0.')


# ===================
# compute eigenvalues
# ===================

def compute_eigenvalues(
        A,
        assume_matrix,
        non_zero_eig_fraction,
        tol=1e-4):
    """
    """

    if scipy.sparse.isspmatrix(A):

        # Sparse matrix
        n = A.shape[0]
        eigenvalues = numpy.empty(n)
        eigenvalues[:] = numpy.nan

        # find 90% of eigenvalues, assume the rest are very close to zero.
        num_none_zero_eig = int(n*non_zero_eig_fraction)

        if assume_matrix == 'sym':

            # The logarithm function on the eigenvalues (for computing logdet)
            # takes the greatest effet from both the smallest and largest
            # eigenvalues. The logarithm of the smallest eigenvalues is a large
            # negative value, and the logarithm of the largest eigenvalues are
            # positive large values. ``BE`` option in scipy.sparse.linalg.eigh
            # computes half of eigenvalues from each end of the spectrum, which
            # is what exactly we need here.
            which_eigenvalues = 'BE'

            eigenvalues[:num_none_zero_eig] = \
                scipy.sparse.linalg.eigsh(A, num_none_zero_eig,
                                          which=which_eigenvalues,
                                          return_eigenvectors=False,
                                          tol=tol)
        else:
            # Computing half of eigenvalues from the largest magnitude
            eigenvalues_large = scipy.sparse.linalg.eigs(
                    A, num_none_zero_eig//2, which='LM',
                    return_eigenvectors=False, tol=tol)

            # Computing half of eigenvalues from the smallest magnitude
            eigenvalues_small = scipy.sparse.linalg.eigs(
                    A, num_none_zero_eig//2, which='SM',
                    return_eigenvectors=False, tol=tol)

            # Combine large and small eigenvalues into one array
            num_eigenvalues = eigenvalues_large.size + \
                eigenvalues_small.size
            eigenvalues[:num_eigenvalues] = numpy.r_[eigenvalues_large,
                                                     eigenvalues_small]
    else:

        # Dense matrix
        if assume_matrix == 'sym':
            eigenvalues = scipy.linalg.eigh(A, check_finite=False,
                                            eigvals_only=True)
        else:
            eigenvalues = scipy.linalg.eig(A, check_finite=False)[0]

    return eigenvalues
