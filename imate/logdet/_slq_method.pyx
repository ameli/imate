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

from .._trace_estimator import trace_estimator
from .._trace_estimator cimport trace_estimator
from ..functions import pyFunction
from ..functions cimport pyFunction, Function, Logarithm


# ==========
# slq method
# ==========

def slq_method(
        A,
        parameters=None,
        gram=False,
        exponent=1.0,
        min_num_samples=10,
        max_num_samples=50,
        error_atol=None,
        error_rtol=1e-2,
        confidence_level=0.95,
        outlier_significance_level=0.001,
        lanczos_degree=20,
        lanczos_tol=None,
        orthogonalize=0,
        num_threads=0,
        num_gpu_devices=0,
        verbose=False,
        plot=False,
        gpu=False):
    """
    Computes the trace of inverse of matrix based on stochastic Lanczos
    quadrature (SLQ) method.

    :param A: invertible matrix or linear operator
    :type A: numpy.ndarray, scipy.sparse matrix, or LinearOperator object

    :param max_num_samples: Number of Monte-Carlo trials
    :type max_num_samples: unsigned int

    :param lanczos_degree: Lanczos degree
    :type lanczos_degree: unsigned int

    :param gram: If ``True``, the Gram matrix ``A.T @ A`` is considered instead
        of ``A`` itself. In this case, ``A`` itself can be a generic, but
        square, matrix. If ``False``, matrix ``A`` is used, which then it has
        to be symmetric and positive semi-definite.
    :type gram: bool

    :param lanczos_tol: The tolerance used for inner-computation of Lanczos
        stochastic quadrature method. If the tolerance is ``None`` (default
        value), the machine' epsilon precision is used. The machine's epsilon
        precision for 32-bit precision data is 2**(-23) = 1.1920929e-07, for
        64-bit precision data is 2**(-52) = 2.220446049250313e-16, and for
        128-bit precision data is 2**(-63) = -1.084202172485504434e-19.
    :type lanczos_tol: float

    :return: Trace of ``A``
    :rtype: float

    .. note::

        In Lanczos tri-diagonalization method, :math:`\\theta`` is the
        eigenvalue of ``T``. However, in Golub-Kahn bi-diagonalization method,
        :math:`\\theta` is the singular values of ``B``. The relation between
        these two methods are as follows: ``B.T*B`` is the ``T`` for ``A.T*A``.
        That is, if we have the input matrix ``A.T*T``, its Lanczos
        tri-diagonalization ``T`` is the same matrix as if we bi-diagonalize
        ``A`` (not ``A.T*A``) with Golub-Kahn to get ``B``, then ``T = B.T*B``.
        This has not been highlighted in the above paper.

        To correctly implement Golub-Kahn, here :math:`\\theta` should be the
        singular values of ``B``, **NOT** the square of the singular values of
        ``B`` (as described in the above paper incorrectly!).

    Technical notes:
        This function is a wrapper to a cython function. In cython we cannot
        have function arguments with default values (neither ``cdef``,
        ``cpdef``). As a work around, this function (defined with ``def``)
        accepts default values for arguments, and then calls the cython
        function (defined with ``cdef``).

    Reference:
        * `Ubaru, S., Chen, J., and Saad, Y. (2017)
          <https://www-users.cs.umn.edu/~saad/PDF/ys-2016-04.pdf>`_,
          Fast Estimation of :math:`\\mathrm{tr}(F(A))` Via Stochastic Lanczos
          Quadrature, SIAM J. Matrix Anal. Appl., 38(4), 1075-1099.
    """

    # Define inverse matrix function
    cdef Function* matrix_function = new Logarithm()
    py_matrix_function = pyFunction()
    py_matrix_function.set_function(matrix_function)

    trace, info = trace_estimator(
        A,
        parameters,
        py_matrix_function,
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
        gpu)

    del matrix_function

    return trace, info
