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

# Python
import numpy

# Cython
from libc.math cimport NAN
from libc.stdlib cimport malloc, free
from .._c_linear_operator cimport pycLinearOperator
from .._c_linear_operator cimport cLinearOperator
from .c_trace_estimator cimport cTraceEstimator
from .._definitions.types cimport IndexType, FlagType, MemoryViewIndexType, \
        MemoryViewFlagType
from ..functions cimport pyFunction, Function

# Include cython wrapper functions to be compiled in C++ api.
include "lapack_api.pxi"
include "special_functions.pxi"

# To avoid cython's bug that does not recognizes "long double" in template []
ctypedef long double long_double


# ===================
# pyc trace estimator
# ===================

cpdef FlagType pyc_trace_estimator(
        pycLinearOperator Aop,
        parameters,
        num_inquiries,
        pyFunction py_matrix_function,
        exponent,
        symmetric,
        reorthogonalize,
        lanczos_degree,
        lanczos_tol,
        min_num_samples,
        max_num_samples,
        error_atol,
        error_rtol,
        confidence_level,
        outlier_significance_level,
        num_threads,
        data_type_name,
        trace,
        error,
        samples,
        processed_samples_indices,
        num_samples_used,
        num_outliers,
        converged,
        alg_wall_times) except *:
    """
    """

    cdef FlagType all_converged = 0
    cdef float alg_wall_time = 0.0

    if data_type_name == b'float32':

        all_converged = _pyc_trace_estimator_float(
            Aop,
            parameters,
            num_inquiries,
            py_matrix_function,
            exponent,
            int(symmetric),
            reorthogonalize,
            lanczos_degree,
            lanczos_tol,
            min_num_samples,
            max_num_samples,
            error_atol,
            error_rtol,
            confidence_level,
            outlier_significance_level,
            num_threads,
            trace,
            error,
            samples,
            processed_samples_indices,
            num_samples_used,
            num_outliers,
            converged,
            alg_wall_time)

    elif data_type_name == b'float64':

        all_converged = _pyc_trace_estimator_double(
            Aop,
            parameters,
            num_inquiries,
            py_matrix_function,
            exponent,
            int(symmetric),
            reorthogonalize,
            lanczos_degree,
            lanczos_tol,
            min_num_samples,
            max_num_samples,
            error_atol,
            error_rtol,
            confidence_level,
            outlier_significance_level,
            num_threads,
            trace,
            error,
            samples,
            processed_samples_indices,
            num_samples_used,
            num_outliers,
            converged,
            alg_wall_time)

    elif data_type_name == b'float128':

        all_converged = _pyc_trace_estimator_long_double(
            Aop,
            parameters,
            num_inquiries,
            py_matrix_function,
            exponent,
            int(symmetric),
            reorthogonalize,
            lanczos_degree,
            lanczos_tol,
            min_num_samples,
            max_num_samples,
            error_atol,
            error_rtol,
            confidence_level,
            outlier_significance_level,
            num_threads,
            trace,
            error,
            samples,
            processed_samples_indices,
            num_samples_used,
            num_outliers,
            converged,
            alg_wall_time)

    else:
        raise TypeError('Data type should be "float32", "float64", or ' +
                        '"float128".')

    # Return gpu proc time via a numpy array of size 1
    alg_wall_times[0] = alg_wall_time

    return all_converged


# =========================
# pyc trace estimator float
# =========================

cdef FlagType _pyc_trace_estimator_float(
        pycLinearOperator Aop,
        parameters,
        const IndexType num_inquiries,
        pyFunction py_matrix_function,
        const float exponent,
        const FlagType symmetric,
        const FlagType reorthogonalize,
        const IndexType lanczos_degree,
        const float lanczos_tol,
        const IndexType min_num_samples,
        const IndexType max_num_samples,
        const float error_atol,
        const float error_rtol,
        const float confidence_level,
        const float outlier_significance_level,
        const IndexType num_threads,
        float[:] trace,
        float[:] error,
        float[:, ::1] samples,
        MemoryViewIndexType processed_samples_indices,
        MemoryViewIndexType num_samples_used,
        MemoryViewIndexType num_outliers,
        MemoryViewFlagType converged,
        float& alg_wall_time) except *:
    """
    """

    # Get a pointer to parameters
    cdef float* c_parameters
    cdef float scalar_parameters
    cdef float[:] array_parameters

    if parameters is None:
        c_parameters = NULL

    elif numpy.isscalar(parameters):
        scalar_parameters = parameters
        c_parameters = &scalar_parameters

    else:
        array_parameters = parameters
        c_parameters = &array_parameters[0]

    # C pointers from memoryviews
    cdef float* c_trace = &trace[0]
    cdef float* c_error = &error[0]
    cdef IndexType* c_processed_samples_indices = &processed_samples_indices[0]
    cdef IndexType* c_num_samples_used = &num_samples_used[0]
    cdef IndexType* c_num_outliers = &num_outliers[0]
    cdef FlagType* c_converged = &converged[0]

    # For samples array, instead of memoryview, allocate a 2D C array
    cdef float** c_samples = \
        <float**> malloc(sizeof(float*) * max_num_samples)
    cdef IndexType i, j
    for i in range(max_num_samples):
        c_samples[i] = <float*> malloc(sizeof(float) * num_inquiries)

        for j in range(num_inquiries):
            c_samples[i][j] = NAN

    # Get cLinearOperator
    cdef cLinearOperator[float]* Aop_float = Aop.get_linear_operator_float()

    cdef Function* matrix_function = py_matrix_function.get_function()

    # Call templated c++ module
    cdef FlagType all_converged = \
        cTraceEstimator[float].c_trace_estimator(
            Aop_float,
            c_parameters,
            num_inquiries,
            matrix_function,
            exponent,
            int(symmetric),
            reorthogonalize,
            lanczos_degree,
            lanczos_tol,
            min_num_samples,
            max_num_samples,
            error_atol,
            error_rtol,
            confidence_level,
            outlier_significance_level,
            num_threads,
            c_trace,
            c_error,
            c_samples,
            c_processed_samples_indices,
            c_num_samples_used,
            c_num_outliers,
            c_converged,
            alg_wall_time)

    # Write the processed samples to samples to a numpy array. The unprocessed
    # elements of samples array is nan.
    for j in range(num_inquiries):
        for i in range(num_samples_used[j]):
            samples[i, j] = c_samples[processed_samples_indices[i]][j]
        # for i in range(max_num_samples):
        #     samples[i, j] = c_samples[i][j]

    # Deallocate dynamic memory
    for i in range(max_num_samples):
        free(c_samples[i])
    free(c_samples)

    return all_converged


# ==========================
# pyc trace estimator double
# ==========================

cdef FlagType _pyc_trace_estimator_double(
        pycLinearOperator Aop,
        parameters,
        const IndexType num_inquiries,
        pyFunction py_matrix_function,
        const double exponent,
        const FlagType symmetric,
        const FlagType reorthogonalize,
        const IndexType lanczos_degree,
        const double lanczos_tol,
        const IndexType min_num_samples,
        const IndexType max_num_samples,
        const double error_atol,
        const double error_rtol,
        const double confidence_level,
        const double outlier_significance_level,
        const IndexType num_threads,
        double[:] trace,
        double[:] error,
        double[:, ::1] samples,
        MemoryViewIndexType processed_samples_indices,
        MemoryViewIndexType num_samples_used,
        MemoryViewIndexType num_outliers,
        MemoryViewFlagType converged,
        float& alg_wall_time) except *:
    """
    """

    # Get a pointer to parameters
    cdef double* c_parameters
    cdef double scalar_parameters
    cdef double[:] array_parameters

    if parameters is None:
        c_parameters = NULL

    elif numpy.isscalar(parameters):
        scalar_parameters = parameters
        c_parameters = &scalar_parameters

    else:
        array_parameters = parameters
        c_parameters = &array_parameters[0]

    # C pointers from memoryviews
    cdef double* c_trace = &trace[0]
    cdef double* c_error = &error[0]
    cdef IndexType* c_processed_samples_indices = &processed_samples_indices[0]
    cdef IndexType* c_num_samples_used = &num_samples_used[0]
    cdef IndexType* c_num_outliers = &num_outliers[0]
    cdef FlagType* c_converged = &converged[0]

    # For samples array, instead of memoryview, allocate a 2D C array
    cdef double** c_samples = \
        <double**> malloc(sizeof(double) * max_num_samples)
    cdef IndexType i, j
    for i in range(max_num_samples):
        c_samples[i] = <double*> malloc(sizeof(double*) * num_inquiries)

        for j in range(num_inquiries):
            c_samples[i][j] = NAN

    # Get cLinearOperator
    cdef cLinearOperator[double]* Aop_double = Aop.get_linear_operator_double()

    cdef Function* matrix_function = py_matrix_function.get_function()

    # Call templated c++ module
    cdef FlagType all_converged = \
        cTraceEstimator[double].c_trace_estimator(
            Aop_double,
            c_parameters,
            num_inquiries,
            matrix_function,
            exponent,
            int(symmetric),
            reorthogonalize,
            lanczos_degree,
            lanczos_tol,
            min_num_samples,
            max_num_samples,
            error_atol,
            error_rtol,
            confidence_level,
            outlier_significance_level,
            num_threads,
            c_trace,
            c_error,
            c_samples,
            c_processed_samples_indices,
            c_num_samples_used,
            c_num_outliers,
            c_converged,
            alg_wall_time)

    # Write the processed samples to samples to a numpy array. The unprocessed
    # elements of samples array is nan.
    for j in range(num_inquiries):
        for i in range(num_samples_used[j]):
            samples[i, j] = c_samples[processed_samples_indices[i]][j]
        # for i in range(max_num_samples):
        #     samples[i, j] = c_samples[i][j]

    # Deallocate dynamic memory
    for i in range(max_num_samples):
        free(c_samples[i])
    free(c_samples)

    return all_converged


# ===============================
# pyc trace estimator long double
# ===============================

cdef FlagType _pyc_trace_estimator_long_double(
        pycLinearOperator Aop,
        parameters,
        const IndexType num_inquiries,
        pyFunction py_matrix_function,
        const long double exponent,
        const FlagType symmetric,
        const FlagType reorthogonalize,
        const IndexType lanczos_degree,
        const long double lanczos_tol,
        const IndexType min_num_samples,
        const IndexType max_num_samples,
        const long double error_atol,
        const long double error_rtol,
        const long double confidence_level,
        const long double outlier_significance_level,
        const IndexType num_threads,
        long double[:] trace,
        long double[:] error,
        long double[:, ::1] samples,
        MemoryViewIndexType processed_samples_indices,
        MemoryViewIndexType num_samples_used,
        MemoryViewIndexType num_outliers,
        MemoryViewFlagType converged,
        float& alg_wall_time) except *:
    """
    """

    # Get a pointer to parameters
    cdef long double* c_parameters
    cdef long double scalar_parameters
    cdef long double[:] array_parameters

    if parameters is None:
        c_parameters = NULL

    elif numpy.isscalar(parameters):
        scalar_parameters = parameters
        c_parameters = &scalar_parameters

    else:
        array_parameters = parameters
        c_parameters = &array_parameters[0]

    # C pointers from memoryviews
    cdef long double* c_trace = &trace[0]
    cdef long double* c_error = &error[0]
    cdef IndexType* c_processed_samples_indices = &processed_samples_indices[0]
    cdef IndexType* c_num_samples_used = &num_samples_used[0]
    cdef IndexType* c_num_outliers = &num_outliers[0]
    cdef FlagType* c_converged = &converged[0]

    # For samples array, instead of memoryview, allocate a 2D C array
    cdef long double** c_samples = \
        <long double**> malloc(sizeof(long double*) * max_num_samples)
    cdef IndexType i, j
    for i in range(max_num_samples):
        c_samples[i] = \
                <long double*> malloc(sizeof(long double) * num_inquiries)

        for j in range(num_inquiries):
            c_samples[i][j] = NAN

    # Get cLinearOperator
    cdef cLinearOperator[long double]* Aop_long_double = \
        Aop.get_linear_operator_long_double()

    cdef Function* matrix_function = py_matrix_function.get_function()

    # Call templated c++ module
    cdef FlagType all_converged = \
        cTraceEstimator[long_double].c_trace_estimator(
            Aop_long_double,
            c_parameters,
            num_inquiries,
            matrix_function,
            exponent,
            int(symmetric),
            reorthogonalize,
            lanczos_degree,
            lanczos_tol,
            min_num_samples,
            max_num_samples,
            error_atol,
            error_rtol,
            confidence_level,
            outlier_significance_level,
            num_threads,
            c_trace,
            c_error,
            c_samples,
            c_processed_samples_indices,
            c_num_samples_used,
            c_num_outliers,
            c_converged,
            alg_wall_time)

    # Write the processed samples to samples to a numpy array. The unprocessed
    # elements of samples array is nan.
    for j in range(num_inquiries):
        for i in range(num_samples_used[j]):
            samples[i, j] = c_samples[processed_samples_indices[i]][j]
        # for i in range(max_num_samples):
        #     samples[i, j] = c_samples[i][j]

    # Deallocate dynamic memory
    for i in range(max_num_samples):
        free(c_samples[i])
    free(c_samples)

    return all_converged
