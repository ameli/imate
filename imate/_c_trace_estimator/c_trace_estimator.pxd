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

from .._c_linear_operator cimport cLinearOperator
from .._definitions.types cimport IndexType, FlagType
from ..functions cimport Function


# =======
# Externs
# =======

cdef extern from "c_trace_estimator.h":

    cdef cppclass cTraceEstimator[DataType]:

        @staticmethod
        FlagType c_trace_estimator(
                cLinearOperator[DataType]* A,
                DataType* parameters,
                const IndexType num_inquiries,
                const Function* matrix_function,
                const DataType exponent,
                const FlagType symmetric,
                const FlagType reorthogonalize,
                const IndexType lanczos_degree,
                const DataType lanczos_tol,
                const IndexType min_num_samples,
                const IndexType max_num_samples,
                const DataType error_atol,
                const DataType error_rtol,
                const DataType confidence_level,
                const DataType outlier_significance_level,
                const IndexType num_threads,
                DataType* trace,
                DataType* error,
                DataType** samples,
                IndexType* processed_samples_indices,
                IndexType* num_samples_used,
                IndexType* num_outliers,
                FlagType* converged,
                float& alg_wall_time) nogil
