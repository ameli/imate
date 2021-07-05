# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.

from ..functions cimport Function
from .._definitions.types cimport FlagType
from .._c_linear_operator cimport pycLinearOperator
from ..functions cimport pyFunction


# ============
# Declarations
# ============

# pyc trace estimator
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
        alg_wall_times) except *
