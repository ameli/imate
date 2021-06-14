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

from ..functions cimport pyFunction


# ============
# Declarations
# ============

cpdef trace_estimator(
        A,
        parameters,
        pyFunction py_matrix_function,
        exponent,
        symmetric,
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
