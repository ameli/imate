# SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Externs
# =======

cdef extern from "c_get_config.h":

    cdef bint is_use_cblas() noexcept nogil
    cdef bint is_use_mkl() noexcept nogil
    cdef bint is_use_openmp() noexcept nogil
    cdef bint is_use_loop_unrolling() noexcept nogil
    cdef bint is_use_cuda() noexcept nogil
    cdef bint is_cuda_dynamic_loading() noexcept nogil
    cdef bint is_debug_mode() noexcept nogil
    cdef bint is_cython_build_in_source() noexcept nogil
    cdef bint is_cython_build_for_doc() noexcept nogil
    cdef bint is_use_long_int() noexcept nogil
    cdef bint is_use_unsigned_long_int() noexcept nogil
