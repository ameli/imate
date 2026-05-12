# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Externs
# =======

cdef extern from "./conditional_openmp.h":

    ctypedef struct omp_lock_t:
        pass

    extern void omp_init_lock(omp_lock_t *lock) noexcept nogil
    extern void omp_set_lock(omp_lock_t *lock) noexcept nogil
    extern void omp_unset_lock(omp_lock_t *lock) noexcept nogil
    extern int omp_get_max_threads() noexcept nogil
    extern int omp_get_thread_num() noexcept nogil
    extern void omp_set_num_threads(int) noexcept nogil

    cdef int use_openmp
