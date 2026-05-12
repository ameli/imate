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

from imate._openmp.openmp cimport omp_lock_t, omp_init_lock, omp_set_lock, \
    omp_unset_lock, omp_get_max_threads, omp_get_thread_num, \
    omp_set_num_threads, use_openmp

__all__ = ['omp_lock_t', 'omp_init_lock', 'omp_set_lock', 'omp_unset_lock',
           'omp_get_max_threads', 'omp_get_thread_num', 'omp_set_num_threads',
           'use_openmp']
