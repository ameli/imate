/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


// =======
// Headers
// =======

#include "./conditional_openmp.h"  // use_openmp


// Nullify all openmp functions
#if !defined(USE_OPENMP) || (USE_OPENMP != 1)

    // =============
    // omp init lock
    // =============

    void omp_init_lock(omp_lock_t *lock)
    {
        (void) lock;
    }

    // ============
    // omp set lock
    // ============
    
    void omp_set_lock(omp_lock_t *lock)
    {
        (void) lock;
    }

    // ==============
    // omp unset lock
    // ==============

    void omp_unset_lock(omp_lock_t *lock)
    {
        (void) lock;
    }

    // ===================
    // omp get max threads
    // ===================

    int omp_get_max_threads()
    {
        return 1;
    }

    // ==================
    // omp get thread num
    // ==================

    int omp_get_thread_num()
    {
        return 0;
    }

    // ===================
    // omp set num threads
    // ===================
    
    void omp_set_num_threads(int num_threads)
    {
        (void) num_threads;
    }

#endif
