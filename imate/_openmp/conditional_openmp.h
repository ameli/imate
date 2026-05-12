/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef CONDITIONAL_OPENMP_H_
#define CONDITIONAL_OPENMP_H_

// =======
// Headers
// =======

#if defined(USE_OPENMP) && (USE_OPENMP == 1)
    #include <omp.h>
    #define use_openmp 1
#else

    #define use_openmp 0
    typedef int omp_lock_t;
    void omp_init_lock(omp_lock_t *lock);
    void omp_set_lock(omp_lock_t *lock);
    void omp_unset_lock(omp_lock_t *lock);
    int omp_get_max_threads();
    int omp_get_thread_num();
    void omp_set_num_threads(int num_threads);

#endif


#endif  // CONDITIONAL_OPENMP_H_
