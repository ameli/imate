/*
 *  SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
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

#include "./c_get_config.h"


// ============
// is use cblas
// ============

/// \brief Returns \c USE_CBLAS.
///

bool is_use_cblas()
{
    #if defined(USE_CBLAS) && (USE_CBLAS == 1)
        return 1;
    #else
        return 0;
    #endif
}

// ==========
// is use mkl
// ==========

/// \brief Returns \c USE_MKL.
///

bool is_use_mkl()
{
    #if defined(USE_MKL) && (USE_MKL == 1)
        return 1;
    #else
        return 0;
    #endif
}


// =============
// is use openmp
// =============

/// \brief Returns \c USE_OPENMP.
///

bool is_use_openmp()
{
    #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        return 1;
    #else
        return 0;
    #endif
}


// =====================
// is use loop unrolling
// =====================

/// \brief Returns \c USE_LOOP_UNROLLING.
///

bool is_use_loop_unrolling()
{
    #if defined(USE_LOOP_UNROLLING) && (USE_LOOP_UNROLLING == 1)
        return 1;
    #else
        return 0;
    #endif
}


// ===========
// is use cuda
// ===========

/// \brief Returns \c USE_CUDA.
///

bool is_use_cuda()
{
    #if defined(USE_CUDA) && (USE_CUDA == 1)
        return 1;
    #else
        return 0;
    #endif
}


// =======================
// is cuda dynamic loading
// =======================

/// \brief Returns \c CUDA_DYNAMIC_LOADING.
///

bool is_cuda_dynamic_loading()
{
    #if defined(CUDA_DYNAMIC_LOADING) && (CUDA_DYNAMIC_LOADING == 1)
        return 1;
    #else
        return 0;
    #endif
}


// =============
// is debug mode
// =============

/// \brief Returns \c DEBUG_MODE.
///

bool is_debug_mode()
{
    #if defined(DEBUG_MODE) && (DEBUG_MODE == 1)
        return 1;
    #else
        return 0;
    #endif
}


// =========================
// is cython build in source
// =========================

/// \brief Returns \c CYTHON_BUILD_IN_SOURCE.
///

bool is_cython_build_in_source()
{
    #if defined(CYTHON_BUILD_IN_SOURCE) && (CYTHON_BUILD_IN_SOURCE == 1)
        return 1;
    #else
        return 0;
    #endif
}


// =======================
// is cython build for doc
// =======================

/// \brief Returns \c CYTHON_BUILD_FOR_DOC.
///

bool is_cython_build_for_doc()
{
    #if defined(CYTHON_BUILD_FOR_DOC) && (CYTHON_BUILD_FOR_DOC == 1)
        return 1;
    #else
        return 0;
    #endif
}


// ===============
// is use long int
// ===============

/// \brief Returns \c USE_LONG_INT.
///

bool is_use_long_int()
{
    #if defined(USE_LONG_INT) && (USE_LONG_INT == 1)
        return 1;
    #else
        return 0;
    #endif
}


// ========================
// is use unsigned long int
// ========================

/// \brief Returns \c USE_UNSIGNED_LONG_INT.
///

bool is_use_unsigned_long_int()
{
    #if defined(USE_UNSIGNED_LONG_INT) && (USE_UNSIGNED_LONG_INT == 1)
        return 1;
    #else
        return 0;
    #endif
}
