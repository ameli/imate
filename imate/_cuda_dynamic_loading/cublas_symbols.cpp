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

#include "./cublas_symbols.h"
#include <cublas_api.h>  // CUBLAS_VER_MAJOR
#include <cstdlib>  // NULL
#include <sstream>  // std::ostringstream
#include "./dynamic_loading.h"  // dynamic_loading


// =========================
// Initialize static members
// =========================

cublasCreate_type cublasSymbols::cublasCreate = NULL;
cublasDestroy_type cublasSymbols::cublasDestroy = NULL;
cublasSgemv_type cublasSymbols::cublasSgemv = NULL;
cublasDgemv_type cublasSymbols::cublasDgemv = NULL;
cublasScopy_type cublasSymbols::cublasScopy = NULL;
cublasDcopy_type cublasSymbols::cublasDcopy = NULL;
cublasSaxpy_type cublasSymbols::cublasSaxpy = NULL;
cublasDaxpy_type cublasSymbols::cublasDaxpy = NULL;
cublasSdot_type cublasSymbols::cublasSdot = NULL;
cublasDdot_type cublasSymbols::cublasDdot = NULL;
cublasSnrm2_type cublasSymbols::cublasSnrm2 = NULL;
cublasDnrm2_type cublasSymbols::cublasDnrm2 = NULL;
cublasSscal_type cublasSymbols::cublasSscal = NULL;
cublasDscal_type cublasSymbols::cublasDscal = NULL;


// ============
// get lib name
// ============

/// \brief Returns the name of cublas shared library.
///

std::string cublasSymbols::get_lib_name()
{
    // Get the extension name of a shared library depending on OS
    std::string lib_extension;

    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32) || \
        defined(__NT__)
        lib_extension = "lib";
    #elif __APPLE__
        lib_extension = "dylib";
    #elif __linux__
        lib_extension = "so";
    #else
        #error "Unknown compiler"
    #endif

    // Check cublas version
    #ifndef CUBLAS_VER_MAJOR
        #error "CUBLAS_VER_MAJOR is not defined."
    #endif

    // cublas shared library base name
    std::string lib_base_name = "libcublas";

    // Construct the lib name
    std::ostringstream oss;
    // oss << lib_base_name << "." << lib_extension;
    oss << lib_base_name << "." << lib_extension << "." \
        << CUBLAS_VER_MAJOR;

    std::string lib_name = oss.str();
    return lib_name;
}


#ifdef __cplusplus
    extern "C" {
#endif


// =============
// cublas Create
// =============

/// \brief Definition of CUDA's \c cublasCreate function using dynamically
///        loaded cublas library.

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle)
{
    if (cublasSymbols::cublasCreate == NULL)
    {
        std::string lib_name = cublasSymbols::get_lib_name();
        const char* symbol_name = "cublasCreate_v2";

        cublasSymbols::cublasCreate = \
                dynamic_loading::load_symbol<cublasCreate_type>(
                        lib_name.c_str(),
                        symbol_name);
    }

    return cublasSymbols::cublasCreate(handle);
}


// ==============
// cublas Destroy
// ==============


/// \brief Definition of CUDA's \c cublasDestroy function using dynamically
///        loaded cublas library.

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle)
{
    if (cublasSymbols::cublasDestroy == NULL)
    {
        std::string lib_name = cublasSymbols::get_lib_name();
        const char* symbol_name = "cublasDestroy_v2";

        cublasSymbols::cublasDestroy = \
                dynamic_loading::load_symbol<cublasDestroy_type>(
                        lib_name.c_str(),
                        symbol_name);
    }

    return cublasSymbols::cublasDestroy(handle);
}

// ===========
// cublasSgemv
// ===========

/// \brief Definition of CUDA's \c cublasSgemv function using dynamically
///        loaded cublas library.

cublasStatus_t cublasSgemv_v2(
        cublasHandle_t handle,
        cublasOperation_t trans,
        int m,
        int n,
        const float* alpha,
        const float* A,
        int lda,
        const float* x,
        int incx,
        const float* beta,
        float* y,
        int incy)
{
    if (cublasSymbols::cublasSgemv == NULL)
    {
        std::string lib_name = cublasSymbols::get_lib_name();
        const char* symbol_name = "cublasSgemv_v2";

        cublasSymbols::cublasSgemv = \
            dynamic_loading::load_symbol<cublasSgemv_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cublasSymbols::cublasSgemv(handle, trans, m, n, alpha, A, lda, x,
                                      incx, beta, y, incy);
}


// ===========
// cublasDgemv
// ===========

/// \brief Definition of CUDA's \c cublasDgemv function using dynamically
///        loaded cublas library.

cublasStatus_t cublasDgemv_v2(
        cublasHandle_t handle,
        cublasOperation_t trans,
        int m,
        int n,
        const double* alpha,
        const double* A,
        int lda,
        const double* x,
        int incx,
        const double* beta,
        double* y,
        int incy)
{
    if (cublasSymbols::cublasDgemv == NULL)
    {
        std::string lib_name = cublasSymbols::get_lib_name();
        const char* symbol_name = "cublasDgemv_v2";

        cublasSymbols::cublasDgemv = \
            dynamic_loading::load_symbol<cublasDgemv_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cublasSymbols::cublasDgemv(handle, trans, m, n, alpha, A, lda, x,
                                      incx, beta, y, incy);
}


// ===========
// cublasScopy
// ===========

/// \brief Definition of CUDA's \c cublasScopy function using dynamically
///        loaded cublas library.

cublasStatus_t cublasScopy(
        cublasHandle_t handle, int n,
        const float* x,
        int incx,
        float* y,
        int incy)
{
    if (cublasSymbols::cublasScopy == NULL)
    {
        std::string lib_name = cublasSymbols::get_lib_name();
        const char* symbol_name = "cublasScopy_v2";

        cublasSymbols::cublasScopy = \
            dynamic_loading::load_symbol<cublasScopy_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cublasSymbols::cublasScopy(handle, n, x, incx, y, incy);
}


// ===========
// cublasDcopy
// ===========

/// \brief Definition of CUDA's \c cublasDcopy function using dynamically
///        loaded cublas library.

cublasStatus_t cublasDcopy(
        cublasHandle_t handle, int n,
        const double* x,
        int incx,
        double* y,
        int incy)
{
    if (cublasSymbols::cublasDcopy == NULL)
    {
        std::string lib_name = cublasSymbols::get_lib_name();
        const char* symbol_name = "cublasDcopy_v2";

        cublasSymbols::cublasDcopy = \
            dynamic_loading::load_symbol<cublasDcopy_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cublasSymbols::cublasDcopy(handle, n, x, incx, y, incy);
}


// ===========
// cublasSaxpy
// ===========

/// \brief Definition of CUDA's \c cublasSaxpy function using dynamically
///        loaded cublas library.

cublasStatus_t cublasSaxpy(
        cublasHandle_t handle,
        int n,
        const float *alpha,
        const float *x,
        int incx,
        float *y,
        int incy)
{
    if (cublasSymbols::cublasSaxpy == NULL)
    {
        std::string lib_name = cublasSymbols::get_lib_name();
        const char* symbol_name = "cublasSaxpy_v2";

        cublasSymbols::cublasSaxpy = \
            dynamic_loading::load_symbol<cublasSaxpy_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cublasSymbols::cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}


// ===========
// cublasDaxpy
// ===========

/// \brief Definition of CUDA's \c cublasDaxpy function using dynamically
///        loaded cublas library.

cublasStatus_t cublasDaxpy(
        cublasHandle_t handle,
        int n,
        const double *alpha,
        const double *x,
        int incx,
        double *y,
        int incy)
{
    if (cublasSymbols::cublasDaxpy == NULL)
    {
        std::string lib_name = cublasSymbols::get_lib_name();
        const char* symbol_name = "cublasDaxpy_v2";

        cublasSymbols::cublasDaxpy = \
            dynamic_loading::load_symbol<cublasDaxpy_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cublasSymbols::cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}


// ==========
// cublasSdot
// ==========

/// \brief Definition of CUDA's \c cublasSdot function using dynamically
///        loaded cublas library.

cublasStatus_t cublasSdot(
        cublasHandle_t handle,
        int n,
        const float *x,
        int incx,
        const float *y,
        int incy,
        float *result)
{
    if (cublasSymbols::cublasSdot == NULL)
    {
        std::string lib_name = cublasSymbols::get_lib_name();
        const char* symbol_name = "cublasSdot_v2";

        cublasSymbols::cublasSdot = \
            dynamic_loading::load_symbol<cublasSdot_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cublasSymbols::cublasSdot(handle, n, x, incx, y, incy, result);
}


// ==========
// cublasDdot
// ==========

/// \brief Definition of CUDA's \c cublasDdot function using dynamically
///        loaded cublas library.

cublasStatus_t cublasDdot(
        cublasHandle_t handle,
        int n,
        const double *x,
        int incx,
        const double *y,
        int incy,
        double *result)
{
    if (cublasSymbols::cublasDdot == NULL)
    {
        std::string lib_name = cublasSymbols::get_lib_name();
        const char* symbol_name = "cublasDdot_v2";

        cublasSymbols::cublasDdot = \
            dynamic_loading::load_symbol<cublasDdot_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cublasSymbols::cublasDdot(handle, n, x, incx, y, incy, result);
}


// ===========
// cublasSnrm2
// ===========

/// \brief Definition of CUDA's \c cublasSnrm2 function using dynamically
///        loaded cublas library.

cublasStatus_t cublasSnrm2(
        cublasHandle_t handle,
        int n,
        const float *x,
        int incx,
        float *result)
{
    if (cublasSymbols::cublasSnrm2 == NULL)
    {
        std::string lib_name = cublasSymbols::get_lib_name();
        const char* symbol_name = "cublasSnrm2_v2";

        cublasSymbols::cublasSnrm2 = \
            dynamic_loading::load_symbol<cublasSnrm2_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cublasSymbols::cublasSnrm2(handle, n, x, incx, result);
}


// ===========
// cublasDnrm2
// ===========

/// \brief Definition of CUDA's \c cublasDnrm2 function using dynamically
///        loaded cublas library.

cublasStatus_t cublasDnrm2(
        cublasHandle_t handle,
        int n,
        const double *x,
        int incx,
        double *result)
{
    if (cublasSymbols::cublasDnrm2 == NULL)
    {
        std::string lib_name = cublasSymbols::get_lib_name();
        const char* symbol_name = "cublasDnrm2_v2";

        cublasSymbols::cublasDnrm2 = \
            dynamic_loading::load_symbol<cublasDnrm2_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cublasSymbols::cublasDnrm2(handle, n, x, incx, result);
}


// ===========
// cublasSscal
// ===========

/// \brief Definition of CUDA's \c cublasSscal function using dynamically
///        loaded cublas library.

cublasStatus_t cublasSscal(
        cublasHandle_t handle,
        int n,
        const float *alpha,
        float *x,
        int incx)
{
    if (cublasSymbols::cublasSscal == NULL)
    {
        std::string lib_name = cublasSymbols::get_lib_name();
        const char* symbol_name = "cublasSscal_v2";

        cublasSymbols::cublasSscal = \
            dynamic_loading::load_symbol<cublasSscal_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cublasSymbols::cublasSscal(handle, n, alpha, x, incx);
}


// ===========
// cublasDscal
// ===========

/// \brief Definition of CUDA's \c cublasDscal function using dynamically
///        loaded cublas library.

cublasStatus_t cublasDscal(
        cublasHandle_t handle,
        int n,
        const double *alpha,
        double *x,
        int incx)
{
    if (cublasSymbols::cublasDscal == NULL)
    {
        std::string lib_name = cublasSymbols::get_lib_name();
        const char* symbol_name = "cublasDscal_v2";

        cublasSymbols::cublasDscal = \
            dynamic_loading::load_symbol<cublasDscal_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cublasSymbols::cublasDscal(handle, n, alpha, x, incx);
}

#ifdef __cplusplus
    }
#endif
