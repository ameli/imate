/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CUDA_DYNAMIC_LOADING_CUBLAS_SYMBOLS_H_
#define _CUDA_DYNAMIC_LOADING_CUBLAS_SYMBOLS_H_


// =======
// Headers
// =======

#include <string>  // std::string
#include "./cublas_types.h"  // cublasSgemv_type, cublasDgemv_type,
                             // cublasScopy_type, cublasDcopy_type,
                             // cublasSaxpy_type, cublasDaxpy_type,
                             // cublasSdot_type, cublasDdot_type,
                             // cublasSnrm2_type, cublasDnrm2_type,
                             // cublasSscal_type, cublasDscal_type
                             // cublasHandle_t, cublasStatus_t

// ==============
// cublas Symbols
// ==============

/// \class cublasSymbols
///
/// \brief A static container to store symbols of loaded cublas library.
///
/// \note      When this package is compiled with dynamic loading enabled, make
///            sure that cuda toolkit is available at run-time. For instance
///            on a linux cluster, run:
///
///                module load cuda
///
/// \sa    dynamic_loading,
///        cudartSymbols
///        cusparseSymbols

class cublasSymbols
{
    public:
        // Methods
        static std::string get_lib_name();

        // Data
        static cublasCreate_type cublasCreate;
        static cublasDestroy_type cublasDestroy;
        static cublasSgemv_type cublasSgemv;
        static cublasDgemv_type cublasDgemv;
        static cublasScopy_type cublasScopy;
        static cublasDcopy_type cublasDcopy;
        static cublasSaxpy_type cublasSaxpy;
        static cublasDaxpy_type cublasDaxpy;
        static cublasSdot_type cublasSdot;
        static cublasDdot_type cublasDdot;
        static cublasSnrm2_type cublasSnrm2;
        static cublasDnrm2_type cublasDnrm2;
        static cublasSscal_type cublasSscal;
        static cublasDscal_type cublasDscal;
};

#endif  // _CUDA_DYNAMIC_LOADING_CUBLAS_SYMBOLS_H_
