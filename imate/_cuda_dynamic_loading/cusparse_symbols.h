/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CUDA_DYNAMIC_LOADING_CUSPARSE_SYMBOLS_H_
#define _CUDA_DYNAMIC_LOADING_CUSPARSE_SYMBOLS_H_


// =======
// Headers
// =======

#include <string>  // std::string
#include "./cusparse_types.h"  // cusparseCreateCsr_type,
                               // cusparseCreateDnVec_type,
                               // cusparseDestroySpMat_type,
                               // cusparseDestroyDnVec_type,
                               // cusparseSpMV_bufferSize_type,
                               // cusparseSpMV_type


// ================
// cusparse Symbols
// ================

/// \class cusparseSymbols
///
/// \brief A static container to store symbols of loaded cusparse library.
/// 
/// \note      When this package is compiled with dynamic loading enabled, make
///            sure that cuda toolkit is available at run-time. For instance
///            on a linux cluster, run:
///
///                module load cuda
///
/// \sa    dynamic_loading
///        cublasSymbols,
///        cudartSymbols

class cusparseSymbols
{
    public:
        // Methods
        static std::string get_lib_name();

        // Data
        static cusparseCreate_type cusparseCreate;
        static cusparseDestroy_type cusparseDestroy;
        static cusparseCreateCsr_type cusparseCreateCsr;
        static cusparseCreateDnVec_type cusparseCreateDnVec;
        static cusparseDestroySpMat_type cusparseDestroySpMat;
        static cusparseDestroyDnVec_type cusparseDestroyDnVec;
        static cusparseSpMV_bufferSize_type cusparseSpMV_bufferSize;
        static cusparseSpMV_type cusparseSpMV;
};

#endif  // _CUDA_DYNAMIC_LOADING_CUSPARSE_SYMBOLS_H_
