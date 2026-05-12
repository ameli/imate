/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CU_DEFINITIONS_CU_DEFINITIONS_H_
#define _CU_DEFINITIONS_CU_DEFINITIONS_H_


// ===========
// Definitions
// ===========

// If USE_CUBLAS is set to 1, the CuBLAS library is used for dense vector and
// matrix operations. By default, this is set to "1". If set to "0", the
// in-house implementation of basic matrix and vector operations will be used.
#ifndef USE_CUBLAS
    #define USE_CUBLAS 1
#endif

// If USE_CUDA_FP8_E5M2 is set to 1, the templated class and functions in
// float8 of type E5M2 in cuda are compiled. Default is 0.
#ifndef USE_CUDA_FP8_E5M2
    #define USE_CUDA_FP8_E5M2 0
#endif

// If USE_CUDA_FP8_E4M3 is set to 1, the templated class and functions in
// float8 of type E4M3 in cuda are compiled. Default is 0.
#ifndef USE_CUDA_FP8_E4M3
    #define USE_CUDA_FP8_E4M3 0
#endif

// If USE_CUDA_FP16 is set to 1, the templated class and functions in float16
// type in cuda are compiled. Default is 0.
#ifndef USE_CUDA_FP16
    #define USE_CUDA_FP16 0
#endif

// If USE_CUDA_BF16 is set to 1, the templated class and functions in bfloat16
// type in cuda are compiled. Default is 0.
#ifndef USE_CUDA_BF16
    #define USE_CUDA_BF16 0
#endif

// If USE_CUDA_FP32 is set to 1, the templated class and functions in float32
// type in cuda are compiled. Default is 1.
#ifndef USE_CUDA_FP32
    #define USE_CUDA_FP32 1
#endif

// If USE_CUDA_FP64 is set to 1, the templated class and functions in float64
// type in cuda are compiled. Default is 1.
#ifndef USE_CUDA_FP64
    #define USE_CUDA_FP64 1
#endif


#endif  // _CU_DEFINITIONS_CU_DEFINITIONS_H_
