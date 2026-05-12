/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CU_DEFINITIONS_CU_TYPES_H_
#define _CU_DEFINITIONS_CU_TYPES_H_


// =======
// Headers
// =======

#include "./cu_definitions.h"  // USE_CUDA_FP8_E5M2, USE_CUDA_FP8_E4M3,
                               // USE_CUDA_FP16, USE_CUDA_BF16

#if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    #include <cuda_fp8.h>  // __nv_fp8_e5m2
#else
    // Define a fallback struct when cuda_fp8.h is not included.
    struct __nv_fp8_e5m2 {};
#endif

#if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    #include <cuda_fp8.h>  // __nv_fp8_e4m3
#else
    // Define a fallback struct when cuda_fp8.h is not included.
    struct __nv_fp8_e4m3 {};
#endif

#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    #include <cuda_fp16.h>  // __half
#else
    // No need to define a fallback struct when USE_CUDA_FP16 is not 1. This is
    // because other headers (such as cuda_api.h cusparse.h) already include
    // cuda_fp16.h, whether we include cuda_fp16.h or not. Hence, defining a
    // fallback struct indeed conflicts with the already existing definition.
#endif

#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    #include <cuda_bf16.h>  // __nv_bfloat16
    // No need to define a fallback struct when USE_CUDA_BF16 is not 1. This is
    // because other headers (such as cublas_api.h) already include
    // cuda_bf16.h, whether we include cuda_bf16.h or not. Hence, defining a
    // fallback struct indeed conflicts with the already existing definition.
#endif


#endif  // _CU_DEFINITIONS_CU_TYPES_H_
