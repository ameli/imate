/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CUDA_UTILITIES_CUDA_TIMER_H_
#define _CUDA_UTILITIES_CUDA_TIMER_H_


// =======
// Headers
// =======

#include <cuda_runtime_api.h>  // cudaEvent_t, cudaEventCreate,
                               // cudaEventDestropy, cudaEventRecord,
                               // cudaEventSynchronize, cudaEventElapsedTime


// ==========
// Cuda Timer
// ==========

/// \class    CudaTimer
///
/// \brief    Records elasped time between two CUDA events.
///
/// \details  The measured time is the *wall* time, not the *process time* of
///           the GPU. In fact, the measured time the same as the wall clock
///           time of CPU.
///
///           **Example:**
///
///           It is better to *synchronize* all GPU threads before reading the
///           time. For instance:
///
///               CudaTimer cuda_timer;
///               cuda_timer.start();
///
///               // Some GPU threads here.
///               // ...
///
///               // Note, this CudaTime measures wall time, so the sleep()
///               // time counts toward the measured time.
///               sleep(1);
///
///               // Sync threads so the CPU do not jump to the next before
///               // gpu threads are done.
///               cudaDeviceSynchronize();
///
///               cuda_timer.stop();
///               float elapsed_time = cuda_timer.elapsed();
///
/// \sa       Timer

class CudaTimer
{
    public:

        CudaTimer();
        ~CudaTimer();
        void start();
        void stop();
        float elapsed() const;

    protected:
        cudaEvent_t start_time;
        cudaEvent_t stop_time;
};

#endif  // _CUDA_UTILITIES_CUDA_TIMER_H_
