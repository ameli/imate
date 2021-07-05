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

#include "./cuda_timer.h"


// ===========
// Constructor
// ===========

/// \brief constructor for \c CudaTimer
///

CudaTimer::CudaTimer()
{
    cudaEventCreate(&this->start_time);
    cudaEventCreate(&this->stop_time);
}


// ==========
// Destructor
// ==========

/// \brief Destructor for \c CudaTimer
///

CudaTimer::~CudaTimer()
{
    cudaEventDestroy(this->start_time);
    cudaEventDestroy(this->stop_time);
}


// =====
// start
// =====

/// \brief Starts the timer.
///

void CudaTimer::start()
{
    cudaEventRecord(this->start_time, 0);
}


// ====
// stop
// ====

/// \brief Stops the timer.
///

void CudaTimer::stop()
{
    cudaEventRecord(this->stop_time, 0);
}


// =======
// elapsed
// =======

/// \brief Returns the elapsed time in seconds.
///

float CudaTimer::elapsed() const
{
    float elapsed_time;
    cudaEventSynchronize(this->stop_time);
    cudaEventElapsedTime(&elapsed_time, this->start_time, this->stop_time);

    // Convert from milli-second to second
    return elapsed_time / 1.0e+3;
}
