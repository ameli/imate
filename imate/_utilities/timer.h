/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _UTILITIES_TIMER_H_
#define _UTILITIES_TIMER_H_


// =====
// Timer
// =====

/// \class    Timer
///
/// \brief    Records elasped wall time between two events.
///
/// \details  The measured time is the *wall* time, not the *process time* of
///           the CPU.
///
///           **Example:**
///
///               Timer timer;
///               timer.start();
///
///               // Some CPU threads here.
///               // ...
///
///               // Note, this Time measures wall time, so the sleep() time
///               // counts toward the measured time.
///               sleep(1);
///
///               timer.stop();
///               double elapsed_time = timer.elapsed();
///
/// \note     The start and stop time variables *inside* this class must be
///           declared as \double, and not \float to have enough precision for
///           the subtraction of stop minus start time. However, the elapsed
///           time *outside* of this class can be declared as \c float.
///
/// \sa       CudaTimer

class Timer
{
    public:

        Timer();
        ~Timer();
        void start();
        void stop();
        double elapsed() const;

    protected:

        static double get_wall_time();

        // Data
        double start_time;
        double stop_time;
};

#endif  // _UTILITIES_TIMER_H_
