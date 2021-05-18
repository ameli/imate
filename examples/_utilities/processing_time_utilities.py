# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import os
import time

# Check python version
import sys
if sys.version_info[0] == 2:
    Python2 = True
else:
    Python2 = False


# ============
# Time Counter
# ============

class TimeCounter(object):
    """
    This class is used to measure the elapsed time of computing trace only for
    the exact (non-interpolation) method.

    In the interpolation methods using :mod:`Trace.InterpolateTraceOfInverse`,
    the trace at interpolant points are *pre-computed*, so we can easily find
    how much time did it take to compute trace in the pre-computation stage.

    However, in the direct method of minimizing GCV, we can only measure the
    total time of minimization process. To measure only the elapsed time of
    computing trace (which is a part of computing GCV) we pass an object of
    this class to accumulatively measure the elapsed time of a portion related
    to computing trace.

    **Example:**

    .. code-block:: python

        >>> # Imports
        >>> from _utilities.processing_time_utilities import \
        >>>     TimeCounter, process_time, \
        >>>     restrict_computation_to_single_processor

        >>> # Declare time counter object
        >>> time_counter = TimeCounter()

        >>> # Define a function
        >>> def function(time_counter):
        ...
        ...     # Count time with time object
        ...     time1 = process_time()
        ...     ...
        ...     time2 = process_time()
        ...
        ...     # Store the time in time_counter object
        ...     time_counter.add(time2 - time1)

        >>> # The main script
        >>> if __name__ == "__main__":
        >>>
        >>>     # Create time counter object
        >>>     time_counter = time_counter()
        >>>
        >>>     # Pass the time_counter in some functions
        >>>     function(time_counter)
        >>>     print(time_counter.elapsed_time)
        >>>
        >>>     # reset the time counter
        >>>     time_counter.reset()
    """

    # ====
    # init
    # ====

    def __init__(self):
        """
        Initialize attribute ``self.elapsed_time`` to zero.
        """

        self.elapsed_time = 0

    # ===
    # add
    # ===

    def add(self, time):
        """
        Adds the input ``Time`` to the member data.

        :param Time: Input time to be added to ``self.elapsed_time``.
        :type Time: float
        """

        self.elapsed_time = self.elapsed_time + time

    # =====
    # reset
    # =====

    def reset(self):
        """
        Resets the member data ``self.elapsed_time`` to zero.
        """

        self.elapsed_time = 0


# ========================================
# restrict computation to single processor
# ========================================

def restrict_computation_to_single_processor():
    """
    To measure the CPU time of all processors we use time.process_time() which
    takes into account of elapsed time of all running threads. However, it
    seems when I use scipy.optimize.differential_evolution method with either
    worker=-1 or worker=1, the CPU time is not measured properly.

    After all failed trials, the only solution that measures time (for only
    scipy.optimize.differential_evolution) is to restrict the whole python
    script to use a single code. This function does that.

    Note, other scipy.optimzie methods (like shgo) do not have this issue. That
    means, you can still run the code in parallel and the time.process_time()
    measures the CPU time of all cores properly.
    """

    # Uncomment lines below if measuring elapsed time. These will restrict
    # python to only use one processing thread.
    os.environ["OMP_NUM_THREADS"] = "1"         # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "1"    # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = "1"         # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_TH...=1
    os.environ["NUMEXPR_NUM_THREADS"] = "1"     # export NUMEXPR_NUM_THREADS=1


# ============
# process time
# ============

def process_time():
    """
    Returns the CPU time.

    For python 2, it returns time.time() and for python 3, it returns
    time.process_time().
    """

    if Python2:
        return time.time()
    else:
        return time.process_time()
