# =======
# Imports
# =======

import os
import time

# Check python version
import sys
if  sys.version_info[0] == 2:
    Python2 = True
else:
    Python2 = False

# ============
# Time Counter
# ============

class TimeCounterClass(object):
    """
    This class is used to measure the elapsed time of computing trace only for the
    exact (non-interpolation) method.

    In the interpolation methods using :mod:`Trace.InterpolateTraceOfInverse`, the trace at interpolant points are
    *pre-computed*, so we can easily find how much time did it take to compute trace in the pre-computation stage.

    However, in the direct method of minimizing GCV, we can only measure the total time of minimization process.
    To measure only the elapsed time of computing trace (which is a part of computing GCV) we pass an object
    of this class to accumulatively measure the elapsed time of a portion related to computing trace.

    **Example:**

    .. code-block:: python

        >>> # Imports
        >>> from Utilities.ProcessingTimeUtilities import RestrictComputationToSingleProcessor
        >>> from Utilities.ProcessingTimeUtilities import TimeCounterClass
        >>> from Utilities.ProcessingTimeUtilities import ProcessTime

        >>> # Define a function
        >>> def function(TimeCounter):
        ...     
        ...     # Count time with time object
        ...     time1 = ProcessTime()
        ...     ...
        ...     time2 = ProcessTime()
        ...
        ...     # Store the time in TimeCounter object
        ...     TimeCounter.Add(time2 - time1)

        >>> # The main script
        >>> if __name__ == "__main__":
        >>>     
        >>>     # Create time counter object
        >>>     TimeCounter = TimeCounterClass()
        >>> 
        >>>     # Pass the TimeCounter in some functions
        >>>     function(TimeCounter)
        >>>     print(TimeCounter.ElapsedTime)
        >>> 
        >>>     # Reset the time counter
        >>>     TimeCounter.Reset()
    """

    # ----
    # Init
    # ----
    def __init__(self):
        """
        Initialize attribute ``self.ElapsedTime`` to zero.
        """

        self.ElapsedTime = 0

    # ---
    # Add
    # ---
    
    def Add(self,Time):
        """
        Adds the input ``Time`` to the member data.

        :param Time: Input time to be added to ``self.ElapsedTime``.
        :type Time: float
        """

        self.ElapsedTime = self.ElapsedTime + Time

    # -----
    # Reset
    # -----

    def Reset(self):
        """
        Resets the member data ``self.ElapsedTime`` to zero.
        """

        self.ElapsedTime = 0

# ========================================
# Restrict Computation To Single Processor
# ========================================

def RestrictComputationToSingleProcessor():
    """
    To measure the CPU time of all processors we use time.process_time() which takes into account 
    of elapsed time of all running threads. However, it seems when I use scipy.optimize.differential_evolution
    method with either worker=-1 or worker=1, the CPU time is not measured properly.

    After all failed trials, the only solution that measures time (for only scipy.optimize.differential_evolution) 
    is to restrict the whole python script to use a single code. This function does that.

    Note, other scipy.optimzie methods (like shgo) do not have this issue. That means, you can still run the code
    in parallel and the time.process_time() measures the CPU time of all cores properly.
    """

    # Uncomment lines below if measuring elapsed time. These will restrict python to only use one processing thread.
    os.environ["OMP_NUM_THREADS"]        = "1"    # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"]   = "1"    # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"]        = "1"    # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"    # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"]    = "1"    # export NUMEXPR_NUM_THREADS=1

# ============
# Process Time
# ============

def ProcessTime():
    """
    Returns the CPU time.
    
    For python 2, it returns time.time() and for python 3, it returns time.process_time().
    """

    if Python2:
        return time.time()
    else:
        return time.process_time()
