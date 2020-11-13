# =======
# Imports
# =======

import os

# ============
# Time Counter
# ============

class TimeCounterClass(object):
    """
    This class is used to measure the elapsed time of computing trace only for the
    exact (non-interpolation) method.

    In the interpolation method, the trace is "pre-computed" (in TraceEstimationUtilities), so we can easily find
    how much time did it take to compute trace in the pre-computation.

    However, in the direct method of minimizing GCV, we can only measue the total time of minimization process.
    To measure only the elapsed time of computing trace (which is a part of computing GCV) we pass an object
    of this class to accumulatively measure the elapsed time of a portion related to computing trace.
    """

    # ----
    # Init
    # ----
    def __init__(self):
        """
        """

        self.ElapsedTime = 0

    # ---
    # Add
    # ---
    
    def Add(self,Time):
        """
        """

        self.ElapsedTime = self.ElapsedTime + Time

    # -----
    # Reset
    # -----

    def Reset(self):
        """
        """

        self.ElapsedTime = 0

# ========================================
# Restrict Computation To Single Processor
# ========================================

def RestrictComputationToSingleProcessor():
    """
    To measure the CPU time of all processors we use time.process_time() which takes into acount 
    of elapsed time of all running threads. However, it seems when I use scipy.optimize.differential_evolution
    method with either worker=-1 or worker=1, the CPU time is not measured properly.

    After all failed trials, the only solution that measures time (for only scipy.optimize.differential_evolution) 
    is to restrict the whole python script to use a single code. This function does that.

    Note, other scipy.optimzie methods (like shgo) do not have this issue. That means, you can still run the code
    in parallel and the time.process_time() measures the CPU time of all cores properly.
    """

    # Uncomment lines below if measureing elapsed time. These will restrict python to only use one processing thread.
    os.environ["OMP_NUM_THREADS"]        = "1"    # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"]   = "1"    # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"]        = "1"    # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"    # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"]    = "1"    # export NUMEXPR_NUM_THREADS=1
