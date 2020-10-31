# Depending on python 2 or 3, import relative to the directory or the package
import sys
if sys.version_info[0] == 2:
    # For python 2
    from .ComputeTraceOfInverse import ComputeTraceOfInverse
    from .InterpolateTraceOfInverse import InterpolateTraceOfInverse
    from .GenerateMatrix import GenerateMatrix
else:
    # For python 3
    # from TraceInv.ComputeTraceOfInverse import ComputeTraceOfInverse
    # from TraceInv.InterpolateTraceOfInverse import InterpolateTraceOfInverse
    # from TraceInv.GenerateMatrix import GenerateMatrix
    from .ComputeTraceOfInverse import ComputeTraceOfInverse
    from .InterpolateTraceOfInverse import InterpolateTraceOfInverse
    from .GenerateMatrix import GenerateMatrix

from.__version__ import __version__
