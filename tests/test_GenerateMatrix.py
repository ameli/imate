#! /usr/bin/env python

# =======
# Imports
# =======

import sys
from TraceInv import GenerateMatrix

# ====================
# Test Generate Matrix
# ====================

def test_GenerateMatrix():
    """
    Test for :mod:`TraceInv.GenerateMatrix` sub-package.
    """

    # Generate a dense matrix
    K1 = GenerateMatrix(NumPoints=20,UseSparse=False)

    # Generate sparse matrix and run in parallel
    K2 = GenerateMatrix(NumPoints=20,UseSparse=True,RunInParallel=True)

# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_GenerateMatrix())
