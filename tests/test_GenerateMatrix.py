#! /usr/bin/env python

# =======
# Imports
# =======

# For plotting matrix, we disable interactive display
import os
os.environ['TRACEINV_DISPLAY'] = ''  # This should be before importing TraceInv packages

import sys
from TraceInv import GenerateMatrix

# =================
# Remove Saved Plot
# =================

def RemoveSavedPlot():
    """
    When the option ``Plot=True`` is used in :mod:`TraceInv.GenerateMatrix`, a file named
    ``CorrelationMatrix.svg`` is saved in the current directory. Call this function
    to delete this file.
    """

    from os import path
    SaveDir = os.getcwd()
    Filename_SVG = 'CorrelationMatrix' + '.svg'
    SaveFullname_SVG = os.path.join(SaveDir,Filename_SVG)

    if os.path.exists(SaveFullname_SVG):
        try:
            os.remove(SaveFullname_SVG)
        except:
            pass

    print('File %s is deleted.'%SaveFullname_SVG)

# ====================
# Test Generate Matrix
# ====================

def test_GenerateMatrix():
    """
    Test for :mod:`TraceInv.GenerateMatrix` sub-package.
    """

    # Generate a dense matrix using points on a grid
    K1 = GenerateMatrix(NumPoints=20,UseSparse=False,Plot=True)

    # Generate a dense matric using random set of points
    K2 = GenerateMatrix(NumPoints=20,GridOfPoints=False,UseSparse=False,Plot=True)

    # Generate sparse matrix and run in parallel
    K3 = GenerateMatrix(NumPoints=20,UseSparse=True,RunInParallel=True,Plot=True)

    # Remove saved plot
    RemoveSavedPlot()

# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_GenerateMatrix())
