#! /usr/bin/env python

# =======
# Imports
# =======

import os
import sys

# ======================================
# Test Plot Generalized Cross Validation
# ======================================

def test_Plot_GeneralizedCrossValidation():
    """
    """
 
    # Get the root directory of the package (parent directory of this script)
    FileDirectory = os.path.dirname(os.path.realpath(__file__))
    ParentDirectory = os.path.dirname(FileDirectory)  
    ExamplesDirectory = os.path.join(ParentDirectory,'examples')

    # Put the examples directory on the path
    sys.path.append(ExamplesDirectory) 

    # Run example
    from examples import Plot_GeneralizedCrossValidation
    Plot_GeneralizedCrossValidation.main(test=True)

# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_Plot_GeneralizedCrossValidation())
