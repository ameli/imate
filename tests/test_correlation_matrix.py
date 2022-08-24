#! /usr/bin/env python

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

import sys
import os

# For plotting matrix, we disable interactive display
os.environ['IMATE_NO_DISPLAY'] = 'True'   # define before importing imate
from imate.sample_matrices import correlation_matrix               # noqa: E402


# =================
# remove saved plot
# =================

def remove_saved_plot(filename):
    """
    When the option ``plot=True`` is used in :mod:`imate.correlationmatrix`, a
    file named ``CorrelationMatrix.svg`` is saved in the current directory.
    Call this function to delete this file.
    """

    save_dir = os.getcwd()
    fullname = os.path.join(save_dir, filename)

    if os.path.exists(fullname):
        try:
            os.remove(fullname)
            print('File %s is deleted.' % fullname)
        except OSError:
            pass

    else:
        print('File %s does not exists.' % fullname)


# =======================
# test correlation matrix
# =======================

def test_correlation_matrix():
    """
    Test for :mod:`imate.sample_matrices.correlation_matrix` sub-package.
    """
    # Generate a dense matrix using points on a grid
    correlation_matrix(size=20, dimension=2, sparse=False, plot=True)

    # Generate a dense matrix using random set of points
    correlation_matrix(size=20, dimension=2, grid=False, sparse=False,
                       plot=True)

    # Generate sparse matrix
    correlation_matrix(size=20, dimension=2, sparse=True, density=1e-2,
                       plot=True)

    # Remove saved plot
    filename_svg = 'correlation_matrix' + '.svg'
    filename_pdf = 'correlation_matrix' + '.pdf'
    remove_saved_plot(filename_svg)
    remove_saved_plot(filename_pdf)


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_correlation_matrix())
