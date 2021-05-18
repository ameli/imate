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
from imate import generate_matrix                                  # noqa: E402


# =================
# remove saved plot
# =================

def remove_saved_plot():
    """
    When the option ``plot=True`` is used in :mod:`imate.generate_matrix`, a
    file named ``CorrelationMatrix.svg`` is saved in the current directory.
    Call this function to delete this file.
    """

    save_dir = os.getcwd()
    filename_svg = 'CorrelationMatrix' + '.svg'
    save_fullname_svg = os.path.join(save_dir, filename_svg)

    if os.path.exists(save_fullname_svg):
        try:
            os.remove(save_fullname_svg)
        except OSError:
            pass

    print('File %s is deleted.' % save_fullname_svg)


# ====================
# test generate matrix
# ====================

def test_generate_matrix():
    """
    Test for :mod:`imate.generate_matrix` sub-package.
    """
    # Generate a dense matrix using points on a grid
    generate_matrix(size=20, dimension=2, sparse=False, plot=True)

    # Generate a dense matrix using random set of points
    generate_matrix(size=20, dimension=2, grid=False, sparse=False, plot=True)

    # Generate sparse matrix
    generate_matrix(size=20, dimension=2, sparse=True, density=1e-2, plot=True)

    # Remove saved plot
    remove_saved_plot()


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_generate_matrix())
