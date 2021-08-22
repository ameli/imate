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
from imate.sample_matrices import band_matrix                      # noqa: E402


# ================
# test band matrix
# ================

def test_band_matrix():
    """
    Test for :mod:`imate.sample_matrices.band_matrix` sub-package.
    """

    A = band_matrix(2, 1, size=20, gram=False, format='csr',  # noqa: F841
                    dtype='float32')

    B = band_matrix(3, 1, size=20, gram=True, format='csc',   # noqa: F841
                    dtype='float64')


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_band_matrix())
