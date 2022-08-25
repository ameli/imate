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
from imate.sample_matrices import toeplitz, toeplitz_logdet, toeplitz_trace, \
        toeplitz_traceinv, toeplitz_schatten


# ================
# test band matrix
# ================

def test_toeplitz():
    """
    Test for :mod:`imate.sample_matrices.toeplitz` sub-package.
    """

    A = toeplitz(2, 1, size=20, gram=False, format='csr',  # noqa: F841
                 dtype='float32')

    B = toeplitz(3, 1, size=20, gram=True, format='csc',   # noqa: F841
                 dtype='float64')

    toeplitz_logdet(2, 1, size=20, gram=False)
    toeplitz_logdet(2, 1, size=20, gram=True)
    toeplitz_trace(2, 1, size=20, gram=False)
    toeplitz_trace(2, 1, size=20, gram=True)
    toeplitz_traceinv(2, 1, size=20, gram=False)
    toeplitz_traceinv(2, 1, size=20, gram=True)
    toeplitz_schatten(2, 1, size=20, p=-2)
    toeplitz_schatten(2, 1, size=20, p=2)


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_toeplitz())
