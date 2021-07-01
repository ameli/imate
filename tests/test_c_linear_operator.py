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
from imate._c_linear_operator.tests import test_c_matrix, \
        test_c_affine_matrix_function


# ======================
# test c linear operator
# ======================

def test_c_linear_operator():
    """
    A wrapper for :mod:`imate._linear_operator.tests` test sub-module.
    """

    # A test for linear operator
    test_c_matrix()
    test_c_affine_matrix_function()


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_c_linear_operator())
