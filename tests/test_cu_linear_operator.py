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

# This package might not be compiled with the cuda support.
try:
    from imate._cu_linear_operator.tests import test_cu_matrix, \
            test_cu_affine_matrix_function
    subpackage_exists = True
except ModuleNotFoundError:
    subpackage_exists = False


# =======================
# test cu linear operator
# =======================

def test_cu_linear_operator():
    """
    A wrapper for :mod:`imate._linear_operator.tests` test sub-module.
    """

    # A test for linear operator
    if subpackage_exists:
        try:
            test_cu_matrix()
            test_cu_affine_matrix_function()
        except RuntimeError as e:
            print(e)


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_cu_linear_operator())
