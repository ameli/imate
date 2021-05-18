#! /usr/bin/env python

# =======
# Imports
# =======

import sys
from imate.linear_operator.tests import \
        test_constant_matrix, test_affine_matrix_function


# ====================
# test linear operator
# ====================

def test_linear_operator():
    """
    A wrapper for :mod:`imate._linear_operator.tests` test sub-module.
    """

    # Test linear operator
    test_constant_matrix()
    test_affine_matrix_function()


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_linear_operator())
