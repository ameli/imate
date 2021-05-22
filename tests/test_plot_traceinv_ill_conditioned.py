#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# imports
# =======

import os
import sys


# ==================================
# test plot traceinv ill conditioned
# ==================================

def test_plot_traceinv_ill_conditioned():
    """
    Test for the module :mod:`examples.Plot_imate_IllConditioned`.

    The function :func:`examples.Plot_imate_IllConditioned.main` is called
    with ``test=True`` argument, which evokes computation on smaller matrix
    size. The produced figures are saved with ``test_`` prefix.
    """

    # Get the root directory of the package (parent directory of this script)
    file_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(file_directory)
    examples_directory = os.path.join(parent_directory, 'examples')

    # Put the examples directory on the path
    sys.path.append(examples_directory)

    # Run example
    from examples import plot_traceinv_ill_conditioned
    plot_traceinv_ill_conditioned.main(test=True)


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_plot_traceinv_ill_conditioned())
