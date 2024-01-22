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

import warnings
warnings.resetwarnings()
warnings.filterwarnings("error")


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


# ============================
# test plot traceinv full rank
# ============================

def test_plot_traceinv_full_rank():
    """
    Test for the module :mod:`examples.Plot_imate_FullRank`.

    The function :func:`examples.Plot_imate_FulRank.main` is called
    with ``test=True`` argument, which evokes computation on smaller matrix
    size. The produced figures are saved with ``test_`` prefix.
    """

    # Get the root directory of the package (parent directory of this script)
    file_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(file_directory)
    examples_directory = os.path.join(parent_directory, 'examples')

    # Put the examples directory on the path
    sys.path.append(parent_directory)
    sys.path.append(examples_directory)

    # Run example
    from examples import plot_traceinv_full_rank
    plot_traceinv_full_rank.main(test=True)

    # Remove saved plot
    for p in [0, 1, 2]:
        filename = 'test_traceinv_full_rank_p' + str(int(p))
        filename_svg = filename + '.svg'
        filename_pdf = filename + '.pdf'
        remove_saved_plot(filename_svg)
        remove_saved_plot(filename_pdf)


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_plot_traceinv_full_rank())
