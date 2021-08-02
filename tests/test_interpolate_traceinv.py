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
import numpy

# For plotting matrix, we disable interactive display
os.environ['IMATE_NO_DISPLAY'] = 'True'  # define before importing imate
from imate.sample_matrices import correlation_matrix               # noqa: E402
from imate import InterpolateTraceinv                              # noqa: E402


# =================
# remove saved plot
# =================

def remove_saved_plot():
    """
    When the option ``plot=True`` is used in :mod:`imate.correlation_matrix`, a
    file named ``CorrelationMatrix.svg`` is saved in the current directory.
    Call this function to delete this file.
    """

    save_dir = os.getcwd()
    filename_svg = 'interpolation_results' + '.svg'
    save_fullname_svg = os.path.join(save_dir, filename_svg)

    if os.path.exists(save_fullname_svg):
        try:
            os.remove(save_fullname_svg)
        except OSError:
            pass

    print('File %s is deleted.' % save_fullname_svg)


# =========================
# test interpolate traceinv
# =========================

def test_interpolate_traceinv():
    """
    Test for :mod:`imate.interpolateTraceinv` sub-package.
    """

    # Compute trace of inverse of K using dense matrix
    print('Using dense matrix')
    A = correlation_matrix(size=20, sparse=False)
    B = correlation_matrix(size=20, sparse=False, distance_scale=0.05)

    verbose = True
    interpolant_points = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1]
    inquiry_point = 0.4
    traceinv_options = {'method': 'cholesky', 'invert_cholesky': True}

    # Compute exact trace without interpolation
    TI00 = InterpolateTraceinv(A, B=B, method='EXT',
                               traceinv_options=traceinv_options,
                               verbose=verbose)

    Trace00 = TI00.interpolate(inquiry_point)
    Error00 = 0

    # Eigenvalues Method
    TI01 = InterpolateTraceinv(A, B=B, method='EIG',
                               traceinv_options=traceinv_options,
                               verbose=verbose)
    Trace01 = TI01.interpolate(inquiry_point)
    Error01 = 100.0 * numpy.abs(Trace01 - Trace00) / Trace00

    # Monomial Basis Functions
    TI02 = InterpolateTraceinv(A, B=B, method='MBF',
                               traceinv_options=traceinv_options,
                               verbose=verbose)
    Trace02 = TI02.interpolate(inquiry_point)
    Error02 = 100.0 * numpy.abs(Trace02 - Trace00) / Trace00

    # Root Monomial Basis Functions, basis type: NonOrthogonal
    TI03 = InterpolateTraceinv(A, B=B, interpolant_points=interpolant_points,
                               method='RMBF',
                               basis_functions_type='NonOrthogonal',
                               traceinv_options=traceinv_options,
                               verbose=verbose)
    Trace03 = TI03.interpolate(inquiry_point)
    Error03 = 100.0 * numpy.abs(Trace03 - Trace00) / Trace00

    # Root Monomial Basis Functions, basis type: Orthogonal
    TI04 = InterpolateTraceinv(A, B=B, interpolant_points=interpolant_points,
                               method='RMBF',
                               basis_functions_type='Orthogonal',
                               traceinv_options=traceinv_options,
                               verbose=verbose)
    Trace04 = TI04.interpolate(inquiry_point)
    Error04 = 100.0 * numpy.abs(Trace04 - Trace00) / Trace00

    # Root Monomial Basis Functions, basis type: Orthogonal2
    TI05 = InterpolateTraceinv(A, B=B, interpolant_points=interpolant_points,
                               method='RMBF',
                               basis_functions_type='Orthogonal2',
                               traceinv_options=traceinv_options,
                               verbose=verbose)
    Trace05 = TI05.interpolate(inquiry_point)
    Error05 = 100.0 * numpy.abs(Trace05 - Trace00) / Trace00

    # Radial Basis Functions, function_type 1
    TI06 = InterpolateTraceinv(A, B=B, interpolant_points=interpolant_points,
                               method='RBF', function_type=1,
                               traceinv_options=traceinv_options,
                               verbose=verbose)
    Trace06 = TI06.interpolate(inquiry_point)
    Error06 = 100.0 * numpy.abs(Trace06 - Trace00) / Trace00

    # Radial Basis Functions, function_type 2
    TI07 = InterpolateTraceinv(A, B=B, interpolant_points=interpolant_points,
                               method='RBF', function_type=2,
                               traceinv_options=traceinv_options,
                               verbose=verbose)
    Trace07 = TI07.interpolate(inquiry_point)
    Error07 = 100.0 * numpy.abs(Trace07 - Trace00) / Trace00

    # Radial Basis Functions, function_type 3
    TI08 = InterpolateTraceinv(A, B=B, interpolant_points=interpolant_points,
                               method='RBF', function_type=3,
                               traceinv_options=traceinv_options,
                               verbose=verbose)
    Trace08 = TI08.interpolate(inquiry_point)
    Error08 = 100.0 * numpy.abs(Trace08 - Trace00) / Trace00

    # Rational Polynomial with two interpolating points
    interpolant_points = [1e-1, 1e+1]
    TI09 = InterpolateTraceinv(A, B=B, interpolant_points=interpolant_points,
                               method='RPF', traceinv_options=traceinv_options,
                               verbose=verbose)
    Trace09 = TI09.interpolate(inquiry_point)
    Error09 = 100.0 * numpy.abs(Trace09 - Trace00) / Trace00

    # Rational Polynomial with four interpolating points
    interpolant_points = [1e-2, 1e-1, 1, 1e+1]
    TI10 = InterpolateTraceinv(A, B=B, interpolant_points=interpolant_points,
                               method='RPF', traceinv_options=traceinv_options,
                               verbose=verbose)
    Trace10 = TI10.interpolate(inquiry_point)
    Error10 = 100.0 * numpy.abs(Trace10 - Trace00) / Trace00

    print("")
    print("----------------------------------------")
    print("Method  Options         imate     Error")
    print("------  -------------   --------  ------")
    print("EXT     N/A             %8.4f  %5.2f%%" % (Trace00, Error00))
    print("EIG     N/A             %8.4f  %5.2f%%" % (Trace01, Error01))
    print("MBF     N/A             %8.4f  %5.2f%%" % (Trace02, Error02))
    print("RMBF    NonOrthogonal   %8.4f  %5.2f%%" % (Trace03, Error03))
    print("RMBF    Orthogonal      %8.4f  %5.2f%%" % (Trace04, Error04))
    print("RMBF    Orthogonal2     %8.4f  %5.2f%%" % (Trace05, Error05))
    print("RBF     Type 1          %8.4f  %5.2f%%" % (Trace06, Error06))
    print("RBF     Type 2          %8.4f  %5.2f%%" % (Trace07, Error07))
    print("RBF     Type 3          %8.4f  %5.2f%%" % (Trace08, Error08))
    print("RPF     2-Points        %8.4f  %5.2f%%" % (Trace09, Error09))
    print("RPF     4-Points        %8.4f  %5.2f%%" % (Trace10, Error10))
    print("----------------------------------------")
    print("")

    # Compare with exact soluton and plot results
    inquiry_points = numpy.logspace(numpy.log10(interpolant_points[0]),
                                    numpy.log10(interpolant_points[-1]), 5)

    trace_interpolated, trace_exact, trace_relative_error = \
        TI00.interpolate(inquiry_points, compare_with_exact=True,
                         plot=True)

    trace_interpolated, trace_exact, trace_relative_error = \
        TI01.interpolate(inquiry_points, compare_with_exact=True,
                         plot=True)

    trace_interpolated, trace_exact, trace_relative_error = \
        TI02.interpolate(inquiry_points, compare_with_exact=True,
                         plot=True)

    trace_interpolated, trace_exact, trace_relative_error = \
        TI05.interpolate(inquiry_points, compare_with_exact=True,
                         plot=True)

    trace_interpolated, trace_exact, trace_relative_error = \
        TI08.interpolate(inquiry_points, compare_with_exact=True,
                         plot=True)

    trace_interpolated, trace_exact, trace_relative_error = \
        TI09.interpolate(inquiry_points, compare_with_exact=True,
                         plot=True)

    # Remove saved plot
    remove_saved_plot()


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_interpolate_traceinv())
