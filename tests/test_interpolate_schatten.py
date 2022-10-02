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
from imate import InterpolateSchatten                              # noqa: E402


# =================
# remove saved plot
# =================

def _remove_saved_plot(filename):
    """
    When the option ``plot=True`` is used in :mod:`imate.correlation_matrix`, a
    file named ``CorrelationMatrix.svg`` is saved in the current directory.
    Call this function to delete this file.
    """

    save_dir = os.getcwd()
    save_fullname = os.path.join(save_dir, filename)

    if os.path.exists(save_fullname):
        try:
            os.remove(save_fullname)
        except OSError:
            pass

    print('File %s is deleted.' % save_fullname)


# ==========================
# test interpolate trace exp
# ==========================

def _interpolate_trace_exp(p):
    """
    Runs test for a given exponent p.
    """

    # Compute trace of inverse of K using dense matrix
    print('Using dense matrix')
    A = correlation_matrix(size=20, sparse=False)
    B = correlation_matrix(size=20, sparse=False, scale=0.05)

    verbose = True
    interpolant_points = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1]
    inquiry_point = 0.4

    if p < 0:
        options = {'method': 'cholesky', 'invert_cholesky': True}
    elif p == 0:
        options = {'method': 'cholesky'}
    else:
        options = {'method': 'exact'}

    # Compute exact trace without interpolation
    TI00 = InterpolateSchatten(A, B=B, p=p, kind='EXT', options=options,
                               verbose=verbose)

    Schatten00 = TI00.interpolate(inquiry_point)
    Error00 = 0

    # Eigenvalues (B should be None)
    TI00_no_B = InterpolateSchatten(A, B=None, p=p, kind='EXT',
                                    options=options, verbose=verbose)
    Schatten00_no_B = TI00_no_B.interpolate(inquiry_point)
    TI01 = InterpolateSchatten(A, B=None, p=p, kind='EIG', options=options,
                               verbose=verbose)
    Schatten01 = TI01.interpolate(inquiry_point)
    Error01 = 100.0 * numpy.abs(Schatten01 - Schatten00_no_B) / Schatten00_no_B

    # Monomial Basis Functions
    TI02 = InterpolateSchatten(A, B=B, p=p, kind='MBF', options=options,
                               verbose=verbose)
    Schatten02 = TI02.interpolate(inquiry_point)
    Error02 = 100.0 * numpy.abs(Schatten02 - Schatten00) / Schatten00

    # Root Monomial Basis Functions, basis type: NonOrthogonal
    TI03 = InterpolateSchatten(A, B=B, p=p, ti=interpolant_points, kind='IMBF',
                               basis_func_type='non-ortho', options=options,
                               verbose=verbose)
    Schatten03 = TI03.interpolate(inquiry_point)
    Error03 = 100.0 * numpy.abs(Schatten03 - Schatten00) / Schatten00

    # Root Monomial Basis Functions, basis type: Orthogonal
    TI04 = InterpolateSchatten(A, B=B, p=p, ti=interpolant_points, kind='IMBF',
                               basis_func_type='ortho', options=options,
                               verbose=verbose)
    Schatten04 = TI04.interpolate(inquiry_point)
    Error04 = 100.0 * numpy.abs(Schatten04 - Schatten00) / Schatten00

    # Root Monomial Basis Functions, basis type: Orthogonal2
    TI05 = InterpolateSchatten(A, B=B, p=p, ti=interpolant_points, kind='IMBF',
                               basis_func_type='ortho2', options=options,
                               verbose=verbose)
    Schatten05 = TI05.interpolate(inquiry_point)
    Error05 = 100.0 * numpy.abs(Schatten05 - Schatten00) / Schatten00

    # Radial Basis Functions, func_type 1
    TI06 = InterpolateSchatten(A, B=B, p=p, ti=interpolant_points, kind='RBF',
                               func_type=1, options=options, verbose=verbose)
    Schatten06 = TI06.interpolate(inquiry_point)
    Error06 = 100.0 * numpy.abs(Schatten06 - Schatten00) / Schatten00

    # Radial Basis Functions, func_type 2
    TI07 = InterpolateSchatten(A, B=B, p=p, ti=interpolant_points, kind='RBF',
                               func_type=2, options=options, verbose=verbose)
    Schatten07 = TI07.interpolate(inquiry_point)
    Error07 = 100.0 * numpy.abs(Schatten07 - Schatten00) / Schatten00

    # Rational Polynomial with two interpolating points
    interpolant_points = [1e-1, 1e+1]
    TI08 = InterpolateSchatten(A, B=B, p=p, ti=interpolant_points, kind='RPF',
                               options=options, verbose=verbose)
    Schatten08 = TI08.interpolate(inquiry_point)
    Error08 = 100.0 * numpy.abs(Schatten08 - Schatten00) / Schatten00

    # Rational Polynomial with four interpolating points
    interpolant_points = [1e-2, 1e-1, 1, 1e+1]
    TI09 = InterpolateSchatten(A, B=B, p=p, ti=interpolant_points, kind='RPF',
                               options=options, verbose=verbose)
    Schatten09 = TI09.interpolate(inquiry_point)
    Error09 = 100.0 * numpy.abs(Schatten09 - Schatten00) / Schatten00

    # Chebyshev Rational Polynomial with four interpolating points
    interpolant_points = [1e-2, 1e-1, 1, 1e+1]
    TI10 = InterpolateSchatten(A, B=B, p=p, ti=interpolant_points, kind='CRF',
                               scale=None, func_type=1, options=options,
                               verbose=verbose)
    Schatten10 = TI10.interpolate(inquiry_point)
    Error10 = 100.0 * numpy.abs(Schatten10 - Schatten00) / Schatten00

    # Chebyshev Rational Polynomial with four interpolating points
    interpolant_points = [1e-2, 1e-1, 1, 1e+1]
    TI11 = InterpolateSchatten(A, B=B, p=p, ti=interpolant_points, kind='CRF',
                               scale=0.1, func_type=2, options=options,
                               verbose=verbose)
    Schatten11 = TI11.interpolate(inquiry_point)
    Error11 = 100.0 * numpy.abs(Schatten11 - Schatten00) / Schatten00

    # Spline with four interpolating points
    interpolant_points = [1e-2, 1e-1, 1, 1e+1]
    TI12 = InterpolateSchatten(A, B=B, p=p, ti=interpolant_points, kind='SPL',
                               func_type=1, options=options, verbose=verbose)
    Schatten12 = TI12.interpolate(inquiry_point)
    Error12 = 100.0 * numpy.abs(Schatten12 - Schatten00) / Schatten00

    # Spline with four interpolating points
    interpolant_points = [1e-2, 1e-1, 1, 1e+1]
    TI13 = InterpolateSchatten(A, B=B, p=p, ti=interpolant_points, kind='SPL',
                               func_type=1, options=options, verbose=verbose)
    Schatten13 = TI13.interpolate(inquiry_point)
    Error13 = 100.0 * numpy.abs(Schatten13 - Schatten00) / Schatten00

    print("")
    print("----------------------------------------")
    print("Method  Options         imate     Error")
    print("------  -------------   --------  ------")
    print("EXT     N/A             %8.4f  %5.2f%%" % (Schatten00, Error00))
    print("EIG     N/A             %8.4f  %5.2f%%" % (Schatten01, Error01))
    print("MBF     N/A             %8.4f  %5.2f%%" % (Schatten02, Error02))
    print("IMBF    NonOrthogonal   %8.4f  %5.2f%%" % (Schatten03, Error03))
    print("IMBF    Orthogonal      %8.4f  %5.2f%%" % (Schatten04, Error04))
    print("IMBF    Orthogonal2     %8.4f  %5.2f%%" % (Schatten05, Error05))
    print("RBF     Type 1          %8.4f  %5.2f%%" % (Schatten06, Error06))
    print("RBF     Type 2          %8.4f  %5.2f%%" % (Schatten07, Error07))
    print("RPF     2-Points        %8.4f  %5.2f%%" % (Schatten08, Error08))
    print("RPF     4-Points        %8.4f  %5.2f%%" % (Schatten09, Error09))
    print("CRF     Type 1          %8.4f  %5.2f%%" % (Schatten10, Error10))
    print("CRF     Type 2          %8.4f  %5.2f%%" % (Schatten11, Error11))
    print("SPL     Type 1          %8.4f  %5.2f%%" % (Schatten12, Error12))
    print("SPL     Type 2          %8.4f  %5.2f%%" % (Schatten13, Error13))
    print("----------------------------------------")
    print("")

    # Compare with exact solution and plot results
    inquiry_points = numpy.logspace(numpy.log10(interpolant_points[0]),
                                    numpy.log10(interpolant_points[-1]), 5)

    TI00.plot(inquiry_points, normalize=True, compare=True)
    TI01.plot(inquiry_points, normalize=True, compare=True)
    TI02.plot(inquiry_points, normalize=True, compare=True)
    TI03.plot(inquiry_points, normalize=True, compare=True)
    TI06.plot(inquiry_points, normalize=True, compare=True)
    TI08.plot(inquiry_points, normalize=True, compare=True)
    TI09.plot(inquiry_points, normalize=True, compare=True)
    TI10.plot(inquiry_points, normalize=True, compare=True)
    TI12.plot(inquiry_points, normalize=True, compare=True)

    # Remove saved plot
    _remove_saved_plot('interpolation.pdf')
    _remove_saved_plot('interpolation.svg')


# ======================
# test interpolate trace
# ======================

def test_interpolate_trace():
    """
    Tests imate.InterpolateSchatten class.
    """

    _interpolate_trace_exp(p=-2)
    _interpolate_trace_exp(p=-1)
    _interpolate_trace_exp(p=0)
    _interpolate_trace_exp(p=2)


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_interpolate_trace())
