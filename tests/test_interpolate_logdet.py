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
from imate import InterpolateLogdet                                # noqa: E402


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


# =======================
# test interpolate logdet
# =======================

def test_interpolate_logdet():
    """
    Runs test for a given exponent p.
    """

    # Compute logdet of K using dense matrix
    print('Using dense matrix')
    A = correlation_matrix(size=20, sparse=False)
    B = correlation_matrix(size=20, sparse=False, scale=0.05)

    verbose = True
    interpolant_points = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1]
    inquiry_point = 0.4
    options = {'method': 'cholesky'}

    # Compute exact logdet without interpolation
    TI00 = InterpolateLogdet(A, B=B, kind='EXT', options=options,
                             verbose=verbose)

    Logdet00 = TI00.interpolate(inquiry_point)
    Error00 = 0

    # Eigenvalues (B should be None)
    TI00_no_B = InterpolateLogdet(A, B=None, kind='EXT', options=options,
                                  verbose=verbose)
    Logdet00_no_B = TI00_no_B.interpolate(inquiry_point)
    TI01 = InterpolateLogdet(A, B=None, kind='EIG', options=options,
                             verbose=verbose)
    Logdet01 = TI01.interpolate(inquiry_point)
    Error01 = 100.0 * numpy.abs(Logdet01 - Logdet00_no_B) / Logdet00_no_B

    # Monomial Basis Functions
    TI02 = InterpolateLogdet(A, B=B, kind='MBF', options=options,
                             verbose=verbose)
    Logdet02 = TI02.interpolate(inquiry_point)
    Error02 = 100.0 * numpy.abs(Logdet02 - Logdet00) / Logdet00

    # Root Monomial Basis Functions, basis type: NonOrthogonal
    TI03 = InterpolateLogdet(A, B=B, ti=interpolant_points, kind='IMBF',
                             basis_func_type='non-ortho', options=options,
                             verbose=verbose)
    Logdet03 = TI03.interpolate(inquiry_point)
    Error03 = 100.0 * numpy.abs(Logdet03 - Logdet00) / Logdet00

    # Root Monomial Basis Functions, basis type: Orthogonal
    TI04 = InterpolateLogdet(A, B=B, ti=interpolant_points, kind='IMBF',
                             basis_func_type='ortho', options=options,
                             verbose=verbose)
    Logdet04 = TI04.interpolate(inquiry_point)
    Error04 = 100.0 * numpy.abs(Logdet04 - Logdet00) / Logdet00

    # Root Monomial Basis Functions, basis type: Orthogonal2
    TI05 = InterpolateLogdet(A, B=B, ti=interpolant_points, kind='IMBF',
                             basis_func_type='ortho2', options=options,
                             verbose=verbose)
    Logdet05 = TI05.interpolate(inquiry_point)
    Error05 = 100.0 * numpy.abs(Logdet05 - Logdet00) / Logdet00

    # Radial Basis Functions, func_type 1
    TI06 = InterpolateLogdet(A, B=B, ti=interpolant_points, kind='RBF',
                             func_type=1, options=options, verbose=verbose)
    Logdet06 = TI06.interpolate(inquiry_point)
    Error06 = 100.0 * numpy.abs(Logdet06 - Logdet00) / Logdet00

    # Radial Basis Functions, func_type 2
    TI07 = InterpolateLogdet(A, B=B, ti=interpolant_points, kind='RBF',
                             func_type=2, options=options, verbose=verbose)
    Logdet07 = TI07.interpolate(inquiry_point)
    Error07 = 100.0 * numpy.abs(Logdet07 - Logdet00) / Logdet00

    # Rational Polynomial with two interpolating points
    interpolant_points = [1e-1, 1e+1]
    TI08 = InterpolateLogdet(A, B=B, ti=interpolant_points, kind='RPF',
                             options=options, verbose=verbose)
    Logdet08 = TI08.interpolate(inquiry_point)
    Error08 = 100.0 * numpy.abs(Logdet08 - Logdet00) / Logdet00

    # Rational Polynomial with four interpolating points
    interpolant_points = [1e-2, 1e-1, 1, 1e+1]
    TI09 = InterpolateLogdet(A, B=B, ti=interpolant_points, kind='RPF',
                             options=options, verbose=verbose)
    Logdet09 = TI09.interpolate(inquiry_point)
    Error09 = 100.0 * numpy.abs(Logdet09 - Logdet00) / Logdet00

    # Chebyshev Rational Polynomial with four interpolating points
    interpolant_points = [1e-2, 1e-1, 1, 1e+1]
    TI10 = InterpolateLogdet(A, B=B, ti=interpolant_points, kind='CRF',
                             scale=None, func_type=1, options=options,
                             verbose=verbose)
    Logdet10 = TI10.interpolate(inquiry_point)
    Error10 = 100.0 * numpy.abs(Logdet10 - Logdet00) / Logdet00

    # Chebyshev Rational Polynomial with four interpolating points
    interpolant_points = [1e-2, 1e-1, 1, 1e+1]
    TI11 = InterpolateLogdet(A, B=B, ti=interpolant_points, kind='CRF',
                             scale=0.1, func_type=2, options=options,
                             verbose=verbose)
    Logdet11 = TI11.interpolate(inquiry_point)
    Error11 = 100.0 * numpy.abs(Logdet11 - Logdet00) / Logdet00

    # Spline with four interpolating points
    interpolant_points = [1e-2, 1e-1, 1, 1e+1]
    TI12 = InterpolateLogdet(A, B=B, ti=interpolant_points, kind='SPL',
                             func_type=1, options=options, verbose=verbose)
    Logdet12 = TI12.interpolate(inquiry_point)
    Error12 = 100.0 * numpy.abs(Logdet12 - Logdet00) / Logdet00

    # Spline with four interpolating points
    interpolant_points = [1e-2, 1e-1, 1, 1e+1]
    TI13 = InterpolateLogdet(A, B=B, ti=interpolant_points, kind='SPL',
                             func_type=1, options=options, verbose=verbose)
    Logdet13 = TI13.interpolate(inquiry_point)
    Error13 = 100.0 * numpy.abs(Logdet13 - Logdet00) / Logdet00

    print("")
    print("----------------------------------------")
    print("Method  Options         imate     Error")
    print("------  -------------   --------  ------")
    print("EXT     N/A             %8.4f  %5.2f%%" % (Logdet00, Error00))
    print("EIG     N/A             %8.4f  %5.2f%%" % (Logdet01, Error01))
    print("MBF     N/A             %8.4f  %5.2f%%" % (Logdet02, Error02))
    print("IMBF    NonOrthogonal   %8.4f  %5.2f%%" % (Logdet03, Error03))
    print("IMBF    Orthogonal      %8.4f  %5.2f%%" % (Logdet04, Error04))
    print("IMBF    Orthogonal2     %8.4f  %5.2f%%" % (Logdet05, Error05))
    print("RBF     Type 1          %8.4f  %5.2f%%" % (Logdet06, Error06))
    print("RBF     Type 2          %8.4f  %5.2f%%" % (Logdet07, Error07))
    print("RPF     2-Points        %8.4f  %5.2f%%" % (Logdet08, Error08))
    print("RPF     4-Points        %8.4f  %5.2f%%" % (Logdet09, Error09))
    print("CRF     Type 1          %8.4f  %5.2f%%" % (Logdet10, Error10))
    print("CRF     Type 2          %8.4f  %5.2f%%" % (Logdet11, Error11))
    print("SPL     Type 1          %8.4f  %5.2f%%" % (Logdet12, Error12))
    print("SPL     Type 2          %8.4f  %5.2f%%" % (Logdet13, Error13))
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


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_interpolate_logdet())
