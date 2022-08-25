# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

from __future__ import print_function
from ._interpolant_base import InterpolantBase

import numpy
import scipy
import scipy.interpolate
import scipy.optimize


# =============
# Spline Method
# =============

class SplineMethod(InterpolantBase):
    """
    Computes the trace of inverse of an invertible matrix :math:`\\mathbf{A} +
    t \\mathbf{B}` using an interpolation scheme based on rational polynomial
    functions (see details below).

    **Class Inheritance:**

    .. inheritance-diagram::
        imate.InterpolateTraceinv.RadialBasisFunctionsMethod
        :parts: 1

    :param A: Invertible matrix, can be either dense or sparse matrix.
    :type A: numpy.ndarray

    :param B: Invertible matrix, can be either dense or sparse matrix.
    :type B: numpy.ndarray

    :param options: A dictionary of input arguments for
        :mod:`imate.traceinv.traceinv` module.
    :type options: dict

    :param verbose: If ``True``, prints some information on the computation
        process. Default is ``False``.
    :type verbose: bool

    :param function_type: Can be ``1``, ``2``, or ``3``, which defines
        different radial basis functions (see details below).
    :type function_type: int

    **Interpolation Method**

    Define the function

    .. math::

        \\tau(t) = \\frac{\\mathrm{trace}\\left( (\\mathbf{A} +
        t \\mathbf{B})^{-1} \\right)}{\\mathrm{trace}(\\mathbf{B}^{-1})}

    and :math:`\\tau_0 = \\tau(0)`. Then, we approximate :math:`\\tau(t)` by
    radial basis functions as follows. Define

    .. math::

        x(t) = \\log t

    Depending whether ``function_type`` is set to ``1``, ``2``, or ``3``, one
    of the following functions is defined:

    .. math::
        :nowrap:

        \\begin{eqnarray}
        y_1(t) &= \\frac{1}{\\tau(t)} - \\frac{1}{\\tau_0} - t, \\
        y_2(t) &= \\frac{\\frac{1}{\\tau(t)}}{\\frac{1}{\\tau_0} + t} - 1, \\
        y_3(t) &= 1 - \\tau(t) \\left( \\frac{1}{\\tau_0} + t \\right).
        \\end{eqnarray}

    * The set of data :math:`(x, y_1(x))` are interpolated using
      *cubic splines*.
    * The set of data :math:`(x, y_2(x))` and :math:`(x, y_3(x))` are
      interpolated using *Gaussian radial basis functions*.

    **Example**

    This class can be invoked from
    :class:`imate.InterpolateTraceinv.InterpolateTraceinv` module using
    ``method='RBF'`` argument.

    .. code-block:: python

        >>> from imate import generate_matrix
        >>> from imate import InterpolateTraceinv

        >>> # Create a symmetric positive-definite matrix, size (20**2, 20**2)
        >>> A = generate_matrix(size=20)

        >>> # Create an object that interpolates trace of inverse of A+tI
        >>> # where I is identity matrix.
        >>> TI = InterpolateTraceinv(A, method='RBF')

        >>> # Interpolate A+tI at some input point t
        >>> t = 4e-1
        >>> trace = TI.interpolate(t)
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, p=0, ti=[], func_type=1, options={},
                 verbose=False):
        """
        Initializes the base class and attributes.
        """

        if not isinstance(ti, (list, numpy.ndarray)):
            raise ValueError('"ti" should be a list or array.')

        # Base class constructor. This will compute self.tau0 and self.t_i
        super(SplineMethod, self).__init__(
                A, B=B, p=p, ti=ti, options=options, verbose=verbose)

        # Initialize attributes
        self.func_type = func_type
        self.interp_obj = None

        # Initialize interpolator
        self.initialize_interpolator()

    # =======================
    # initialize interpolator
    # =======================

    def initialize_interpolator(self):
        """
        Finds the coefficients of the interpolating function.
        """

        if self.verbose:
            print('Initialize Interpolator ...')

        if self.func_type == 1:
            yi = self.tau_i / (self.tau0 + self.t_i) - 1.0
        elif self.func_type == 2:
            yi = (self.tau_i - self.tau0) / self.t_i - 1.0
        else:
            raise ValueError('"type" should be 1 or 2.')

        xi = (self.t_i - 1.0) / (self.t_i + 1.0)

        if self.func_type == 1:
            xi = numpy.r_[-1.0, xi, 1.0]
            yi = numpy.r_[0.0, yi, 0.0]
        else:
            xi = numpy.r_[xi, 1.0]
            yi = numpy.r_[yi, 0.0]

        self.interp_obj = scipy.interpolate.CubicSpline(xi, yi)

        if self.verbose:
            print('Done.')

    # ===========
    # interpolate
    # ===========

    def interpolate(self, t):
        """
        Interpolates :math:`\\mathrm{trace} \\left( (\\mathbf{A} +
        t \\mathbf{B})^{-1} \\right)` at :math:`t`.

        This is the main interface function of this module and it is used after
        the interpolation
        object is initialized.

        :param t: The inquiry point(s).
        :type t: float, list, or numpy.array

        :return: The interpolated value of the trace.
        :rtype: float or numpy.array
        """

        x = (t - 1.0) / (t + 1.0)
        y = self.interp_obj(x)

        if self.func_type == 1:
            tau = (y+1.0)*(self.tau0 + t)
        elif self.func_type == 2:
            tau = (y+1.0)*t + self.tau0
        else:
            raise ValueError('"type" should be 1 or 2.')

        schatten = tau * self.schatten_B
        return schatten
