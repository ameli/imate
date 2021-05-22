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


# =============================
# Radial Basis Functions Method
# =============================

class RadialBasisFunctionsMethod(InterpolantBase):
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

    :param traceinv_options: A dictionary of input arguments for
        :mod:`imate.traceinv.traceinv` module.
    :type traceinv_options: dict

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

    def __init__(self, A, B=None, interpolant_points=None, traceinv_options={},
                 verbose=False, function_type=1):
        """
        Initializes the base class and attributes.
        """

        # Base class constructor
        super(RadialBasisFunctionsMethod, self).__init__(
                A, B=B, interpolant_points=interpolant_points,
                traceinv_options=traceinv_options, verbose=verbose)

        # Initialize Interpolator
        self.RBF = None
        self.low_log_threshold = None
        self.high_log_threshold = None
        self.function_type = function_type
        self.initialize_interpolator()

    # =======================
    # initialize interpolator
    # =======================

    def initialize_interpolator(self):
        """
        Finds the coefficients of the interpolating function.
        """

        if self.verbose:
            print('Initialize interpolator ...')

        # Take logarithm of t_i
        xi = numpy.log10(self.t_i)

        if xi.size > 1:
            dxi = numpy.mean(numpy.diff(xi))
        else:
            dxi = 1

        # Function Type
        if self.function_type == 1:
            # Ascending function
            yi = 1.0/self.tau_i - (1.0/self.tau0 + self.t_i)
        elif self.function_type == 2:
            # Bell shape, going to zero at boundaries
            yi = (1.0/self.tau_i)/(1.0/self.tau0 + self.t_i) - 1.0
        elif self.function_type == 3:
            # Bell shape, going to zero at boundaries
            yi = 1.0 - (self.tau_i)*(1.0/self.tau0 + self.t_i)
        else:
            raise ValueError('Invalid function type.')

        # extend boundaries to zero
        self.low_log_threshold = -4.5   # SETTING
        self.high_log_threshold = 3.5   # SETTING
        num_extend = 3                  # SETTING

        # Avoid thresholds to cross interval of data
        if self.low_log_threshold >= numpy.min(xi):
            self.low_log_threshold = numpy.min(xi) - dxi
        if self.high_log_threshold <= numpy.max(xi):
            self.high_log_threshold = numpy.max(xi) + dxi

        # Extend interval of data by adding zeros to left and right
        if (self.function_type == 2) or (self.function_type == 3):
            extend_left_x = numpy.linspace(self.low_log_threshold-dxi,
                                           self.low_log_threshold, num_extend)
            extend_right_x = numpy.linspace(self.high_log_threshold,
                                            self.high_log_threshold+dxi,
                                            num_extend)
            extend_y = numpy.zeros(num_extend)
            xi = numpy.r_[extend_left_x, xi, extend_right_x]
            yi = numpy.r_[extend_y, yi, extend_y]

        # Radial Basis Function
        if self.function_type == 1:
            # Best interpolation method is good for ascending shaped function
            self.RBF = scipy.interpolate.CubicSpline(xi, yi, bc_type=((1, 0.0),
                                                     (2, 0)), extrapolate=True)
            # Good
            # self.RBF = scipy.interpolate.PchipInterpolator(xi, yi,
            #                                                extrapolate=True)
            #
            # Bad
            # self.RBF = scipy.interpolate.UnivariateSpline(xi, yi, k=3, s=0.0)
        elif (self.function_type == 2) or (self.function_type == 3):
            # These interpolation methods are good for the Bell shaped function

            # Best for function type 2, 3, 4
            self.RBF = scipy.interpolate.Rbf(xi, yi, function='gaussian',
                                             epsilon=dxi)
            # self.RBF = scipy.interpolate.Rbf(xi, yi, function='inverse',
            #                                  epsilon=dxi)
            # self.RBF = scipy.interpolate.CubicSpline(
            #     xi, yi, bc_type=((1, 0.0), (1, 0.0)), extrapolate=True)

        # Plot interpolation with RBF
        # PlotFlag = False
        # if PlotFlag:
        #     import matplotlib.pyplot as plt
        #     t = numpy.logspace(self.low_log_threshold-dxi,
        #                        self.high_log_threshold+dxi, 100)
        #     x = numpy.log10(t)
        #     y = self.RBF(x)
        #     fig, ax = plt.subplots()
        #     ax.plot(x, y)
        #     ax.plot(xi, yi, 'o')
        #     ax.grid(True)
        #     ax.set_xlim([self.low_log_threshold-dxi,
        #                 self.high_log_threshold+dxi])
        #     # ax.set_ylim(-0.01, 0.18)
        #     plt.show()

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

        x = numpy.log10(t)
        if (x < self.low_log_threshold) or (x > self.high_log_threshold):
            y = 0
        else:
            y = self.RBF(x)

        if self.function_type == 1:
            tau = 1.0/(y + 1.0/self.tau0 + t)
        elif self.function_type == 2:
            tau = 1.0/((y+1.0)*(1.0/self.tau0 + t))
        elif self.function_type == 3:
            tau = (1.0-y)/(1.0/self.tau0 + t)
        else:
            raise ValueError('Invalid function type.')

        trace = self.trace_Binv*tau

        return trace
