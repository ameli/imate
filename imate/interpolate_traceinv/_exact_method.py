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


# ============
# Exact Method
# ============

class ExactMethod(InterpolantBase):
    """
    Computes the trace of inverse of an invertible matrix :math:`\\mathbf{A}
    + t \\mathbf{B}` using exact method (no interpolation is performed).
    This class does not accept interpolant points as the result is not
    interpolated, rather, used as a benchmark to compare the exact versus the
    interpolated solution of the other classes.

    **Class Inheritance:**

    .. inheritance-diagram:: imate.InterpolateTraceinv.ExactMethod
        :parts: 1

    :param A: A positive-definite matrix. Matrix can be dense or sparse.
    :type A: numpy.ndarray or scipy.sparse.csc_matrix

    :param B: A positive-definite matrix.
        If ``None`` or not provided, it is assumed that ``B`` is an identity
        matrix of the shape of ``A``.
    :type B: numpy.ndarray or scipy.sparse.csc_matrix

    :param traceinv_options: A dictionary of arguments to pass to
        :mod:`imate.traceinv` module.
    :type traceinv_options: dict

    :param verbose: If ``True``, prints some information on the computation
        process. Default is ``False``.
    :type verbose: bool

    :example:

    This class can be invoked from
        :class:`imate.InterpolateTraceinv.InterpolateTraceinv` module
    using ``method='EXT'`` argument.

    .. code-block:: python

        >>> from imate import generate_matrix
        >>> from imate import InterpolateTraceinv

        >>> # Create a symmetric positive-definite matrix, size (20**2, 20**2)
        >>> A = generate_matrix(size=20)

        >>> # Create an object that interpolates trace of inverse of A+tI
        >>> # where I is identity matrix
        >>> TI = InterpolateTraceinv(A, method='EXT')

        >>> # Interpolate A+tI at some input point t
        >>> t = 4e-1
        >>> trace = TI.interpolate(t)

    .. seealso::

        The result of the ``EXT`` method is identical with the eigenvalue
        method ``EIG``, which is given by
        :class:`imate.InterpolateTraceinv.EigenvaluesMethod`.
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, traceinv_options={}, verbose=False):
        """
        Initializes the parent class.
        """

        # Base class constructor
        super(ExactMethod, self).__init__(
                A, B=B, traceinv_options=traceinv_options, verbose=verbose)

        # Attributes
        self.p = 0

    # ===========
    # Interpolate
    # ===========

    def interpolate(self, t):
        """
        This function does not interpolate, rather exactly computes
        :math:`\\mathrm{trace} \\left( (\\mathbf{A} + t \\mathbf{B})^{-1}
        \\right)`

        :param t: The inquiry point(s).
        :type t: float, list, or numpy.array

        :return: The exact value of the trace.
        :rtype: float or numpy.array
        """

        # Do not interpolate, instead compute the exact value
        trace = self.compute(t)

        return trace
