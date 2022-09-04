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
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, p=0, options={}, verbose=False):
        """
        Initializes the parent class.
        """

        # Base class constructor
        super(ExactMethod, self).__init__(
                A, B=B, p=p, ti=None, options=options, verbose=verbose)

        # Attributes
        self.q = 0

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
        return self.eval(t)
