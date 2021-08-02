# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

from cython import boundscheck, wraparound
from scipy.special.cython_special cimport gamma, kv
from libc.math cimport sqrt, exp, isnan, isinf, INFINITY
from libc.stdio cimport printf
from libc.string cimport strcmp
from libc.stdlib cimport abort
from .._definitions.types cimport kernel_type

__all__ = ['kernel', 'euclidean-distance']


# =====
# Types
# =====

# In cython, a function pointer can be created by declaring a function type
# inside a struct.
ctypedef struct Kernels:
    kernel_type kernel_function


# ==========
# get kernel
# ==========

cdef kernel_type get_kernel(const char* kernel):
    """
    Returns a function pointer of a given kernel name.
    """

    cdef kernel_type kernel_function
    cdef Kernels kernels

    if strcmp(kernel, b'matern') == 0:
        kernels.kernel_function = _matern_kernel

    elif strcmp(kernel, b'exponential') == 0:
        kernels.kernel_function = _exponential_kernel

    elif strcmp(kernel, b'square-exponential') == 0:
        kernels.kernel_function = _square_exponential_kernel

    elif strcmp(kernel, b'rational-quadratic') == 0:
        kernels.kernel_function = _rational_quadratic_kernel

    else:
        printf('ERROR: invalid kernel type.\n')
        abort()

    return kernels.kernel_function


# =============
# matern kernel
# =============

cdef double _matern_kernel(
        const double x,
        const double param) nogil:
    """
    Computes the Matern class correlation function for a given Euclidean
    distance of two spatial points.

    The Matern correlation function defined by

    .. math::
        K(\\boldsymbol{x},\\boldsymbol{x}'|\\rho,\\nu) =
            \\frac{2^{1-\\nu}}{\\Gamma(\\nu)}
            \\left( \\sqrt{2 \\nu} \\frac{\\| \\boldsymbol{x} -
            \\boldsymbol{x}' \\|}{\\rho} \\right)
            K_{\\nu}\\left(\\sqrt{2 \\nu}  \\frac{\\|\\boldsymbol{x} -
            \\boldsymbol{x}' \\|}{\\rho} \\right)

    where

        * :math:`\\rho` is the correlation scale of the function,
        * :math:`\\Gamma` is the Gamma function,
        * :math:`\\| \\cdot \\|` is the Euclidean distance,
        * :math:`\\nu` is the smoothness parameter.
        * :math:`K_{\\nu}` is the modified Bessel function of the second kind
          of order :math:`\\nu`

    .. warning::

        When the distance :math:`\\| \\boldsymbol{x} - \\boldsymbol{x}' \\|` is
        zero, the correlation function produces :math:`\\frac{0}{0}` but in the
        limit, the correlation function is :math:`1`. If the distance is not
        exactly zero, but close to zero, this function might produce unstable
        results.

    In this function, it is assumed that :math:`\\nu = \\frac{5}{2}`, and the
    Matern correlation in this case can be represented by:

    .. math::
        K(\\boldsymbol{x},\\boldsymbol{x}'|\\rho,\\nu) =
        \\left( 1 + \\sqrt{5} \\frac{\\| \\boldsymbol{x} -
        \\boldsymbol{x}'\\|}{\\rho} + \\frac{5}{3} \\frac{\\| \\boldsymbol{x} -
        \\boldsymbol{x}'\\|^2}{\\rho^2} \\right)
        \\exp \\left( -\\sqrt{5} \\frac{\\| \\boldsymbol{x} -
        \\boldsymbol{x}'\\|}{\\rho} \\right)

    :param x: The distance that represents the Euclidean distance between
        mutual points.
    :type x: ndarray

    :param nu: The parameter :math:`\\nu` of the Matern kernel.
    :type nu: double

    :return: Matern correlation
    :rtype: double
    """

    # scaled distance
    cdef double nu = param
    cdef double correlation

    if x == 0:
        correlation = 1.0
    else:
        if nu == 0.5:
            correlation = exp(-x)
        elif nu == 1.5:
            correlation = (1.0 + sqrt(3.0) * x) * exp(-sqrt(3.0) * x)
        elif nu == 2.5:
            correlation = (1.0 + sqrt(5.0) * x + (5.0 / 3.0) * (x**2)) * \
                    exp(-sqrt(5.0) * x)
        elif nu < 100:

            # Change zero elements of y to a dummy number, to avoid
            # multiplication of zero by Inf in Bessel function below
            correlation = ((2.0**(1.0-nu)) / gamma(nu)) * \
                    ((sqrt(2.0*nu) * x)**nu) * kv(nu, sqrt(2.0*nu)*x)

        else:
            # For nu > 100, assume nu is Inf. In this case, Matern function
            # approaches Gaussian kernel
            correlation = exp(-0.5*x**2)

        if isnan(correlation):
            printf('correlation is nan.\n')
        if isinf(correlation):
            printf('correlation is inf.\n')

    return correlation


# ===========
# exponential
# ===========

cdef double _exponential_kernel(
        const double x,
        const double param) nogil:
    """
    Exponential kernel is also known as the Gaussian kernel. The exponential
    kernel is a special case of Matern class kernel where
    :math:`\\nu = \\infty`.

    :param x: The distance that represents the Euclidean distance between
        mutual points.
    :type x: ndarray

    :return: Matern correlation
    :rtype: double
    """

    return _matern_kernel(x, 0.5)


# ==================
# square exponential
# ==================

cdef double _square_exponential_kernel(
        const double x,
        const double param) nogil:
    """
    Square exponential kernel is also known as the Gaussian kernel. The square
    exponential kernel is a special case of Matern class kernel where
    :math:`\\nu = \\infty`.
    """

    return _matern_kernel(x, INFINITY)


# ==================
# ratioanl quadratic
# ==================

cdef double _rational_quadratic_kernel(
        const double x,
        const double param) nogil:
    """
    Rational quadratic kernel.
    """

    cdef double alpha = param
    cdef double correlation = 1.0 + x**2 / (2.0 * alpha)

    if alpha != 1.0:
        correlation = correlation**(-alpha)

    return correlation


# ==================
# euclidean_distance
# ==================

@boundscheck(False)
@wraparound(False)
cdef double euclidean_distance(
        const double[:] point1,
        const double[:] point2,
        const double distance_scale,
        const int dimension) nogil:
    """
    Returns the Euclidean distance between two points.

    :param point1: 1D array of the coordinates of a point
    :type point1: cython memoryview (double)

    :param point2: 1D array of the coordinates of a point
    :type point2: cython memoryview (double)

    :param distance_scale: A parameter to scale the distance
    :type distance_scale: double

    :param dimension: Dimension of the coordinates of the points.
    :type dimension: int

    :return: Euclidean distance betwrrn point1 and point2
    :rtype: double
    """

    cdef double distance2 = 0
    cdef int dim

    for dim in range(dimension):
        distance2 += (point1[dim] - point2[dim])**2

    return sqrt(distance2) / distance_scale
