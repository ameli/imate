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

import numpy
import scipy

__all__ = ['kde']


# =============
# normal kernel
# =============

def _normal_kernel(x, data, bw, derivative):
    """
    Probability density function (or its derivative or integral) of normal
    distribution centred at various data point and bandwidths.

    Parameters
    ----------

    data : array_like
        An array representing the arithmetic mean values :math:`\\mu_i`.

    x : float
        A scalar value where the probability density function is evaluated at.

    bw : array_like
        An array of the same size as ``data`` representing the standard
        deviations :math:`\\sigma_i`. The ``bw`` stands for bandwidth of the
        kernel.

    derivative : int {-1, 0, 1}
        The derivative order of the probability density function and it can
        take either of the following values:

        * ``-1``: The first anti-derivative (integral) of the PDF. This is
          useful to compute the cumulative distribution function.
        * ``0``: The probability density function itself.
        * ``1``: The first derivative of the PDF.

    Returns
    -------

    p : array_like
        An array of the size of ``data`` and ``bw`` representing the PDF.

    See Also
    --------

    _log_normal_kernel

    Notes
    -----

    The PDF of normal distribution is:

    .. math::

        p_i(x \\vert \\mu, \\sigma) = \\frac{1}{\\sqrt{2 \\pi} \\sigma}
        \\exp \\left( -\\frac{1}{2} \\left( \\frac{x - \\mu}{\\sigma}
        \\right)^2 \\right).

    This function returns an array with elements:

    .. math::
        \\frac{\\mathrm{d}^m p_i(x \\vert \\mu_i, \\sigma_i)}{\\mathrm{d} x^m}
        , \\quad i=1,\\dots, n

    where :math:`n` is the size of the arguments ``data`` or ``bw``, and
    :math:`\\mu_i` and `:math:`\\sigma_i` are the elements of ``bw`` and
    ``data`` arguments, respectively. The order of integration, :math:`m`, can
    be ``-1`` representing the first anti-derivative (integral), ``0``
    representing no derivative (the function itself), or ``1`` representing the
    first derivative of the function.
    """

    t = (x - data) / bw

    if derivative == -1:
        phi = 0.5 * (1.0 + scipy.special.erf(t / numpy.sqrt(2.0)))
    elif derivative == 0:
        phi = (1.0 / (numpy.sqrt(2.0 * numpy.pi) * bw)) * \
            numpy.exp(-0.5 * t**2)
    elif derivative == 1:
        phi = (-t / (numpy.sqrt(2.0 * numpy.pi) * (bw**2))) * \
            numpy.exp(-0.5 * t**2)
    else:
        raise ValueError('"derivative" can be either "-1", "0", or "1".')

    return phi


# =================
# log normal kernel
# =================

def _log_normal_kernel(x, data, bw, derivative):
    """
    Probability density function (or its derivative or integral) of log-normal
    distribution centred at various data point and bandwidths.

    Parameters
    ----------

    data : array_like
        An array representing the geometric mean :math:`\\mu_i`.

    x : float
        A scalar value where the probability density function is evaluated at.

    bw : array_like
        An array of the same size as ``data`` representing the parameter
        :math:`\\sigma_i`. The ``bw`` stands for bandwidth of the kernel.

    derivative : int {-1, 0, 1}
        The derivative order of the probability density function and it can
        take either of the following values:

        * ``-1``: The first anti-derivative (integral) of the PDF. This is
          useful to compute the cumulative distribution function.
        * ``0``: The probability density function itself.
        * ``1``: The first derivative of the PDF.

    Returns
    -------

    p : array_like
        An array of the size of ``data`` and ``bw`` representing the PDF.

    See Also
    --------

    _normal_kernel

    Notes
    -----

    The PDF of log-normal distribution is:

    .. math::

        p_i(x \\vert \\mu, \\sigma) = \\frac{1}{\\sqrt{2 \\pi} \\sigma x}
        \\exp \\left( -\\frac{1}{2} \\left( \\frac{\\log(x) - \\mu}{\\sigma}
        \\right)^2 \\right).

    This function returns an array with elements:

    .. math::
        \\frac{\\mathrm{d}^m p_i(x \\vert \\mu_i, \\sigma_i)}{\\mathrm{d} x^m}
        , \\quad i=1,\\dots, n

    where :math:`n` is the size of the arguments ``data`` or ``bw``, and
    :math:`\\mu_i` and `:math:`\\sigma_i` are the elements of ``bw`` and
    ``data`` arguments, respectively. The order of integration, :math:`m`, can
    be ``-1`` representing the first anti-derivative (integral), ``0``
    representing no derivative (the function itself), or ``1`` representing the
    first derivative of the function.
    """

    t = (numpy.log(x) - numpy.log(data)) / bw

    if derivative == -1:
        phi = 0.5 * (1.0 + scipy.special.erf(t / numpy.sqrt(2.0)))
    elif derivative == 0:
        phi = (1.0 / (numpy.sqrt(2.0 * numpy.pi) * bw)) * \
            numpy.exp(-0.5 * t**2) / x
    elif derivative == 1:
        phi = -(1.0 / (numpy.sqrt(2.0 * numpy.pi) * (bw**3))) * \
            numpy.exp(-0.5 * t**2) * \
            (numpy.log(x) - numpy.log(data) + (bw**2)) / (x**2)
    else:
        raise ValueError('"derivative" can be either "-1", "0", or "1".')

    return phi


# =======================
# marchenko pastur kernel
# =======================

def _marchenko_pastur_kernel(x, data, bw, derivative):
    """
    Probability density function (or its derivative or integral) of Marchenko
    Pastur distribution centred at various data point and bandwidths.

    Parameters
    ----------

    data : array_like
        An array representing the geometric mean :math:`\\mu_i`.

    x : float
        A scalar value where the probability density function is evaluated at.

    bw : array_like
        An array of the same size as ``data`` representing the parameter
        :math:`\\sigma_i`. The ``bw`` stands for bandwidth of the kernel.

    derivative : int {-1, 0, 1}
        The derivative order of the probability density function and it can
        take either of the following values:

        * ``-1``: The first anti-derivative (integral) of the PDF. This is
          useful to compute the cumulative distribution function.
        * ``0``: The probability density function itself.
        * ``1``: The first derivative of the PDF.

    Returns
    -------

    p : array_like
        An array of the size of ``data`` and ``bw`` representing the PDF with
        mean ``data`` and standard deviation ``bw``.

    Notes
    -----

    The PDF of Marchenko-Pastur distribution is:

    .. math::

        p_i(x \\vert \\mu, \\sigma) = \\frac{1}{2 \\pi \\mu^2 \\sigma x}
        \\sqrt{(b - x) (x - a)}

    where

    .. math::

        a &= \\mu^2 (1 - \\sqrt{\\sigma})^2 \\\\
        b &= \\mu^2 (1 + \\sqrt{\\sigma})^2.

    This function returns an array with elements:

    .. math::
        \\frac{\\mathrm{d}^m p_i(x \\vert \\mu_i, \\sigma_i)}{\\mathrm{d} x^m}
        , \\quad i=1,\\dots, n

    where :math:`n` is the size of the arguments ``data`` or ``bw``, and
    :math:`\\mu_i` and `:math:`\\sigma_i` are the elements of ``bw`` and
    ``data`` arguments, respectively. The order of integration, :math:`m`, can
    be ``-1`` representing the first anti-derivative (integral), ``0``
    representing no derivative (the function itself), or ``1`` representing the
    first derivative of the function.
    """

    if numpy.isscalar(bw):
        b = numpy.tile(bw, data.size)

    a = data**2 * (1 - numpy.sqrt(bw))**2
    b = data**2 * (1 + numpy.sqrt(bw))**2
    phi = numpy.zeros_like(data)

    if derivative == -1:

        for i in range(phi.size):
            if x <= a[i]:
                if bw[i] > 1.0:
                    phi[i] = (bw[i] - 1.0) / bw[i]
                else:
                    phi[i] = 0.0
            elif x >= b[i]:
                phi[i] = 1.0
            else:
                sq = numpy.sqrt((b[i] - x) * (x - a[i]))
                r = numpy.sqrt((b[i] - x) / (x - a[i]))
                F = (1.0 / (2.0 * numpy.pi * bw[i])) * (
                        numpy.pi * bw[i] + sq / data[i]**2 -
                        (1.0+bw[i]) * numpy.arctan2(r**2-1.0, 2.0*r) +
                        (1.0-bw[i]) * numpy.arctan2(
                            a[i]*r**2 - b[i], 2.0*data[i]**2 * (1.0-bw[i])*r))
                if bw[i] > 1.0:
                    phi[i] = ((bw[i]-1.0) / (2.0*bw[i])) + F
                else:
                    phi[i] = F

    elif (derivative == 0) or (derivative == 1):

        # Nonzero part of the distribution
        nn = numpy.logical_and(b >= x, x >= a)

        denom = 2.0 * numpy.pi * bw[nn] * data[nn]**2
        sq = numpy.sqrt((b[nn] - x) * (x - a[nn]))

        if derivative == 0:
            phi[nn] = (1.0 / (denom * x)) * sq
        elif derivative == 1:
            phi[nn] = (a[nn] + b[nn] - 2*x) / (2.0 * denom * x * sq) - \
                sq / (denom * x**2)
    else:
        raise ValueError('"derivative" can be either "-1", "0", or "1".')

    return phi


# ==============
# estimate sigma
# ==============

def _estimate_sigma(
        data,
        pilot_density,
        log_scale=False,
        integrator='simpson'):
    """
    Estimate the parameter :math:`\\sigma` in normal or log-normal
    distribution.

    Parameters
    ----------

    data : array_like
        An array representing the data that is assumed to follow normal or
        log-normal distribution.

    pilot_density : array_like
        An array with the same size as ``data``, which represents the
        estimated density of the data.

    log_scale : bool, default=False
        If `False`, the data is assumed to have normal distribution. If
        `True`, the data is assumed to have log-normal distribution.

    integrator : str {'trapezoid', 'simpson', none'}, default=True
        The integraton method, which can be either the trapezoid or the
        Simpson's rule. If ``'none'``, no integration is performed, rather,
        the integrals are treated as a sum of discret values.

    Notes
    -----

    **Normal Distribution:**

    For normal distribution, :math:`\\sigma` is the standard deviation, which
    can be estimated empirically by

    .. math::

        \\sigma^2 = E[X^2] - E[X]^2

    **Log-Normal Distribution:**

    For the log-normal distribution, this the above no longer true. A common
    mistake of estimating :math:`\\sigma` for log-normal is as follows. If
    :math:`X` is log-normal, it means

    .. math::

        X = e^{\\mu + \\sigma z},

    where :math:`z` has the standard normal distribution. Hence, one might
    think :math:`\\sigma` is the standard deviation of :math:`\\ln X` and use
    estimate it with the standard deviation of :math:`\\ln X`. In fact, several
    papers do so. This is wrong!

    Here is the correct approach. For the log-normal distribution, we have

    .. math::

        E[X] &= e^{\\mu + \\frac{1}{2}\\sigma^2} \\\\
        E[X^2] &= e^{\\mu + 2 \\sigma^2}

    Hence

    .. math::

        \\sigma^2 = \\ln(E[X^2]) - 2 \\ln(E[X])

    **Compute the Expectation Operator:**

    Since we have the density of the data, we can use it and evaluate the
    expectation via an integral:

    .. math::

        E[X^n] = \\int x^n \\mathrm{d} \\P(x) = \\int x^n p(x) \\mathrm{d} x

    Also, the input density density may not be normalized (its integral may not
    be one). Jut to make sure, we normalize the above by :math:`E[X^0]`.

    The integral can be evaluated by the trapezoid or Simpson rule. Despite the
    Simpson rule uses quadratic function (3 data each at point), it has lower
    accuracy at sharp functions. If the spectral density (especially at the
    log scale) has very sharp spikes, trapezoid integration might be better.
    """

    # Zero-th, first, and second moments
    if integrator == 'simpson':
        # Using Simpson's rule
        mu0 = scipy.integrate.simpson(pilot_density, data)
        mu1 = scipy.integrate.simpson(data * pilot_density, data) / mu0
        mu2 = scipy.integrate.simpson(data**2 * pilot_density, data) / mu0
    elif integrator == 'trapezoid':
        # Using trapezoid rule
        mu0 = scipy.integrate.trapezoid(pilot_density, data)
        mu1 = scipy.integrate.trapezoid(data * pilot_density, data) / mu0
        mu2 = scipy.integrate.trapezoid(data**2 * pilot_density, data) / mu0
    elif integrator == 'none':
        # Treating integral as sum of discrete values
        if log_scale:
            mu1 = numpy.exp(numpy.mean(numpy.log(data)))
            mu2 = numpy.exp(numpy.mean(numpy.log(data**2)))
        else:
            mu1 = numpy.mean(data)
            mu2 = numpy.mean(data**2)
    else:
        raise ValueError('"integrator" should be "trapezoid", "simpson", ' +
                         'or "none".')

    if log_scale:
        # Log-normal distribution
        a = numpy.log(mu1)
        b = numpy.log(mu2)
        sigma = numpy.sqrt(b - 2*a)
    else:
        # Normal distribution
        sigma = numpy.sqrt(mu2 - mu1**2)

    return sigma


# =======================
# optimal amise bandwidth
# =======================

def _optimal_amise_bandwidth(sigma, n, log_scale=False):
    """
    Calculate the optimal bandwidth that minimizes asymptotic mean integrated
    square error (AMISE).

    Parameters
    ----------

    sigma : float
        The parameter :math:`\\sigma` in normal or log-normal probability
        distribution function.

    n : int
        Size of data.

    log_scale : bool, default=False
        If `False`, the data is assumed to have normal distribution. If
        `True`, the data is assumed to have log-normal distribution.

    Notes
    -----

    For normal distribution, this is known as Silverman's rule of thumb. See
    "A rule-of-thumb bandwidth estimator" in:
    https://en.wikipedia.org/wiki/Kernel_density_estimation

    For log-normal distribution, see equation (10) in:
    https://arxiv.org/pdf/1804.08365.pdf
    """

    if log_scale:
        # Log-normal distribution
        h = ((8.0 * numpy.exp(sigma**2 / 4.0)) /
             (sigma**4 + 4.0*sigma**2 + 12.0))**(1.0/5.0) * \
                    sigma / (n**(1.0/5.0))
    else:
        # Normal distribution
        h = (4.0 / (3.0 * n))**(1.0/5.0) * sigma

    return h


# ==============
# adaptive scale
# ==============

def _adaptive_scale(pilot_density, sensitivity=0.5):
    """
    For a pilot density function, compute an adaptive bandwidth scale.

    Parameters
    ----------

    pilot_density : array_like
        An array of the size of the data, representing the density of the data.
        The density can be a rough approximation of the actual density, hence
        called the pilot density.

    sensitivity : float, default=0.5
        The scalar between ``0`` and ``1``. Zero means no sensitivity, and as
        such, the bandwidth will not be adaptive, rather is a constant for all
        data points. In contrast, larger sensitivity leads to a bandwidth with
        more variations across data points.
    """

    # Geometric mean of data
    g = numpy.exp(numpy.mean(numpy.log(pilot_density)))

    # Scale the is inversely propositional to density
    scale = (pilot_density / g)**(-sensitivity)

    return scale


# =========
# diff data
# =========

def diff_data(
        data,
        neighbor=1,
        smooth_width=None,
        smooth_iter=3,
        log_scale=False):
    """
    Difference of data points.

    Parameters
    ----------

    data : array_like
        Array of data

    neighbor : int, default=1
        The neighbor ``k`` means each point ``data[i]`` is differentiated with
        respect to its k-th forward neighbor ``data[i+k]``.

    smooth_with : int, default=None
        The length of smoothing filter to apply on the differentiated data. If
        `None`, the square root of the data size is considered.

    smooth_iter : int default=3
        Number of iterations to apply smoothing filter to the differentiated
        data.

    log_scale : bool, default=False
        If `True`, the logarithm of the data is differentiated.
    """

    if log_scale:
        data_ = numpy.log(data)
    else:
        data_ = data

    # Differencing filter
    diff_filter = numpy.zeros((neighbor+1, ))
    diff_filter[0] = 1
    diff_filter[-1] = -1

    # Apply filter to data
    diff_data = numpy.convolve(data_, diff_filter, mode='valid')

    # maintain the length of the original data
    diff_data = numpy.r_[diff_data, diff_data[1-diff_filter.size:]]

    # Smoothing window length
    if smooth_width is None:
        smooth_width = int(numpy.sqrt(data_.size))

    # Apply smoothing filter multiple times
    for i in range(smooth_iter):
        diff_data = scipy.ndimage.uniform_filter1d(
            diff_data, size=smooth_width)

    return diff_data


# =============
# max bandwidth
# =============

def _max_bandwidth(data, bw):
    """
    """

    max_bw2 = numpy.log(data) - numpy.log(numpy.min(data)) + bw[0]**2
    max_bw = numpy.sqrt(max_bw2)

    return max_bw


# ==================
# adaptive bandwidth
# ==================

def _adaptive_bandwidth(
        data,
        kernel='normal',
        sensitivity=0.5,
        bw_iter=3,
        scale_bw=1.0,
        min_quant=0.5,
        log_scale=False,
        integrator='simpson'):
    """
    TODO
    """

    # Lower bound of bandwidth based on data difference
    diff_data_ = diff_data(data, neighbor=1, smooth_width=None, smooth_iter=3,
                           log_scale=log_scale)
    # min_bw = numpy.sqrt(diff_data_)  # works better
    min_bw = diff_data_ / (2 * numpy.sqrt(2) * scipy.special.erfinv(min_quant))

    bw_list = []

    n = data.size

    for i in range(bw_iter):
        if i == 0:
            # The first rough estimate of density. Note that in log scale,
            # this density does not have integral of one. This needs to be
            # normalized later, for instance in _estimate_sigma function.
            pilot_density = numpy.ones((n, )) / n
        else:
            # Estimate density at the data points themselves
            pilot_density = estimate_density(data, data, bw_list[i-1],
                                             kernel=kernel, derivative=0,
                                             log_scale=log_scale)

        sigma = _estimate_sigma(data, pilot_density, log_scale=log_scale,
                                integrator=integrator)
        scale = _adaptive_scale(pilot_density, sensitivity=sensitivity)
        h = _optimal_amise_bandwidth(sigma, n, log_scale=log_scale)
        bw = h * scale * scale_bw

        if i != 0:
            # too_small_bw = numpy.logical_and(bw < min_bw, bw < h)
            too_small_bw = bw < min_bw
            bw[too_small_bw] = min_bw[too_small_bw]

            if log_scale:
                max_bw = _max_bandwidth(data, bw)
                # too_large_bw = numpy.logical_and(bw > max_bw, bw > h)
                too_large_bw = bw > max_bw
                bw[too_large_bw] = max_bw[too_large_bw]
            else:
                max_bw = None

        bw_list.append(bw)

    return bw_list, min_bw, max_bw


# ================
# estimate density
# ================

def estimate_density(
        data,
        x,
        bw,
        kernel='normal',
        derivative=0,
        log_scale=False):
    """
    Estimate density of data got a given bandwidth.

    Parameters
    ----------

    data : array_like
        Input dataset to estimate its density.

    x : scalar or array_like
        A point or an array of points where the density is evaluated at.

    bw : array_like
        The bandwidth of the kernel. This should be an array with the same
        size as the data.

    derivative : int {-1, 0, 1}, default=0
        The derivative of density to be estimated. This can be either ``-1``,
         the integral, ``0``, the function itself, or ``1``, the derivative of
         the density.

    log_scale : bool, default=False
        If `True`, the range of data is assumed to be positive and at the
        logarithmic scale. In this case, the log-normal distribution is used as
        the kernel to estimate the data density. In contrast, if `False`, the
        normal distribution is used.
    """

    # Use log-normal if log scale
    if kernel == 'normal':
        if log_scale:
            kernel_func = _log_normal_kernel
        else:
            kernel_func = _normal_kernel
    elif kernel == 'marchenko_pastur':
        kernel_func = _marchenko_pastur_kernel
    else:
        raise ValueError('"kernel" should be "normal" or "marchenko_pastur".')

    density = numpy.zeros_like(x)

    # Convolve kernel with the data
    for i in range(x.size):
        phi = kernel_func(x[i], data, bw, derivative)
        density[i] = numpy.sum(phi) / data.size

    return density


# ===
# kde
# ===

def kde(
        data,
        x=None,
        bw=None,
        kernel='normal',
        log_scale=False,
        cumulative=False,
        sensitivity=0.5,
        bw_iter=3,
        scale_bw=1.0,
        min_quant=0.5,
        integrator='simpson'):
    """
    Kernel density estimation.

    Parameters
    ----------

    data : array_like
        Input dataset to estimate its density.

    x : scalar or array_like, default=None
        A point or an array of points where the density is evaluated at. If not
        provided, an array of `1000` points in the range slightly larger than
        the range of the input data is considered.

    bw : float or array_like, default=None
        The bandwidth of the kernel. If a scalar is given, the given value is
        considered as the bandwidth for all data points. If `None`, an
        adaptive bandwidth is automatically computed.

    log_scale : bool, default=False
        If `True`, the range of data is assumed to be positive and at the
        logarithmic scale. In this case, the log-normal distribution is used as
        the kernel to estimate the data density. If `False`, the normal
        distribution is used.

    cumulative : bool, default=False
        If `False`, the probability density function of the data is estimated.
        If `True`, the cumulative distribution function is estimated.
    """

    if bw is None:
        # Adaptive bw
        bw_list, min_bw, max_bw = _adaptive_bandwidth(
                data, kernel=kernel, sensitivity=sensitivity, bw_iter=bw_iter,
                scale_bw=scale_bw, min_quant=min_quant, log_scale=log_scale,
                integrator=integrator)
        bw = bw_list[-1]
    else:
        if numpy.isscalar(bw):
            bw = numpy.tile(bw, data.size)
        elif isinstance(bw, (list, tuple)):
            bw = numpy.array(bw)

            # Check bw size is the same as data size
            if bw.size != data.size:
                raise ValueError('When "bw" is given as an array, the "bw" ' +
                                 'size should be the same as "data" size.')

        bw_list = [bw]
        min_bw = None
        max_bw = None

    # Use the first anti-derivative if estimating the CDF instead of PDF
    if cumulative:
        # Estimate CDF
        derivative = -1
    else:
        # Estimate PDF
        derivative = 0

    # Check log scale
    if (log_scale is True) and (numpy.any(data < 0.0)):
        raise ValueError('"log_scale" cannot be set to "True" for negative ' +
                         ' data.')

    # Check sensitivity
    if (sensitivity < 0.0) or (sensitivity > 1.0):
        raise ValueError('"sensitivity" should be between "0" and "1".')

    if x is None:

        data_min = numpy.min(data)
        data_max = numpy.max(data)

        if log_scale:
            log_x_min = numpy.floor(numpy.log10(data_min))
            log_x_max = numpy.ceil(numpy.log10(data_max))
            x = numpy.logspace(log_x_min, log_x_max, 1000)
        else:
            data_scale = 10.0**int(numpy.log10(data_max - data_min) + 0.25)
            x_min = numpy.floor(data_min/data_scale - 0.25) * data_scale
            x_max = numpy.ceil(data_max/data_scale + 0.25) * data_scale
            x = numpy.linspace(x_min, x_max, 1000)

    density = estimate_density(data, x, bw, kernel=kernel,
                               derivative=derivative, log_scale=log_scale)

    return density, x, bw_list, min_bw, max_bw
