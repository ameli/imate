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

__all__ = ['generate_covs']


# =============
# generate covs
# =============

def generate_covs(size, input_dim=1, output_dim=1, grid=True, corr=False):
    """
    Generates covariances.

    :param size: Indicates the number of generated points as follows:
        * If ``grid`` is ``True``, the points are equi-distanced structured
          grid where there are ``size`` points along each axis. Hence, the
          overall number of points are equal to ``size**input_dim``.
        * If ``grid is ``False``, random points are generated with uniform
          distribution. The overall number of points are equal to ``size``.
    :type size: int

    input_dim : int, default=1
        The dimension of the space of points to generate the correlation
        matrix.

    :param output_dim: The output dimension of the model.
    :type output_dim: int

    :param grid: Determines whether the points are generated on a structured
        grid (if ``True``) or randomly (if ``False``).
    :type grid: bool

    :param corr: If `True, returns correlation instead.
    :param corr: bool

    :return: 2D array of covariances. The number of rows equals the output
        dimension.
    :rtype: numpy.ndarray
    """

    if grid:
        num_points = size**input_dim
    else:
        num_points = size

    covs = numpy.zeros((output_dim, num_points * output_dim),
                       dtype=numpy.float64, order='C')

    if output_dim == 1:
        if corr is True:
            covs[0, :] = 1.0
        else:
            covs[0, :] = numpy.abs(numpy.random.randn(num_points, ))
    else:
        for j in range(num_points):
            cov = numpy.random.randn(output_dim, output_dim)
            cov = cov.T @ cov

            if corr is True:
                stddev = numpy.sqrt(numpy.diag(cov))
                cov = cov / numpy.outer(stddev, stddev)

            cov = scipy.linalg.sqrtm(cov)
            covs[:, j*output_dim:(j+1)*output_dim] = cov

    return covs
