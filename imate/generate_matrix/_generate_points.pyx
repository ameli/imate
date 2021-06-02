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
from libc.time cimport time, time_t
from libc.stdlib cimport rand, srand, RAND_MAX

__all__ = ['generated_points']


# ===============
# generate points
# ===============

def generate_points(size, dimension=2, grid=True):
    """
    Generates a set of points in the unit hypercube.

    :param size: Indicates the number of generated points as follows:
        * If ``grid`` is ``True``, the points are equi-distanced structured
          grid where there are ``size`` points along each axis. Hence, the
          overall number of points are equal to ``size**dimension``.
        * If ``grid is ``False``, random points are generated with uniform
          distribution. The overall number of points are equal to ``size``.
    :type size: int

    :param dimension: The dimension of the hypercube.
    :type dimension: int
    :param grid: Determines whether the points are generated on a structured
        grid (if ``True``) or randomly (if ``False``).
    :type grid: bool

    :return: 2D array of the coordinates of the generated points. The first
        index of the array corresponds to the point Ids, and the second axis is
        each of the dimensional coordinates.
    :rtype: numpy.ndarray
    """

    if grid:
        num_points = size**dimension
        coords = numpy.zeros((num_points, dimension), dtype=float)
        _generate_grid_points(size, dimension, coords)

    else:
        num_points = size
        coords = numpy.zeros((num_points, dimension), dtype=float)
        _generate_random_points(num_points, dimension, coords)

    return coords


# ====================
# generate grid points
# ====================

cdef void _generate_grid_points(
        const int grid_size,
        const int dimension,
        double[:, ::1] coords) nogil:
    """
    Generates set of structured grid points in the unit hypercube and outputs
    their coordinates. The grid points are equi-distanced.

    The number of generated points is equal to ``grid_size**dimension``. Hence,
    the output array, ``coords` has the shape
    ``(grid_size**dimension, dimension)``.

    :param grid_size: Number of points along each axis of the grid of points.
    :type grid_size: int

    :param dimension: Dimension of unit hypercube.
    :type dimension: int

    :param coords: Coordinates of generated points. First index of this array
        is the point id from ``0`` to ``num_points`` and the second index is
        the dimension of point coordinate from ``0`` to ``dimension - 1``. This
        array is the output.
    :type coords: cython memoryview (double)
    """

    cdef int point_id
    cdef int dim
    cdef int shift_next_dim
    cdef int num_points = grid_size**dimension

    for point_id in range(num_points):

        # Initialize first point
        if point_id == 0:
            for dim in range(dimension):
                coords[point_id][dim] = 0.0

        else:
            # Increment a coordinate with respect to the previous point
            shift_next_dim = 1
            for dim in range(dimension):
                if shift_next_dim:

                    # If a point reached end of axis, do not increment
                    if coords[point_id - 1][dim] >= grid_size - 1:

                        # Reset back to origin of axis
                        coords[point_id][dim] = 0.0

                        # Increment on axis of the next dimension
                        shift_next_dim = 1

                    else:
                        # Increment the current axis by integer (rescale later)
                        coords[point_id][dim] = coords[point_id - 1][dim] + 1.0

                        # Do not increment other axes as it has been done once
                        shift_next_dim = 0
                else:
                    # Copy from previous point (no increment)
                    coords[point_id][dim] = coords[point_id - 1][dim]

    # Scale an integer grid to unit hypercube
    cdef double dx = 1.0 / (grid_size - 1.0)
    for point_id in range(num_points):
        for dim in range(dimension):
            coords[point_id][dim] = coords[point_id][dim] * dx


# ======================
# generate random points
# ======================

cdef void _generate_random_points(
        const int size,
        const int dimension,
        double[:, ::1] coords) nogil:
    """
    Generates set of spatial random points in the unit hypercube and outputs
    their coordinates. Coordinates in each axis have uniform distribution.

    The number of generated points is equal to ``num_points``. Hence, the
    output array, ``coords` has the shape (num_points, dimension)``.

    :param size: Number of generated points.
    :type size: int

    :param dimension: Dimension of unit hypercube.
    :type dimension: int

    :param coords: Coordinates of generated points. First index of this array
        is the point id from ``0`` to ``num_points`` and the second index is
        the dimension of point coordinate from ``0`` to ``dimension - 1``. This
        array is the output.
    :type coords: cython memoryview (double)
    """

    # Set the seed of rand function
    cdef time_t t
    srand((<unsigned int> time(&t)))

    cdef int dim
    cdef int point_id

    for point_id in range(size):
        for dim in range(dimension):
            coords[point_id][dim] = rand() / (<double> RAND_MAX)
