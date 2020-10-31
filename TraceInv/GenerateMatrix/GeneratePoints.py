# =======
# Imports
# =======

import numpy

# ===============
# Generate Points
# ===============

def GeneratePoints(NumPoints,GridOfPoints=True):
    """
    Generates two column vectors ``x`` and ``y`` of data points.

    :param NumPoints: Depending on ``GridOfPoints``, this is either the number of points or the number of grid points along an axis.
    :type NumPoints: int

    :param GridOfPoints: if ``True``, the number of ``NumPoints^2`` points are generated on a grid inside the unit square.
        If ``False``, the number of ``NumPoints`` points are generated randomly inside the unit square.
    """

    print('Generate data ...')

    # Grid of points
    if GridOfPoints == True:
        x_axis = numpy.linspace(0,1,NumPoints)
        y_axis = numpy.linspace(0,1,NumPoints)
        x_mesh,y_mesh = numpy.meshgrid(x_axis,y_axis)

        # Column vectors of x and y of data
        x = x_mesh.ravel()
        y = y_mesh.ravel()
    else:
        # Randomized points in a square area
        x = numpy.random.rand(NumPoints)
        y = numpy.random.rand(NumPoints)

    return x,y
