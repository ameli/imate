Generate Matrix
===============

This package generates a symmetric and positive-definite matrix for test purposes. The main module is ``GenerateMatrix`` which defines the main function ``GenerateMatrix()``.

Helper modules are:

* ``GeneratePoints``: Generates a set of points in the unit cirle. The set of points can be either on a Cartesian grid or randomly generated.
* ``CorrelationMatrix``: Generates a corelation matrix based on the mutual distance of the set of points. The Matern correlation kernel is used to generate the correlation function. The main function of this module is ``CorrelationMatrix()`.
