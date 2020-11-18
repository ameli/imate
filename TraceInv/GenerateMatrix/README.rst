===============
Generate Matrix
===============

This package generates a symmetric and positive-definite matrix for test purposes. The main module is ``GenerateMatrix`` which defines the main function ``GenerateMatrix()``.

Helper modules are:

* ``GeneratePoints``: Generates a set of points in the unit cirle. The set of points can be either on a Cartesian grid or randomly generated.
* ``CorrelationMatrix``: Generates a corelation matrix based on the mutual distance of the set of points. The Matern correlation kernel is used to generate the correlation function. The main function of this module is ``CorrelationMatrix()``.

-------
Example
-------

Generate a matrix of the shape ``(20**2,20**2)`` based on mutual correlation of a grid of 20x20 points on unit square

.. code-block:: python

   >>> from TraceInv import GenerateMatrix
   >>> A = GenerateMatrix(NumPoints=20)

Generate a correlation matrix of shape ``(20,20)`` based on 20 random points in unit square. Default for ``GridOfPoints`` is True.

.. code-block:: python

   >>> A = GenerateMatrix(NumPoints=20,GridOfPoints=False)

Generate a matrix of shape ``(20**2,20**2)`` with stronger spatial correlation. Default for ``DecorrelationScale`` is ``0.1``.

.. code-block:: python

   >>> A = GenerateMatrix(NumPoints=20,DecorrelationScale=0.3)

Generate a corelation matrix with more smoothness.. Default for ``nu`` is ``0.5``.

.. code-block:: python

   >>> A = GenerateMatrix(NumPoints=20,nu=2.5)

Sparsify correlation matrix (makes all entries below 0.03 to zero). Default for ``KernelThreshold`` is ``0.03``.

.. code-block:: python

   >>> A = GenerateMatrix(NumPoints=20,UseSparse=True,KernelThreshold=0.03)

For very large correlation matrices, generate the rows and columns are generated in parallel.
To use ``RunInParallel`` option, the package ``ray`` should be installed.

.. code-block:: python

   >>> A = GenerateMatrix(NumPoints=100,UseSparse=True,RunInParallel=True)
