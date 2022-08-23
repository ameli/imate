.. _generate_matrix_UserGuide:

*************************************************
Generate Matirx (:mod:`imate.generate_matrix`)
*************************************************

The sub package :mod:`imate.generate_matrix` is tangent to the purpose of the imate package as it does not provide a computational utility. Rather, it generates sample matrices for test purposes only. imate works with generic invertible matrices. However, in most applications where imate can be utilized, the matrices are of the form of covariance or correlation matrices (such as in machine learning purposes). Such matrices have these characteristics:

1. symmetric
2. positive-definite or positive-semidefinite

This sub-package generates such matrices to tests imate on applicable matrices.

=====
Usage
=====

.. code-block:: python

   >>> from imate import generate_matrix
   
   >>> # Generate a symmetric positive-definite matrix of the shape (20**2,20**2)
   >>> K = generate_matrix(NumPoints=20)

The matrix ``K`` in the above has the shape ``(20**2,20**2)``, is symmetric and positive-definite.

==========
Parameters
==========

The :mod:`imate.generate_matrix` module accepts the following attributes as input argument to customize the generated matrix.

.. attribute:: NumPoints
   :type: int
   :value: 20

   Determines the size of the matrix.
   
   * When the parameter :data:`GridOfPoints` is set to ``True``, the matrix has the shape of ``(NumPoints**2,NumPoints**2)``.
     This is corresponding to the mutual correlation between a set of :math:`n^2` points generated on an equidistanced rectangular
     grid of :math:`n \times n` points in the unit square.
   * When the parameter :data:`GridOfPoints` is set to ``False``, the matrix has the shape of ``(NumPoints,NumPoints)``. 
     This is corresponding to the mutual corelation between a set of :math:`n` points generated randomly in the unit square.

.. attribute:: GridOfPoints
   :type: bool
   :value: True

   Determines the structure of the points which the correlation matrix if formed based on.

   * When set to ``True``, the set of :math:`n \times n` points is generated on an equispaced rectangular grid in the unit square. 
     Here, :math:`n` is set by :attr:`NumPoints` argument. As a result, the generated matrix is of the shape :math:`(n^2 \times n^2)`.
   * When set to ``False``, the set of :math:`n \times n` points is generated randomly in the unit square. 
     Here, :math:`n` is set by :attr:`NumPoints` argument. As a result, the generated matrix is of the shape :math:`(n \times n)`.

.. attribute:: DecorrelationScale
   :type: float
   :value: 0.1

   Decorrelation scale parameter :math:`\rho` of the :ref:`Correlation Function`.
   Generally, :math:`\rho > 0`, but in this class it should be set to :math:`\rho \in (0,1]`.
   Smaller :math:`\rho` yields correlation matrix closer to the identity matrix, whereas
   larget :math:`\rho` leads to a more correlated matrix.

.. attribute:: nu
   :type: float
   :value: 0.5

   Sets the smoothness of the stochastic process that is represented by the correlation matrix. (see the parameter :math:`\nu` in :ref:`Correlation Function`).
   :math:`\nu` should be positive. Small values of :math:`\nu` yields correlation matrix closer to the identity matrix.
   For values :math:`\nu > 100`, the Gaussian correlation function is used which corresponds to :math:`\nu \to \infty`.

.. attribute:: UseSparse
   :type: bool
   :value: False

   Determines to generate dense or sparse matrix. If set to ``True``, the sparse denisity of the matrix can be set by :attr:`KernelThreshold` argument.

.. attribute:: KernelThreshold
   :type: float
   :value: 0.03

   To sparsify the matrix, the matrix elements smaller than this threshold are considered to be zero
   (see parameter :math:`\kappa` in :ref:`Sparse Matrix`).
   Increasing the threshold yields a more sparse matrix.

.. attribute:: RunInParallel
   :type: bool
   :value: False

   If ``True``, it eneables parallel processing of the rows and columns of thec correlation matrix using ``ray`` package. 
   This option is particularly useful for sparse large matrices.

.. attribute:: Plot
   :type: bool
   :value: False

   If set to ``True``, it plots the generated matrix.

   * If no graphical backend exists (such as running the code on a remote server or manually disabling the X11 backend), the plot will not be shown, rather, it will ve saved as an ``svg`` file in the current directory. 
   * If the executable ``latex`` is on the path, the plot is rendered using :math:`\rm\LaTeX`, which then, it takes a bit longer to produce the plot. 
   * If :math:`\rm\LaTeX` is not installed, it uses any available San-Serif font to render the plot.

   .. note::

       To manually disable interactive plot display, and save the plot as ``SVG`` instead, add the following in the
       very begining of your code before importing ``imate``:

       .. code-block:: python
         
           >>> import os
           >>> os.environ['IMATE_NO_DISPLAY'] = 'True'

.. attribute:: Verbose:
   :type: bool
   :value: False

   If ``True``, prints some information during the process.

====================
Mathematical Details
====================

The module :class:`generate_matrix <imate.generate_matrix>` produces a correlation matrix :math:`\mathbf{K}` as follows.

1. It generates a set of spatial points :math:`\boldsymbol{x}_i \in \mathcal{D}` (see :ref:`Generate Set of Points`).
2. Using a given correlation function :math:`K: \mathcal{D} \times \mathcal{D} \to [0,1]` (see :ref:`Correlation Function`), the component :math:`K_{ij}` of the matrix :math:`\mathbf{K}` is then computed by

   .. math::

       K_{ij} = K(\boldsymbol{x}_i,\boldsymbol{x}_j)

----------------------
Generate Set of Points
----------------------

In this module, the set of points is defined in two ways:

**On a Grid:** When :attr:`GridOfPoints` is set to ``True``, a set of :math:`n \times n` points are generated on a equi-distanced grid in the unit square :math:`\mathcal{D} = [0,1]^2`.

**Random:** When :attr:`GridOfPoints` is set to ``False``, a set of :math:`n` points are generated with uniform random distribution in the unit square :math:`\mathcal{D} = [0,1]^2`.

--------------------
Correlation Function
--------------------

The module uses the Matern correlation function (see [Matern-1960]_, or [Stein-1999]_) defined by

.. math::
    K(\boldsymbol{x},\boldsymbol{x}'|\rho,\nu) = 
    \frac{2^{1-\nu}}{\Gamma(\nu)}
    \left( \sqrt{2 \nu} \frac{\| \boldsymbol{x} - \boldsymbol{x}' \|}{\rho} \right) 
    K_{\nu}\left(\sqrt{2 \nu}  \frac{\|\boldsymbol{x} - \boldsymbol{x}' \|}{\rho} \right)

where

    * :math:`\Gamma( \cdot)` is the Gamma function (see [DLMF]_), 
    * :math:`\| \cdot \|` is the Euclidean distance,
    * :math:`K_{\nu}(\cdot)` is the modified Bessel function of the second kind of order :math:`\nu` (see [DLMF]_).

and with the hyperparameters

    * :math:`\rho > 0`: **decorrelation scale** of the correlation function (see :attr:`DecorrelationScale`).
    * :math:`\nu > 0`: **smoothness** of the stochastic process represented by the correlation function (see :attr:`nu`).

When :math:`\| \boldsymbol{x} - \boldsymbol{x}' \| = 0`, the correlation function is defined to be :math:`1`. This corresponds to the diagonals of the correlation matrix :math:`\mathbf{K}` as the correlation between :math:`\boldsymbol{x}_i` with itself.

.. warning::

    When the distance :math:`\| \boldsymbol{x} - \boldsymbol{x}' \| \to 0^+`, the correlation function produces 
    :math:`\frac{0}{0}`. Thus, if the distance is *not* exactly zero, but close to zero, this function might produce unstable results.    

If :math:`\nu` is half integer, the Matern function has exponential form.
    
* At :math:`\nu = \frac{1}{2}`, the Matern correlation function is just the exponential decay function

  .. math::
      K(\boldsymbol{x},\boldsymbol{x}'|\rho,\nu) = 
      \exp \left( -\frac{\| \boldsymbol{x} - \boldsymbol{x}'\|}{\rho} \right)

* At :math:`\nu = \frac{3}{2}`
  
  .. math::
      K(\boldsymbol{x},\boldsymbol{x}'|\rho,\nu) =
      \left( 1 + \sqrt{3} \frac{\| \boldsymbol{x} - \boldsymbol{x}'\|}{\rho} \right)
      \exp \left( - \sqrt{3} \frac{\| \boldsymbol{x} - \boldsymbol{x}'\|}{\rho} \right)
  
* At :math:`\nu = \frac{5}{2}`
  
  .. math::
      K(\boldsymbol{x},\boldsymbol{x}'|\rho,\nu) =
      \left( 1 + \sqrt{5} \frac{\| \boldsymbol{x} - \boldsymbol{x}'\|}{\rho} + \frac{5}{3} \frac{\| \boldsymbol{x} - \boldsymbol{x}'\|^2}{\rho^2} \right) 
      \exp \left( -\sqrt{5} \frac{\| \boldsymbol{x} - \boldsymbol{x}'\|}{\rho} \right)
  
* At :math:`\nu = \infty`, the Matern function approaches the *Gaussian correlation function* (also known as square exponential decay function):
  
  .. math::
      K(\boldsymbol{x},\boldsymbol{x}'|\rho,\nu) = 
      \exp \left( -\frac{1}{2} \frac{\| \boldsymbol{x} - \boldsymbol{x}'\|^2}{\rho^2} \right)

.. note::

    At :math:`\nu > 100`, this module assumes that :math:`\nu = \infty` and the Gaussian correlation function is used.

-------------
Sparse Matrix
-------------

To produce a sparse matrix, a compactly supported kernel is used by tapering the tail of the correlation function :math:`K`. Define the *indicator function*

.. math::

    \mathbf{1}_{K > \kappa}(\boldsymbol{x},\boldsymbol{x}') = 
    \begin{cases}
        1, & K > \kappa \\
        0, & K \leq \kappa \\
    \end{cases}

The parameter :math:`\kappa` is the threshold of the indicator function (see :attr:`KernelThreshold`). By multiplying the indicator function :math:`\mathbf{1}_{K > \kappa}` with the correlation kernel :math:`K`, any correlation value less than the threshold :math:`\kappa` becomes zero, hence makes the correlation matrix sparse. 

The *sparse density* :math:`d` of the sparse matrix (ratio of non-zero elements over all elements) is

.. math::

   d = \pi \left( \rho \log \kappa \right)^2

.. warning::

    If the threshold :math:`\kappa` is too large, the sparse matrix might no longer be positive-definite [Genton-2002]_. On the other hand, if :math:`\kappa` is not small enough, the matrix will not be sparse enough.


=======
Example
=======

Generate a matrix of the size :math:`20^2 \times 20^2` based on mutual correlation of a rectangular grid of :math:`20 \times 20` points in the unit square

.. code-block:: python

  >>> from imate import generate_matrix
  >>> A = generate_matrix(NumPoints=20)

Generate a correlation matrix of the size :math:`20 \times 20` based on :math:`20` random points in unit square. Default for :attr:`GridOfPoints` is ``True``.

.. code-block:: python

  >>> A = generate_matrix(NumPoints=20,GridOfPoints=False)

Generate a matrix of the size :math:`20^2 \times 20^2` with stronger spatial correlation.

.. code-block:: python

  >>> A = generate_matrix(NumPoints=20,DecorrelationScale=0.3)

Generate a correlation matrix with more smoothness.

.. code-block:: python

  >>> A = generate_matrix(NumPoints=20,nu=2.5)

Sparsify correlation matrix (makes all entries below :attr:`KernelThreshold` to zero).

.. code-block:: python

  >>> A = generate_matrix(NumPoints=20,UseSparse=True,KernelThreshold=0.03)

For very large correlation matrices, generate the rows and columns are generated in parallel.
To use :attr:`RunInParallel` option, the package ``ray`` should be installed.

.. code-block:: python

  >>> A = generate_matrix(NumPoints=100,UseSparse=True,RunInParallel=True)

Plot the matrix using :attr:`Plot` option:

.. code-block:: python
   
   >>> A = generate_matrix(NumPoints=15,DecorrelationScale=0.3,Plot=True)

.. image:: ./images/CorrelationMatrix.svg
   :align: center

==========
References
==========

.. [Matern-1960] Matérn, B. (1960). Spatial variation. In *Meddelanden från Statens Skogsforskningsinstitut*, volume 49, No. 5. Almänna Förlaget, Stockholm. Second edition (1986), Springer-Verlag, Berlin. `doi: 10.1007/978-1-4615-7892-5 <https://www.springer.com/gp/book/9781461578925>`_

.. [Genton-2002] Genton, M. G. (2002). Classes of kernels for machine learning: A statistics perspective. *Journal of Machine Learning Research*, 2, 299–312. `doi: 10.5555/944790.944815 <https://dl.acm.org/doi/10.5555/944790.944815>`_

.. [Stein-1999] Stein, M. L. (1999). Interpolation of Spatial Data: Some Theory for Kriging. Springer Series in Statistics. Springer New York. `doi: 10.1007/978-1-4612-1494-6 <https://www.springer.com/gp/book/9780387986296>`_

.. [DLMF] `NIST Digital Library of Mathematical Functions. <http://dlmf.nist.gov/>`_, F. W. J. Olver, A. B. Olde Daalhuis, D. W. Lozier, B. I. Schneider, R. F. Boisvert, C. W. Clark, B. R. Miller, B. V. Saunders, H. S. Cohl, and M. A. McClain, eds. `doi: 10.1023/A:1022915830921 <https://doi.org/10.1023/A:1022915830921>`_

===
API
===

.. automodapi:: imate.generate_matrix
