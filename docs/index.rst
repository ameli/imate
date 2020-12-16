********
TraceInv
********

|travis-devel| |codecov-devel| |docs| |licence| |platforms| |conda-version| |conda| |format| |pypi| |implementation| |pyversions|

This python package can perform the following matrix operations:

#. Compute the *log-determinant* of dense or sparse matrices.
#. Compute the *trace of the inverse* of dense or sparse matrices.
#. Interpolate the trace of the inverse of *one-parameter affine matrix functions*.

These matrix operations frequently appear in many applications in computational physics, computational biology, data analysis, and machine learning. A few examples are regularization in inverse problems, model-selection in machine learning, and more broadly, in parameter optimization of statistical models.

A common difficulty in such application is that the matrices are generally large and inverting them is impractical. Because of this, evaluation of their trace or log-determinant is a computational challenge. Many algorithms have been developed to address such computational challenge, such as efficient sparse matrix factorizations and randomized estimators. This package aims to implement some of these methods.

.. toctree::
    :maxdepth: 1
    :caption: Documentation

    Introduction <introduction>
    Install <install>
    Quick Start <quickstart>
    Examples <examples>

.. toctree::
    :maxdepth: 1
    :caption: Sub-packages User Guide

    Compute Log Determinant <ComputeLogDeterminant>
    Compute Trace of Inverse <ComputeTraceOfInverse>
    Interpolate Trace of Inverse <InterpolateTraceOfInverse>
    Generate Matrix <GenerateMatrix>

.. toctree::
    :maxdepth: 1
    :caption: Development
              
    Package API <_modules/modules>
    Running Tests <tests>
    Change Log <changelog>

.. =======
.. Modules
.. =======
..
.. .. autosummary::
..    :toctree: _autosummary
..    :recursive:
..    :nosignatures:
..
..    TraceInv.ComputeTraceOfInverse
..    TraceInv.InterpolateTraceOfInverse
..    TraceInv.GenerateMatrix

=========
Tutorials
=========

|binder|

A tutorial and demonstration of examples can be found with `online interactive Jupyter notebook <https://mybinder.org/v2/gh/ameli/TraceInv/HEAD?filepath=notebooks%2FInterpolateTraceOfInverse.ipynb>`_.

============
Useful Links
============

.. For users
..     * `Documentation <https://ameli.github.io/TraceInv/index.html>`_
..     * `PyPi package <https://pypi.org/project/TraceInv/>`_
..     * `Source code <https://github.com/ameli/TraceInv>`_
..
.. For developers
..     * `API <https://ameli.github.io/TraceInv/_modules/modules.html>`_
..     * `Travis-CI <https://travis-ci.com/github/ameli/TraceInv>`_
..     * `Codecov <https://codecov.io/gh/ameli/TraceInv>`_

+---------------------------------------------------------------+-------------------------------------------------------------------+
|    For users                                                  | For developers                                                    |
+===============================================================+===================================================================+
| * `Anaconda package <https://anaconda.org/s-ameli/traceinv>`_ | * `API <https://ameli.github.io/TraceInv/_modules/modules.html>`_ |
| * `PyPi package <https://pypi.org/project/TraceInv/>`_        | * `Travis-CI <https://travis-ci.com/github/ameli/TraceInv>`_      |
| * `Source code <https://github.com/ameli/TraceInv>`_          | * `Codecov <https://codecov.io/gh/ameli/TraceInv>`_               |
+---------------------------------------------------------------+-------------------------------------------------------------------+

=================
How to Contribute
=================

We welcome contributions via `Github's pull request <https://github.com/ameli/TraceInv/pulls>`_. If you do not feel comfortable modifying the code, we also welcome feature request and bug report as `Github issues <https://github.com/ameli/TraceInv/issues>`_.

================
Related Projects
================

* `Orthogonal Functions <https://ameli.github.io/Orthogonal-Functions/>`_: A python package that generates set of orthogonal basis functions used for :ref:`interpolation schemes <InterpolateTraceOfInverse>` in TraceInv.
* `Gaussian Process Regression <https://github.com/ameli/gaussian-process-param-estimation>`_: A python package that makes use of TraceInv expensively to efficiently compute the parameters of Gaussian process regression.

===========
Attribution
===========

If you make use of this package, please consider citing the following manuscript.

.. [Ameli-2020] Ameli, S., and Shadden. S. C. (2020). Interpolating the Trace of the Inverse of Matrix :math:`\mathbf{A} + t \mathbf{B}`. `arXiv:2009.07385 <https://arxiv.org/abs/2009.07385>`__ [math.NA]

::

    @misc{AMELI-2020,
        title={Interpolating the Trace of the Inverse of Matrix $\mathbf{A} + t \mathbf{B}$},
        author={Siavash Ameli and Shawn C. Shadden},
        year={2020},
        month = sep,
        eid = {arXiv:2009.07385},
        eprint={2009.07385},
        archivePrefix={arXiv},
        primaryClass={math.NA},
        howpublished={\emph{arXiv}: 2009.07385 [math.NA]},
    }

================
Acknowledgements
================

* National Science Foundation #1520825
* American Heart Association #18EIA33900046

======
Credit
======

* Some of the algorithms are build on python packages `numpy <https://numpy.org/>`_, `scipy <https://www.scipy.org/>`_, `ray <https://github.com/ray-project/ray>`_, and `scikit-sparse <https://github.com/scikit-sparse/scikit-sparse>`_.

==================
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. .. autosummary::
..
..    TraceInv.GenerateMatrix
..    TraceInv.ComputeTraceOfInverse
..    TraceInv.InterpolateTraceOfInverse

.. |travis-devel| image:: https://img.shields.io/travis/com/ameli/TraceInv
   :target: https://travis-ci.com/github/ameli/TraceInv
.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/TraceInv
   :target: https://codecov.io/gh/ameli/TraceInv
.. |docs| image:: https://github.com/ameli/TraceInv/workflows/deploy-docs/badge.svg
   :target: https://github.com/ameli/TraceInv/actions?query=workflow%3Adeploy-docs
.. |licence| image:: https://img.shields.io/github/license/ameli/TraceInv
   :target: https://opensource.org/licenses/MIT
.. |travis-devel-linux| image:: https://img.shields.io/travis/com/ameli/TraceInv?env=BADGE=linux&label=build&branch=main
   :target: https://travis-ci.com/github/ameli/TraceInv
.. |travis-devel-osx| image:: https://img.shields.io/travis/com/ameli/TraceInv?env=BADGE=osx&label=build&branch=main
   :target: https://travis-ci.com/github/ameli/TraceInv
.. |travis-devel-windows| image:: https://img.shields.io/travis/com/ameli/TraceInv?env=BADGE=windows&label=build&branch=main
   :target: https://travis-ci.com/github/ameli/TraceInv
.. |implementation| image:: https://img.shields.io/pypi/implementation/TraceInv
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/TraceInv
.. |format| image:: https://img.shields.io/pypi/format/TraceInv
.. |pypi| image:: https://img.shields.io/pypi/v/TraceInv
.. |conda| image:: https://anaconda.org/s-ameli/traceinv/badges/installer/conda.svg
   :target: https://anaconda.org/s-ameli/traceinv
.. |platforms| image:: https://img.shields.io/conda/pn/s-ameli/traceinv?color=orange?label=platforms
   :target: https://anaconda.org/s-ameli/traceinv
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/traceinv
   :target: https://anaconda.org/s-ameli/traceinv
.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/ameli/TraceInv/HEAD?filepath=notebooks%2FInterpolateTraceOfInverse.ipynb
