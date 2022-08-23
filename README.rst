******
|logo|
******

|licence| |codecov-devel|

A python package to compute the trace of the inverse of a matrix or a linear matrix function.

.. For users
..     * `Documentation <https://ameli.github.io/imate/index.html>`_
..     * `PyPi package <https://pypi.org/project/imate/>`_
..     * `Source code <https://github.com/ameli/imate>`_
..
.. For developers
..     * `API <https://ameli.github.io/imate/_modules/modules.html>`_
..     * `Travis-CI <https://travis-ci.com/github/ameli/imate>`_
..     * `Codecov <https://codecov.io/gh/ameli/imate>`_

+---------------------------------------------------------------+----------------------------------------------------------------+
|    For users                                                  | For developers                                                 |
+===============================================================+================================================================+
| * `Documentation <https://ameli.github.io/imate/index.html>`_ | * `API <https://ameli.github.io/imate/_modules/modules.html>`_ |
| * `PyPi package <https://pypi.org/project/imate/>`_           | * `Travis-CI <https://travis-ci.com/github/ameli/imate>`_      |
| * `Anaconda Cloud <https://anaconda.org/s-ameli/traceinv>`_   | * `Codecov <https://codecov.io/gh/ameli/imate>`_               |
+---------------------------------------------------------------+----------------------------------------------------------------+

***********
Description
***********

This package computes the trace of inverse of two forms of matrices:

1. **Fixed Matrix:** For an invertible matrix |image01| (sparse of dense), this package computes |image02|.
2. **One-Parameter Affine Matrix Function:** |image05|, where |image01| and |image03| are symmetric and positive-definite matrices and ``t`` is a real parameter. This package can interpolate the function

.. image:: https://raw.githubusercontent.com/ameli/imate/main/docs/images/image06.svg
   :align: center

**Application:**
    The above function is featured in a wide range of applications in statistics and machine learning. Particular applications are in model selection and optimizing hyperparameters with gradient-based maximum likelihood methods. In such applications, computing the above function is often a computational challenge for large matrices. Often, this function is evaluated for a wide range of the parameter |image00| while |image01| and |image03| remain fixed. As such, an interpolation scheme enables fast computation of the function.

These interpolation methods are described in [Ameli-2020]_. 

.. |image00| image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/image00.svg
.. |image01| image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/image01.svg
.. |image02| image:: https://raw.githubusercontent.com/ameli/imate/main/docs/imagessource/_static/images/image02.svg
.. |image03| image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/image03.svg
.. |image04| image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/image04.svg
.. |image05| image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/image05.svg
.. |image06| image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/image06.svg

*******
Install
*******

|format| |pypi| |implementation| |pyversions|

* Install through `PyPi <https://pypi.org/project/imate>`_:

  ::

    pip install imate

* Install through `Anaconda <https://anaconda.org/s-ameli/imate>`_:

  ::

    conda install -c s-ameli imate


* Download the source code, compile, and install by:

  ::

    git clone https://github.com/ameli/imate.git
    cd imate
    pip install -e .

********
Citation
********

.. [Ameli-2020] Ameli, S., and Shadden. S. C. (2020). Interpolating the Trace of the Inverse of Matrix **A** + t **B**. `arXiv:2009.07385 <https://arxiv.org/abs/2009.07385>`__ [math.NA]

****************
Acknowledgements
****************

* National Science Foundation #1520825
* American Heart Association #18EIA33900046

.. |logo| image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/logo-imate-light.svg
   :width: 160
.. |examplesdir| replace:: ``/examples`` 
.. _examplesdir: https://github.com/ameli/imate/blob/main/examples
.. |example1| replace:: ``/examples/Plot_imate_FullRank.py``
.. _example1: https://github.com/ameli/imate/blob/main/examples/Plot_imate_FullRank.py
.. |example2| replace:: ``/examples/Plot_imate_IllConditioned.py``
.. _example2: https://github.com/ameli/imate/blob/main/examples/Plot_imate_IllConditioned.py
.. |example3| replace:: ``/examples/Plot_GeneralizedCorssValidation.py``
.. _example3: https://github.com/ameli/imate/blob/main/examples/Plot_GeneralizedCrossValidation.py

.. |travis-devel| image:: https://img.shields.io/travis/com/ameli/imate
   :target: https://travis-ci.com/github/ameli/imate
.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/imate
   :target: https://codecov.io/gh/ameli/imate
.. |licence| image:: https://img.shields.io/github/license/ameli/imate
   :target: https://opensource.org/licenses/BSD-3-Clause
.. |travis-devel-linux| image:: https://img.shields.io/travis/com/ameli/imate?env=BADGE=linux&label=build&branch=main
   :target: https://travis-ci.com/github/ameli/imate
.. |travis-devel-osx| image:: https://img.shields.io/travis/com/ameli/imate?env=BADGE=osx&label=build&branch=main
   :target: https://travis-ci.com/github/ameli/imate
.. |travis-devel-windows| image:: https://img.shields.io/travis/com/ameli/imate?env=BADGE=windows&label=build&branch=main
   :target: https://travis-ci.com/github/ameli/imate
.. |implementation| image:: https://img.shields.io/pypi/implementation/imate
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/imate
.. |format| image:: https://img.shields.io/pypi/format/imate
.. |pypi| image:: https://img.shields.io/pypi/v/imate
