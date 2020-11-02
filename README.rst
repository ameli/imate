|travis-devel| |codecov-devel| |licence| |format| |pypi| |implementation| |pyversions|

TraceInv
========

A python package to compute the trace of the inverse of a matrix or a linear matrix function.

For users
    * `PyPi package <https://pypi.org/project/TraceInv/>`_
    * `Source code <https://github.com/ameli/TraceInv>`_
    * `Documentation <https://ameli.github.io/TraceInv/index.html>`_

For developers
    * `API <https://ameli.github.io/TraceInv/_modules/modules.html>`_
    * `Travis-CI <https://travis-ci.com/github/ameli/TraceInv>`_
    * `Codecov <https://codecov.io/gh/ameli/TraceInv>`_

.. Status
.. ------
..
.. +------------+--------------------------+
.. | Platform   | CI Status                |
.. +============+==========================+
.. | Linux      | |travis-devel-linux|     |
.. +------------+--------------------------+
.. | OSX        | |travis-devel-osx|       |
.. +------------+--------------------------+
.. | Windows    | |travis-devel-windows|   |
.. +------------+--------------------------+

Description
-----------

This package computes the trace of inverse of matrices for two purposes:

Fixed Matrix
~~~~~~~~~~~~

For a given generic invertible matrix :math:`\mathbf{A}`, this package can compute :math:`\mathrm{trace}(\mathbf{A}^{-1})` by either of these three methods:

1. *Cholesky method*: This results the exact computation of the trace.
2. *Hutchinson's method*: This is a randomized approximation and suitable for large or implicit matrices.
3. *Stochastic Lanczos Quadrature Method*: This is a randomized approximation and suitable for large or implicit matrices.

Linear Matrix Function
~~~~~~~~~~~~~~~~~~~~~~

Consider two matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` and a range of real number :math:`t \in [t_0,t_1]` such that :math:`\mathbf{A} + t \mathbf{B}` is invertible. Then, this package can interpolate the function

.. math::

    t \mapsto \mathrm{trace}\left((\mathbf{A} + t \mathbf{B})^{-1} \right)

by the method described in [Ameli-2020]_. The above function is featured in a wide range of applications in statistics and machine learning, particularly, in model selection and optimizing hyperparameters with gradient-based maximum likelihood methods.


Install
-------

Install Prerequisits
~~~~~~~~~~~~~~~~~~~~
    
(*Optional*) If you use sparse matrices, the `*Suite Sarse* <https://people.engr.tamu.edu/davis/suitesparse.html>`_ package should be installed. Depending on the operating system, install Suite Sparse as follows.

* In Linux, install ``libsuitesparse-dev`` package. 

  * Install by using ``apt`` (in Debian, Ubuntu, Mint)

  ::

      sudo apt install libsuitesparse-dev  

  * Or, install by using ``yum`` (in Redhat, Fedora)

    ::

      sudo yum install libsuitesparse-dev  

  * or, install by using ``pacman`` (in Arch Linux)

    ::

      sudo pacman -S install libsuitesparse-dev  

* In MacOSX, install ``libsuitesparse-dev`` package, for instance by ``brew``:

::

    sudo brew install libsuitesparse-dev

* Alternatively, if you are using Anaconda for python distribution (on either of the operating systems), install Suite Sparse by:

::

    sudo conda install -c conda-forge suitesparse

Install this package
~~~~~~~~~~~~~~~~~~~~

- Method 1: install from the package available at `PyPi <https://pypi.org/project/TraceInv>`_:

  ::

    python -m pip install TraceInv

- Method 2: install directly from the source code by:

  ::

    git clone https://github.com/ameli/TraceInv.git
    cd TraceInv
    python -m pip install -e .

Usage
-----

.. code-block:: python

    from TraceInv import GenerateMatrix
    from TraceInv import ComputeTraceOfInverse
    
    # Generate a symmetric positive-definite matrix
    A = GenerateMatrix(NumPoints=20)

    # Compute trace of inverse
    trace = ComputeTraceOfInverse(A,method='cholesky')


Optional arguments
~~~~~~~~~~~~~~~~~~

When the *standalone application* (the second method in the above) is called, the executable accepts some optional arguments as follows.

+--------------------------------------+------------------------------------------------------------------------------------------+
| Option                               | Description                                                                              |
+======================================+==========================================================================================+
| ``-h``, ``--help``                   | Prints a help message.                                                                   |
+--------------------------------------+------------------------------------------------------------------------------------------+
| ``-v``, ``--version``                | Prints version.                                                                          |
+--------------------------------------+------------------------------------------------------------------------------------------+
| ``-l``, ``--license``                | Prints author info, citation and license.                                                |
+--------------------------------------+------------------------------------------------------------------------------------------+
| ``-n``, ``--num-func[=int]``         | Number of orthogonal functions to generate. Positive integer. Default is 9.              |
+--------------------------------------+------------------------------------------------------------------------------------------+
| ``-s``, ``--start-func[=int]``       | Starting function index. Non-negative integer. Default is 1.                             |
+--------------------------------------+------------------------------------------------------------------------------------------+
| ``-e``, ``--end-interval[=float]``   | End of the interval of functions domains. A real number greater than zero. Default is 1. |
+--------------------------------------+------------------------------------------------------------------------------------------+
| ``-c``,\ ``--check``                 | Checks orthogonality of generated functions.                                             |
+--------------------------------------+------------------------------------------------------------------------------------------+
| ``-p``, ``--plot``                   | Plots generated functions, also saves the plot as pdf file in the current directory.     |
+--------------------------------------+------------------------------------------------------------------------------------------+

Parameters
~~~~~~~~~~

The variables |image17|, |image18|, and |image19| can be set in the script by the following arguments,

+-------------+--------------------------+---------------------------------+
| Variable    | Variable in script       | Option                          |
+=============+==========================+=================================+
| |image23|   | ``StartFunctionIndex``   | ``-s``, or ``--start-func``     |
+-------------+--------------------------+---------------------------------+
| |image24|   | ``NumFunctions``         | ``-n``, or ``--num-func``       |
+-------------+--------------------------+---------------------------------+
| |image25|   | ``EndInterval``          | ``-e``, or ``--end-interval``   |
+-------------+--------------------------+---------------------------------+

Citation
--------

.. [Ameli-2020] Ameli, S., and Shadden. S. C. (2020). Interpolating the Trace of the Inverse of Matrix **A** + t **B**. `arXiv:2009.07385 <https://arxiv.org/abs/2009.07385>`__ [math.NA]

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

.. |travis-devel| image:: https://img.shields.io/travis/com/ameli/TraceInv
   :target: https://travis-ci.com/github/ameli/TraceInv
.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/TraceInv
   :target: https://codecov.io/gh/ameli/TraceInv
.. |licence| image:: https://img.shields.io/github/license/ameli/TraceInv
   :target: https://opensource.org/licenses/MIT
.. |travis-devel-linux| image:: https://img.shields.io/travis/com/ameli/TraceInv?env=BADGE=linux&label=build&branch=master
   :target: https://travis-ci.com/github/ameli/TraceInv
.. |travis-devel-osx| image:: https://img.shields.io/travis/com/ameli/TraceInv?env=BADGE=osx&label=build&branch=master
   :target: https://travis-ci.com/github/ameli/TraceInv
.. |travis-devel-windows| image:: https://img.shields.io/travis/com/ameli/TraceInv?env=BADGE=windows&label=build&branch=master
   :target: https://travis-ci.com/github/ameli/TraceInv
.. |implementation| image:: https://img.shields.io/pypi/implementation/TraceInv
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/TraceInv
.. |format| image:: https://img.shields.io/pypi/format/TraceInv
.. |pypi| image:: https://img.shields.io/pypi/v/TraceInv

.. |image09| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/phi_i_perp.svg
.. |image10| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/phi_i.svg
.. |image11| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/phi_i_perp.svg
.. |image12| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/phi_i.svg
.. |image13| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/phi_i_perp.svg
.. |image14| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/interval.svg
.. |image15| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/w.svg
.. |image16| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/delta.svg
.. |image17| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/i_0.svg
.. |image18| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/n.svg
.. |image19| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/L.svg
.. |image20| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/i_0.svg
.. |image21| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/n.svg
.. |image22| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/L.svg
.. |image23| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/i_0.svg
.. |image24| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/n.svg
.. |image25| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/L.svg
.. |image26| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/alpha_i.svg
.. |image27| image:: https://raw.githubusercontent.com/ameli/TraceInv/master/docs/images/a_ij.svg
