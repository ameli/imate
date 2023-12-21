******
|logo|
******

``imate``, short for **I**\ mplicit **Ma**\ trix **T**\ race **E**\ stimator, is a modular and high-performance C++/CUDA library distributed as a Python package that provides scalable randomized algorithms for the computationally expensive matrix functions in machine learning.

Links
=====

* `Documentation <https://ameli.github.io/imate>`__
* `PyPI <https://pypi.org/project/imate/>`__
* `Anaconda <https://anaconda.org/s-ameli/imate>`__
* `Docker Hub <https://hub.docker.com/r/sameli/imate>`__
* `Github <https://github.com/ameli/imate>`__

Install
=======

Install with ``pip``
--------------------

|pypi|

::

    pip install imate

Install with ``conda``
----------------------

|conda-version|

::

    conda install -c s-ameli imate

Docker Image
------------

|docker-pull| |deploy-docker|

::

    docker pull sameli/imate

Supported Platforms
===================

Successful installation and tests performed on the following operating systems, architectures, and Python and `PyPy <https://www.pypy.org/>`__ versions:

.. |y| unicode:: U+2714
.. |n| unicode:: U+2716

+----------+-----------------+--------------+-------+-------+-------+-------+-------+-------+-------+-----------------+
| Platform | Arch            | Device       | Python Version                | PyPy Version :sup:`1` | Continuous      |
+          |                 +              +-------+-------+-------+-------+-------+-------+-------+ Integration     +
|          |                 |              |  3.9  |  3.10 |  3.11 |  3.12 |  3.8  |  3.9  |  3.10 |                 |
+==========+=================+==============+=======+=======+=======+=======+=======+=======+=======+=================+
| Linux    | X86-64          | CPU          |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  | |build-linux|   |
+          +                 +--------------+-------+-------+-------+-------+-------+-------+-------+                 +
|          |                 | GPU          |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |                 |
+          +-----------------+--------------+-------+-------+-------+-------+-------+-------+-------+                 +
|          | AARCH-64        | CPU          |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |                 |
+          +                 +--------------+-------+-------+-------+-------+-------+-------+-------+                 +
|          |                 | GPU          |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |                 |
+----------+-----------------+--------------+-------+-------+-------+-------+-------+-------+-------+-----------------+
| macOS    | X86-64          | CPU          |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  | |build-macos|   |
+          +                 +--------------+-------+-------+-------+-------+-------+-------+-------+                 +
|          |                 | GPU :sup:`2` |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |                 |
+          +-----------------+--------------+-------+-------+-------+-------+-------+-------+-------+                 +
|          | ARM-64          | CPU          |  |y|  |  |y|  |  |y|  |  |y|  |  |n|  |  |y|  |  |y|  |                 |
+          +                 +--------------+-------+-------+-------+-------+-------+-------+-------+                 +
|          |                 | GPU :sup:`2` |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |                 |
+----------+-----------------+--------------+-------+-------+-------+-------+-------+-------+-------+-----------------+
| Windows  | X86-64          | CPU          |  |y|  |  |y|  |  |y|  |  |y|  |  |n|  |  |n|  |  |n|  | |build-windows| |
+          +                 +--------------+-------+-------+-------+-------+-------+-------+-------+                 +
|          |                 | GPU          |  |y|  |  |y|  |  |y|  |  |y|  |  |n|  |  |n|  |  |n|  |                 |
+----------+-----------------+--------------+-------+-------+-------+-------+-------+-------+-------+-----------------+

.. |build-linux| image:: https://img.shields.io/github/actions/workflow/status/ameli/imate/build-linux.yml
   :target: https://github.com/ameli/imate/actions?query=workflow%3Abuild-linux 
.. |build-macos| image:: https://img.shields.io/github/actions/workflow/status/ameli/imate/build-macos.yml
   :target: https://github.com/ameli/imate/actions?query=workflow%3Abuild-macos
.. |build-windows| image:: https://img.shields.io/github/actions/workflow/status/ameli/imate/build-windows.yml
   :target: https://github.com/ameli/imate/actions?query=workflow%3Abuild-windows

Python wheels for ``imate`` for all supported platforms and versions in the above are available through `PyPI <https://pypi.org/project/imate/>`__ and `Anaconda Cloud <https://anaconda.org/s-ameli/imate>`__. If you need ``imate`` on other platforms, architectures, and Python or PyPy versions, `raise an issue <https://github.com/ameli/imate/issues>`__ on GitHub and we build its Python Wheel for you.

.. line-block::

    :sup:`1. Our wheels for PyPy are exclusively available through pip and cannot be installed using conda.`
    :sup:`2. MacOS does not natively support NVIDIA GPUs.`

Supported GPU Architectures
===========================

``imate`` can run on CUDA-capable **multi**-GPU devices. Using the **docker container** is the easiest way to run ``imate`` on GPU devices. The supported GPU micro-architectures and CUDA version are as follows:

+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| Version \\ Arch | Fermi   | Kepler  | Maxwell | Pascal  | Volta   | Turing  | Ampere  | Hopper |
+=================+=========+=========+=========+=========+=========+=========+=========+========+
| CUDA 9          |   |n|   |   |n|   |   |n|   |   |n|   |   |n|   |   |n|   |   |n|   |   |n|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| CUDA 10         |   |n|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| CUDA 11         |   |n|   |   |n|   |   |n|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| CUDA 12         |   |n|   |   |n|   |   |n|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+

Documentation
=============

|deploy-docs| |binder|

See `documentation <https://ameli.github.io/imate/index.html>`__, including:

* `What This Packages Does? <https://ameli.github.io/imate/overview.html>`__
* `Comprehensive Installation Guide <https://ameli.github.io/imate/tutorials/install.html>`__
* `How to Work with Docker Container? <https://ameli.github.io/imate/tutorials/docker.html>`__
* `How to Deploy on GPU Devices? <https://ameli.github.io/imate/tutorials/gpu.html>`__
* `API Reference <https://ameli.github.io/imate/api.html>`__
* `Interactive Notebook Tutorials <https://mybinder.org/v2/gh/ameli/glearn/HEAD?filepath=notebooks%2Fquick_start.ipynb>`__
* `Publications <https://ameli.github.io/imate/cite.html>`__

Performance
===========

``imate`` is scalable to **very large matrices**. Its core library for basic linear algebraic operations is **faster than OpenBLAS**, and its **pseudo-random generator** is a hundred-fold faster than the implementation in the standard C++ library.

Read about the performance of ``imate`` in practical applications:

* `Performance on GPU Farm <https://ameli.github.io/imate/performance/gpu.html#perf-gpu>`__
* `Comparison of Randomized Algorithms <https://ameli.github.io/imate/performance/algorithms.html>`__
* `Comparison With and Without OpenBLAS <https://ameli.github.io/imate/performance/openblas.html#perf-openblas>`__
* `Interpolation of Affine Matrix Functions <https://ameli.github.io/imate/performance/interpolation.html>`__
    
How to Contribute
=================

We welcome contributions via `GitHub's pull request <https://github.com/ameli/imate/pulls>`__. If you do not feel comfortable modifying the code, we also welcome feature requests and bug reports as `GitHub issues <https://github.com/ameli/imate/issues>`__.

How to Cite
===========

If you publish work that uses ``imate``, please consider citing the manuscripts available `here <https://ameli.github.io/imate/cite.html>`__.

License
=======

|license|

This project uses a `BSD 3-clause license <https://github.com/ameli/imate/blob/main/LICENSE.txt>`__, in hopes that it will be accessible to most projects. If you require a different license, please raise an `issue <https://github.com/ameli/imate/issues>`__ and we will consider a dual license.

.. |logo| image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/icons/logo-imate-light.svg
   :width: 160
.. |license| image:: https://img.shields.io/github/license/ameli/imate
   :target: https://opensource.org/licenses/BSD-3-Clause
.. |deploy-docs| image:: https://img.shields.io/github/actions/workflow/status/ameli/imate/deploy-docs.yml?label=docs
   :target: https://github.com/ameli/imate/actions?query=workflow%3Adeploy-docs
.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/ameli/imate/HEAD?filepath=notebooks%2Fquick_start.ipynb
.. |pypi| image:: https://img.shields.io/pypi/v/imate
   :target: https://pypi.org/project/imate/
.. |deploy-docker| image:: https://img.shields.io/github/actions/workflow/status/ameli/imate/deploy-docker.yml?label=build%20docker
   :target: https://github.com/ameli/imate/actions?query=workflow%3Adeploy-docker
.. |docker-pull| image:: https://img.shields.io/docker/pulls/sameli/imate?color=green&label=downloads
   :target: https://hub.docker.com/r/sameli/imate
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/imate
   :target: https://anaconda.org/s-ameli/imate
