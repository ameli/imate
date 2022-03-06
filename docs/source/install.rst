*******
Install
*******

===================
Python Dependencies
===================

The runtime dependencies are:

* **Required:** ``numpy`` and ``scipy``.
* **Required:** ``matplotlib`` and ``seaborn``, but only required to run the :ref:`Examples`.
* **Optional:** ``ray`` and ``scikit-sparse`` can improve performance, but not required.

.. note::
    By installing imate :ref:`below <InstallationMethods>`, the *required* dependencies (but not the *optional* packages) will be installed automatically and no other action is needed. However, if desired, the *optional* packages should be installed :ref:`manually <Install Optional Packages>`.

.. _InstallationMethods:

================
Install imate
================

The package can be installed on Linux, macOS, and Windows platforms and supports both python 2 and 3. Install by either of the following ways, namely, through :ref:`PyPi <Install_PyPi>`, :ref:`Conda <Install_Conda>`, or :ref:`build locally <Build_Locally>`.

.. _Install_PyPi:

-----------------
Install from PyPi
-----------------

|pypi| |format| |implementation| |pyversions|

The recommended way to install imate and its dependencies is through the package available at `PyPi <https://pypi.org/project/imate>`_ by

::
      
    python -m pip install imate

.. _Install_Conda:

---------------------------
Install from Anaconda Cloud
---------------------------

|conda| |conda-version| |conda-platform|

Install imate and its dependencies through the package available at `Conda <https://anaconda.org/s-ameli/traceinv>`_ by

::

    conda install -c s-ameli traceinv

.. _Build_Locally:

--------------------
Build Source Locally
--------------------

|release|

1. Install the build dependencies ``setuptools``, ``cython``, and ``numpy``:

   ::
         
       python -m pip install setuptools 
       python -m pip install cython
       python -m pip install numpy>1.11

2. Clone the source code
   
   ::
       
       git clone https://github.com/ameli/imate.git
       cd imate

3. Build the package locally

   ::
       
       python setup build

4. Install the package

   ::
       
       python setup install

   The above command may need to be run with ``sudo``.

5. Install the runtime dependencies. In the same root directory of the package (where the file ``requirements.txt`` can be found) run
   
   ::
       
       python -m pip install -r requirements.txt

   Alternatively, the dependencies can be installed with ``conda`` by
   
   ::
       
       conda install --file requirements.txt

==============================
Install in Virtual Environment
==============================

If you do not want the installation to occupy your main python's site-packages (either you are testing or the dependencies may clutter your existing installed packages), you may install the package in an isolated virtual environment. Below, we describe the installation procedure in two common virtual environments, namely, :ref:`virtualenv <virtualenv_env>` and :ref:`conda <conda_env>`.

.. _virtualenv_env:

-------------------------------------
Install in ``virtualenv`` Environment
-------------------------------------

1. Install ``virtualenv``:

   ::

       python -m pip install virtualenv

2. Create a virtual environment and give it a name, such as ``imate_env``

   ::

       python -m virtualenv imate_env

3. Activate python in the new environment

   ::

       source imate_env/bin/activate

4. Install ``imate`` package with any of the :ref:`above methods <InstallationMethods>`. For instance:

   ::

       python -m pip install imate
   
   Then, use the package in this environment.

5. To exit from the environment

   ::

       deactivate

.. _conda_env:

--------------------------------
Install in ``conda`` Environment
--------------------------------

In the followings, it is assumed `anaconda <https://www.anaconda.com/products/individual#Downloads>`_ (or `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_) is installed.

1. Initialize conda

   ::

       conda init

   You may need to close and reopen terminal after the above command. Alternatively, instead of the above, you can do

   ::

       sudo sh $(conda info --root)/etc/profile.d/conda.sh

2. Create a virtual environment and give it a name, such as ``imate_env``

   ::

       conda create --name imate_env -y

   The command ``conda info --envs`` shows the list of all environments. The current environment is marked by an asterisk in the list, which should be the default environment at this stage. In the next step, we will change the current environment to the one we created.

3. Activate the new environment

   ::

       source activate imate_env

4. Install ``imate`` with any of the :ref:`above methods <InstallationMethods>`. For instance:

   ::

       conda install -c s-ameli traceinv
   
   Then, use the package in this environment.

5. To exit from the environment

   ::

       conda deactivate

=========================
Install Optional Packages
=========================

Installing the optional packages below can improve the performance for some of the functionalities, but not necessary. 

.. _InstallRay:

---------------
Install ``ray``
---------------

::

    python -m pip install ray

When ``ray`` is needed:
    To generate large sparse matrices with :mod:`imate.GeneratreMatrix` module (particularly to run :ref:`examples <Examples>`), you may install the ``ray`` package to leverage the parallel processing. However, the code and examples can be run without installing ``ray``.


.. _InstallScikitSparse:

-------------------------
Install ``scikit-sparse``
-------------------------

1. Install `Suite Sarse <https://people.engr.tamu.edu/davis/suitesparse.html>`_ development library.
   
   * **In Linux:** Install ``libsuitesparse-dev`` using ``apt`` package manager in Debian-based Linux distros (such as *Debian, Ubuntu, Mint*) by
   
     ::
         
         sudo apt install libsuitesparse-dev  

   Replace ``apt`` in the above with the native package manager of your operating system, such as ``yum`` for  *Redhat, Fedora, and CentOS Linux*, ``pacman`` for *Arch Linux*.
   
   * **In MacOS:** To install ``suite-sparse`` with ``brew``:

     ::
         
         sudo brew install suite-sparse


   * **Using Anaconda:** Alternatively, if you are using *Anaconda* python distribution (on either of the operating systems), install Suite Sparse by:

   ::

       sudo conda install -c conda-forge suitesparse

2. Install ``scikit-sparse`` python package:

   ::
       
       python -m pip install scikit-sparse

When ``scikit-sparse`` is needed:
    In ``imate`` package, one of the methods to compute the trace of a matrix is by the *Cholesky decomposition*. If the input matrix is *sparse*, the Cholesky decomposition is computed using ``scikit-sparse``. But if this package is not installed, the ``scipy`` package is used instead.

.. |implementation| image:: https://img.shields.io/pypi/implementation/imate
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/imate
.. |format| image:: https://img.shields.io/pypi/format/imate
.. |pypi| image:: https://img.shields.io/pypi/v/imate
.. |conda| image:: https://anaconda.org/s-ameli/traceinv/badges/installer/conda.svg
   :target: https://anaconda.org/s-ameli/traceinv
.. |platforms| image:: https://img.shields.io/conda/pn/s-ameli/traceinv?color=orange?label=platforms
   :target: https://anaconda.org/s-ameli/traceinv
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/traceinv
   :target: https://anaconda.org/s-ameli/traceinv
.. |release| image:: https://img.shields.io/github/v/tag/ameli/imate
   :target: https://github.com/ameli/imate/releases/
.. |conda-platform| image:: https://anaconda.org/s-ameli/traceinv/badges/platforms.svg
   :target: https://anaconda.org/s-ameli/traceinv
