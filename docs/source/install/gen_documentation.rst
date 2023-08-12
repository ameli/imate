.. _gen-documentation:

Generate Documentation
======================

The documentation consists of a Doxygen documentation and a Sphinx documentation. You should generate the Doxygen documentation first. Before generating the Sphinx documentation, you should compile the package.

Get the source code from the GitHub repository.

.. prompt:: bash

    git clone https://github.com/ameli/imate.git
    cd imate

If you already had the source code, clean it from any previous build (especially if you built `in-source`):

.. prompt:: bash

    python setup.py clean

Generate Doxygen Documentation
------------------------------

Install `doxygen` and `graphviz` by

.. tab-set::

   .. tab-item:: Ubuntu/Debian
      :sync: ubuntu

      .. prompt:: bash

            sudo apt install doxygen graphviz -y

   .. tab-item:: CentOS 7
      :sync: centos

      .. prompt:: bash

          sudo yum install doxygen graphviz -y

   .. tab-item:: RHEL 9
      :sync: rhel

      .. prompt:: bash

          sudo dnf install doxygen graphviz -y

   .. tab-item:: macOS
      :sync: osx

      .. prompt:: bash

          sudo brew install doxygen graphviz -y

   .. tab-item:: Windows (Powershell)
      :sync: win

      .. prompt:: powershell

          scoop install doxygen graphviz

.. attention::

    Make sure you ran ``python setup.py clean`` as mentioned previously. Otherwise, if the source directory is not cleaned from any previous built (if there is any), Doxygen unwantedly generates all auto-generated cython files.

Generating the Doxygen documentation by

.. prompt:: bash

    cd docs/doxygen
    doxygen doxyfile.in
    cd ../..

Compile Package
---------------

Set ``CYTHON_BUILD_FOR_DOC`` to `1` (see :ref:`Configure Compile-Time Environment variables <config-env-variables>`). Compile and install the package by

.. tab-set::

    .. tab-item:: UNIX
        :sync: unix

        .. prompt:: bash

            export CYTHON_BUILD_FOR_DOC=1
            export USE_CUDA=0
            sudo -E python setup.py install

    .. tab-item:: Windows (Powershell)
        :sync: win

        .. prompt:: powershell

            $env:export CYTHON_BUILD_FOR_DOC = "1"
            $env:export USE_CUDA = "0"
            sudo -E python setup.py install

Generate Sphinx Documentation
-----------------------------

Install `Pandoc <https://pandoc.org/>`_ by

.. tab-set::

   .. tab-item:: Ubuntu/Debian
      :sync: ubuntu

      .. prompt:: bash

            sudo apt install pandoc -y

   .. tab-item:: CentOS 7
      :sync: centos

      .. prompt:: bash

          sudo yum install pandoc -y

   .. tab-item:: RHEL 9
      :sync: rhel

      .. prompt:: bash

          sudo dnf install pandoc -y

   .. tab-item:: macOS
      :sync: osx

      .. prompt:: bash

          sudo brew install pandoc -y

   .. tab-item:: Windows (Powershell)
      :sync: win

      .. prompt:: powershell

          scoop install pandoc

Install the requirements for the Sphinx documentation by

.. prompt:: bash

    python -m pip install -r docs/requirements.txt

The above command installs the required packages in Python's path directory. Make sure python's directory is on the `PATH`, for instance, by

.. tab-set::

    .. tab-item:: UNIX
        :sync: unix

        .. prompt:: bash

            PYTHON_PATH=`python -c "import os, sys; print(os.path.dirname(sys.executable))"`
            export PATH=${PYTHON_PATH}:$PATH

    .. tab-item:: Windows (Powershell)
        :sync: win

        .. prompt:: powershell

            $PYTHON_PATH = (python -c "import os, sys; print(os.path.dirname(sys.executable))")
            $env:Path += ";$PYTHON_PATH"

Now, build the documentation:

.. tab-set::

    .. tab-item:: UNIX
        :sync: unix

        .. prompt:: bash

            make clean html --directory=docs

    .. tab-item:: Windows (Powershell)
        :sync: win

        .. prompt:: powershell

            cd docs
            make.bat clean html

The main page of the documentation can be found in ``/docs/build/html/index.html``. 
