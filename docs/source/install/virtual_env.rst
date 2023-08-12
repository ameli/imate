.. _virtual-env:

Install in Virtual Environments
===============================

If you do not want the installation to occupy your main python's site-packages (either you are testing or the dependencies may clutter your existing installed packages), install the package in an isolated virtual environment. Two common virtual environments are :ref:`virtualenv <virtualenv_env>` and :ref:`conda <conda_env>`.

.. _virtualenv_env:

Install in ``virtualenv`` Environment
-------------------------------------

1. Install ``virtualenv``:

   .. prompt:: bash

       python -m pip install virtualenv

2. Create a virtual environment and give it a name, such as ``imate_env``

   .. prompt:: bash

       python -m virtualenv imate_env

3. Activate python in the new environment

   .. prompt:: bash

       source imate_env/bin/activate

4. Install ``imate`` package with any of the :ref:`above methods <install-wheels>`. For instance:

   .. prompt:: bash

       python -m pip install imate
   
   Then, use the package in this environment.

5. To exit from the environment

   .. prompt:: bash

       deactivate

.. _conda_env:

Install in ``conda`` Environment
--------------------------------

In the followings, it is assumed `anaconda <https://www.anaconda.com/products/individual#Downloads>`_ (or `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_) is installed.

1. Initialize conda

   .. prompt:: bash

       conda init

   You may need to close and reopen your terminal after the above command. Alternatively, instead of the above, you can do

   .. prompt:: bash

       sudo sh $(conda info --root)/etc/profile.d/conda.sh

2. Create a virtual environment and give it a name, such as ``imate_env``

   .. prompt:: bash

       conda create --name imate_env -y

   The command ``conda info --envs`` shows the list of all environments. The current environment is marked by an asterisk in the list, which should be the default environment at this stage. In the next step, we will change the current environment to the one we created.

3. Activate the new environment

   .. prompt:: bash

       source activate imate_env

4. Install ``imate`` with any of the :ref:`above methods <install-wheels>`. For instance:

   .. prompt:: bash

       conda install -c s-ameli imate
   
   Then, use the package in this environment.

5. To exit from the environment

   .. prompt:: bash

       conda deactivate
