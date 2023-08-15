.. _gpu-docker:

Use |project| Docker Container on GPU
=====================================
   
This method neither requires installing CUDA nor |project| as all are pre-installed in a docker image.

Install Docker
--------------

First, `install docker <https://docs.docker.com/engine/install/ubuntu/>`_. Briefly:

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt update
            sudo apt install ca-certificates curl gnupg lsb-release
            sudo mkdir -p /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
                sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
                https://download.docker.com/linux/ubuntu \
                $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            sudo apt update
            sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum install -y yum-utils
            sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
            sudo yum install docker-ce docker-ce-cli containerd.io docker-compose-plugin
            sudo systemctl enable docker.service
            sudo systemctl enable containerd.service
            sudo systemctl start docker

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo yum install -y yum-utils
            sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
            sudo yum install docker-ce docker-ce-cli containerd.io docker-compose-plugin
            sudo systemctl enable docker.service
            sudo systemctl enable containerd.service
            sudo systemctl start docker

Configure docker to run docker `without sudo password <https://docs.docker.com/engine/install/linux-postinstall/>`_ by

.. prompt:: bash

    sudo groupadd docker
    sudo usermod -aG docker $USER

Then, log out and log back. If docker is installed on a *virtual machine*, restart the virtual machine for changes to take effect.

Install NVIDIA Container Toolkit
--------------------------------

To access host's GPU device from a docker container, `install NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ as follows.

Add the package to the repository:

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
            curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
            curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum-config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo

Install `nvidia-contaner-toolkit` by:

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt update
            sudo apt install -y nvidia-container-toolkit

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum install -y https://download.docker.com/linux/centos/7/x86_64/stable/Packages/containerd.io-1.4.3-3.1.el7.x86_64.rpm

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf install -y https://download.docker.com/linux/centos/7/x86_64/stable/Packages/containerd.io-1.4.3-3.1.el7.x86_64.rpm

Restart docker:

.. prompt:: bash

    sudo systemctl restart docker

Get |project| Docker image
--------------------------

|docker-size|

Get the |project| docker image by

.. prompt:: bash

  docker pull sameli/imate

The docker image has the followings pre-installed:

* CUDA: in ``/usr/local/cuda``
* Python 3.9: in ``/usr/bin/python3``
* Python interpreters: `ipython`, `jupyter`
* Editor: `vim`

Use |project| Docker Container on GPU
-------------------------------------

To use host's GPU from the docker container, add  ``--gpus all`` to any of the ``docker run`` commands, such as by

.. prompt:: bash

    docker run --gpus all -it sameli/imate

The followings are some examples of using ``docker run`` with various options:

* To check the host's NVIDIA driver version, CUDA runtime library version, and list of available GPU devices, run ``nvida-smi`` command by:

  .. prompt:: bash
  
      docker run --gpus all sameli/imate nvidia-smi
  
* To run the container and open *Python* interpreter directly at startup:
  
  .. prompt:: bash
  
      docker run -it --gpus all sameli/imate
  
  This also imports |project| package automatically.
  
* To run the container and open *IPython* interpreter directly at startup:
  
  .. prompt:: bash

        docker run -it --gpus all sameli/imate ipython
  
  This also imports `imate` package automatically.
  
* To open *Bash shell* only:
  
  .. prompt:: bash

        docker run -it --gpus all --entrypoint /bin/bash sameli/imate
  
* To *mount* a host's directory, such as ``/home/user/project``, onto a directory of the docker's container, such as ``/root``, use:
  
  .. prompt:: bash
  
        docker run -it --gpus all -v /home/user/project:/root sameli/imate

.. |docker-size| image:: https://img.shields.io/docker/image-size/sameli/imate
   :target: https://hub.docker.com/r/sameli/imate
