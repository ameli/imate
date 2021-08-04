# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE

# -----------------------------------------------------------------------------
# How to build
#   $ docker build -t sameli/imate -f <This-Filename> .
#
# How to run:
#   $ docker run -it -v/host_dir:/image_dir imate
# -----------------------------------------------------------------------------

# -----
# Build
# -----

FROM nvidia/cuda:11.4.1-devel-ubuntu20.04 as builder
MAINTAINER Siavash Ameli <samei@berkeley.edu>
LABEL Description="imate python package with cuda-11 support"

# Install python3
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build package wheel
ENV USE_CUDA=1
RUN mkdir /tmp/imate
ADD . /tmp/imate/
WORKDIR /tmp/imate
RUN /usr/bin/python3 setup.py bdist_wheel

# -------
# Runtime
# -------

FROM nvidia/cuda:11.4.1-base-ubuntu20.04

# Install python3 and libcudablas, libcusparse, since they are not in the NVIDA
# cuda's "base" image.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        libcublas-11-4 \
        libcusparse-11-4 \
        python3-pip \
        libgomp1 \
        libsuitesparse-dev \
        vim \
        texlive-latex-recommended \
        cm-super \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the wheel from the previous build (on NVIDIA cuda-devel image) and
# install it on this image.
COPY --from=builder /tmp/imate/dist/*.whl /tmp/
RUN /usr/bin/python3 -m pip --no-cache-dir install /tmp/*.whl
RUN rm /tmp/*.whl

# Create a startup file to auto-import package when python starts
RUN echo "import imate" > /home/.start_python.py
ENV PYTHONSTARTUP=/home/.start_python.py
ENV PATH=$PATH:/usr/bin/
RUN /usr/bin/python3 -m pip install --upgrade pip ipython jupyter

# Change working directory for entrypoint
VOLUME ["/root"]
WORKDIR /root

# ENTRYPOINT ["/bin/bash"]
CMD ["/usr/bin/python3"]
