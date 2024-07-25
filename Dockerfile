FROM ubuntu:bionic

RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata

RUN apt-get update -y && apt-get install -y \
    apt-utils \
    python3.7 python3.7-dev python-pip python3-pip python-tk \
    build-essential \
    dialog \
    git \
    lsb-release mesa-utils \
    software-properties-common locales x11-apps \
    gedit gedit-plugins nano \
    screen tree \
    sudo ssh synaptic \
    wget curl unzip htop \
    gdb valgrind \
    libcanberra-gtk* \
    xsltproc \
    libgtest-dev \
    iputils-ping iproute2 \
    vim \
    libosmesa6-dev \
    libspatialindex-dev \
    libusb-1.0-0-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Get latest cmake
RUN apt purge --auto-remove cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt update && \
    apt install -y kitware-archive-keyring && \
    rm /etc/apt/trusted.gpg.d/kitware.gpg && \
    apt update && \
    apt install -y cmake

# CUDA
RUN apt-get update && apt-get install -y --no-install-recommends \
	gnupg2 \
	curl \
	ca-certificates && \
	curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
	echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
	echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
	cuda-cudart-10-1=10.1.243-1 \
	cuda-compat-10-1 && \
	ln -s cuda-10.1 /usr/local/cuda

RUN apt-get install -y --no-install-recommends \
	cuda-libraries-10-1=10.1.243-1 \
	cuda-nvtx-10-1=10.1.243-1 \
	cuda-nvml-dev-10-1=10.1.243-1 \
	cuda-command-line-tools-10-1=10.1.243-1 \
	cuda-minimal-build-10-1=10.1.243-1 \
	libnccl2=2.4.8-1+cuda10.1 \
	libcublas10=10.2.1.243-1 \
	libcudnn7=7.6.5.32-1+cuda10.1 && \
	apt-mark hold libcudnn7 && \
	apt-mark hold libnccl2 && \
	rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV CUDA_VERSION 10.1.243
ENV CUDA_PKG_VERSION 10-1=10.1.243-1
ENV NCCL_VERSION 2.4.8
ENV CUDNN_VERSION 7.6.5.32
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}
RUN echo "alias python=python3.7" >> ~/.bashrc
ENV PYTHONPATH=/usr/bin/python3.7
RUN python3.7 -m pip install --upgrade --force pip

# Build Open3D headless
RUN cd /home && git clone https://github.com/intel-isl/Open3D
RUN pip install -U \
    setuptools \
    wheel \
    numpy \
    matplotlib \
    cython

RUN cd /home/Open3D && \
    git submodule update --init --recursive
RUN cd /home/Open3D && \
    sed -i "s/PythonExecutable REQUIRED/PythonExecutable 3.7 EXACT REQUIRED/g" CMakeLists.txt
RUN rm -r /usr/bin/python && ln -s /usr/bin/python3.7 /usr/bin/python
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv DE19EB17684BA42D && \
    apt-get update -y && apt-get install -y \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN cd /home/Open3D && mkdir build && cd build && \
    cmake  -DENABLE_HEADLESS_RENDERING=ON -DBUILD_GUI=OFF -DUSE_SYSTEM_GLEW=OFF -DUSE_SYSTEM_GLFW=OFF .. && \
    make -j12 && \
    make install-pip-package

ARG ssh_prv_key
ARG ssh_pub_key

# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts

# Add the keys and set permissions
RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
    echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub

# Clone RobEx
RUN cd /home && git clone git@github.com:ihaughton/RobEx.git && cd /home/RobEx && git checkout headless && git pull origin headless

# Setup RobEx
RUN curl -sSL https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod && \
    apt-get update && \
    ACCEPT_EULA=y apt-get install -y libk4a1.4 \
    libk4a1.4-dev \
    k4a-tools

# Replace requirements.txt with docker_requirements.txt and install
RUN cd /home/RobEx && \
    sed -i "s/requirements/docker_requirements/g" setup.py && \
    ./.make/install_for_docker.sh

RUN echo "cd /home/RobEx/RobEx/train/examples" >> ~/.bashrc
# Final clean up
RUN cd /home && rm -r Open3D
RUN rm -rf /root/.cache /root/.cmake /var/lib/apt/lists/*
