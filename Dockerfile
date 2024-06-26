# Use the official NVIDIA CUDA image with Conda pre-installed as a base
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install necessary system packages
RUN apt-get update && apt-get install -y wget git libgl1-mesa-glx gnupg software-properties-common

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /miniconda && \
    rm miniconda.sh

# Set the path to include the Conda bin directory
ENV PATH /miniconda/bin:$PATH

# Initialize Conda in bash (not necessary for activation, but can be useful for other purposes)
RUN conda init bash

# Initialize the conda environment
COPY environment.yml /environment.yml
RUN conda env create --name "random_tensor_laboratory" -f environment.yml

# Activate the conda environment for interactive shell sessions
SHELL ["conda", "run", "-n", "random_tensor_laboratory", "/bin/bash", "-c"]

WORKDIR /opt
RUN git clone https://github.com/LLNL/LEAP.git
WORKDIR /opt/LEAP
RUN pip install -v .
WORKDIR /

# copy the directory random_tensor_laboratory to the container
WORKDIR /home
RUN git clone https:///github.com/tivnanmatt/random_tensor_laboratory.git
WORKDIR /home/random_tensor_laboratory/random_tensor_laboratory
# RUN pip install -e .
# WORKDIR /

