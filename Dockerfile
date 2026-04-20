# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    wget \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p /root/miniconda3 \
    && rm ~/miniconda.sh \
    && /root/miniconda3/bin/conda init bash

# Create a new conda environment from the environment.yaml file
COPY environment.yaml .
RUN conda env create -f environment.yaml

RUN conda run -n geval pip install flash-attn --no-build-isolation

RUN mkdir /scripts
RUN mkdir /scripts/results
COPY ./llama_eval.py /scripts/llama_eval.py
COPY ./contrastive_decoding.py /scripts/contrastive_decoding.py
ADD prompts/summeval /scripts/prompts/summeval
COPY data/summeval.json /scripts/data/summeval.json

# Set up entry point to activate the conda environment
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate geval" >> ~/.bashrc

# Set the ENTRYPOINT to use bash with the conda environment activated
#ENTRYPOINT ["/bin/bash", "-c", "source ~/.bashrc && exec \"$@\"", "--"]
#
## Default command
#CMD ["/bin/bash"]


# Make sure NVIDIA runtime is used
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
