# Base image
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Install base utilities (Installing additional software)
RUN apt-get update && \
    apt install -y gcc-9 g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 && \
    apt-get install -y wget git build-essential mesa-utils ffmpeg libsm6 libxext6 libgl1-mesa-dev libosmesa6-dev && \
    apt-get clean

# Set environment variable to use host's X11 display
ENV DISPLAY=:0

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Add Conda to path, to enable future use
ENV PATH=$CONDA_DIR/bin:$PATH

# Create Conda environment with python3.10
RUN conda create --name gaussian-avatars -y python=3.10

# Activate Conda environment
RUN echo "source activate gaussian-avatars" > ~/.bashrc
ENV PATH=/opt/conda/envs/gaussian-avatars/bin:$PATH

# Environment Setup
RUN conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit ninja && \
    ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64" && \
    conda env config vars set CUDA_HOME=/usr/local/cuda-11.7
ENV CUDA_HOME=/usr/local/cuda-11.7
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# Copy resources into the image
COPY . /GaussianAvatars

WORKDIR /GaussianAvatars

