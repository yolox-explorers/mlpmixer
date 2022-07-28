# cuda 11 will run into warning You are using ptxas 11.0.221, which is older than 11.1
# Recommended to use device with Nvidia CUDA >= 11.1
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04 

ARG REPO_DIR="."
ARG CONDA_ENV_FILE="vision_transformers/requirements.yml"
ARG CONDA_ENV_NAME="mlpmixer"
ARG PROJECT_USER="user"
ARG HOME_DIR="/home/$PROJECT_USER"

ARG DVC_VERSION="2.8.3"
ARG DVC_BINARY_NAME="dvc_2.8.3_amd64.deb"

ARG CONDA_HOME="$HOME_DIR/miniconda3"
ARG CONDA_BIN="$CONDA_HOME/bin/conda"
ARG MINI_CONDA_SH="Miniconda3-latest-Linux-x86_64.sh"

WORKDIR $HOME_DIR

RUN groupadd -g 2222 $PROJECT_USER && useradd -u 2222 -g 2222 -m $PROJECT_USER

RUN touch "$HOME_DIR/.bashrc"

RUN apt-get update && \
    apt-get -y install bzip2 curl wget gcc rsync git vim locales && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8 && \
    apt-get clean && \
    curl -O https://repo.anaconda.com/miniconda/$MINI_CONDA_SH && \
    chmod +x $MINI_CONDA_SH && \
    ./$MINI_CONDA_SH -b -p $CONDA_HOME && \
    rm $MINI_CONDA_SH

# Required for cocoapi py lib, opencv-python py lib
RUN apt-get update && \
    apt-get install -y gcc && \
    apt-get install ffmpeg libsm6 libxext6 -y

# Required for opencv-python py lib
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y  \
    git wget sudo build-essential \
    python3 python3-setuptools python3-pip python3-dev python3-tk \
    ffmpeg libsm6 libxext6
RUN ln -svf /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade --force pip


RUN wget https://github.com/mikefarah/yq/releases/download/v4.16.1/yq_linux_amd64.tar.gz -O - |\
    tar xz && mv yq_linux_amd64 /usr/bin/yq

RUN wget "https://github.com/iterative/dvc/releases/download/$DVC_VERSION/$DVC_BINARY_NAME" && \
    apt install -y "./$DVC_BINARY_NAME" && \
    rm "./$DVC_BINARY_NAME"

# Copy repo into image
WORKDIR $HOME_DIR
COPY $REPO_DIR MLPmixer

# Install conda env using yaml file
RUN $CONDA_BIN env create -f MLPmixer/$CONDA_ENV_FILE && \
    $CONDA_BIN init bash && \
    $CONDA_BIN clean -a -y && \
    echo "source activate $CONDA_ENV_NAME" >> "$HOME_DIR/.bashrc"

# env var not detected in new shell, so need hard code the CONDA_BIN and CONDA_ENV_NAME var value
SHELL ["/home/user//miniconda3//bin/conda", "run", "-n", "mlpmixer", "/bin/bash", "-c"]

RUN chown -R 2222:2222 $HOME_DIR && \
    rm /bin/sh && ln -s /bin/bash /bin/sh

WORKDIR $HOME_DIR/MLPmixer

ENV PATH $CONDA_HOME/bin:$HOME_DIR/.local/bin:$PATH
ENV PYTHONIOENCODING utf8
ENV LANG "C.UTF-8"
ENV LC_ALL "C.UTF-8"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

USER 2222