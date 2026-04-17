ARG BASE_IMAGE=nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04

FROM ${BASE_IMAGE}


ARG USER_NAME=<whatever>
ARG USER_ID=1000


# Prevent anything requiring user input
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Basic packages
RUN apt-get -y update \
    && apt-get -y install \
      python3-venv \
      python-is-python3 \
      python3-pip \
      sudo \
      vim \
      wget \
      curl \
      software-properties-common \
      doxygen \
      git \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get -y update \
    && apt-get -y install \
        libglew-dev \
        libassimp-dev \
        libboost-all-dev \
        libgtk-3-dev \
        libglfw3-dev \
        libavdevice-dev \
        libavcodec-dev \
        libeigen3-dev \
        libxxf86vm-dev \
        libembree-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -y update \
    && apt-get -y install \ 
        cmake \
        libosmesa6 \
    && rm -rf /var/lib/apt/lists/*
 


RUN useradd -m -l -u ${USER_ID} -s /bin/bash ${USER_NAME} \
    && usermod -aG video ${USER_NAME} \
    && export PATH=$PATH:/home/${USER_NAME}/.local/bin

RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to user to run user-space commands
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

RUN sudo chown -R ${USER_NAME} /home/${USER_NAME}

COPY ./entrypoint.sh /entrypoint.sh
RUN sudo chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]