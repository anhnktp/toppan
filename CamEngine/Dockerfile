# Stage 1: build a base cuda-enabled image
FROM nvidia/cuda:10.1-devel-ubuntu16.04 as nvidia_base
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

# System packages
RUN apt-get update && apt-get install -y curl git && apt-get install --no-install-recommends -y -q cron
RUN apt-get update && apt-get install -qqy x11-apps && apt-get install -y libgl1-mesa-glx
# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh && \
    bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b && \
    rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

USER root

COPY ./environment.yml ./CamEngine/

RUN conda env create -f ./CamEngine/environment.yml

RUN conda clean --all -y && rm -rf /CamEngine/ && rm -rf /var/lib/apt/lists/*

ENV CONDA_DEFAULT_ENV=proj5
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV AWL_DIR=/awl/cam_engine

WORKDIR ${AWL_DIR}
USER ${USER}

RUN pip install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN python -m pip install --user 'git+https://github.com/facebookresearch/detectron2.git'
RUN python -m pip install --user 'git+https://github.com/KaiyangZhou/deep-person-reid.git'
RUN pip uninstall -y opencv-python
COPY . .

WORKDIR /awl/cam_engine/modules/keypoints

RUN python setup.py build_ext --inplace
RUN conda clean --all -y
WORKDIR ${AWL_DIR}
