ARG CUDA_VERSION=12.4.1
ARG CUDA_TAG_SUFFIX=-cudnn-runtime-ubuntu22.04

FROM nvidia/cuda:${CUDA_VERSION}${CUDA_TAG_SUFFIX}

ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

## Preset this to simplify configuration,
## it is the output of $(pipx environment --value PIPX_LOCAL_VENVS).
ENV PIPX_LOCAL_VENVS=/root/.local/share/pipx/venvs

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    tzdata \
    iproute2 \
    python3 \
    python3-pip \
    python3-venv \
    tini \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

## Install vox-box
#FROM base as voxbox-installer

ARG VOXBOX_VERSION=0.0.18

ENV VOXBOX_VERSION=${VOXBOX_VERSION}

RUN <<EOF
    # - Create virtual environment to place vox-box
    python3 -m venv --system-site-packages ${PIPX_LOCAL_VENVS}/vox-box
    # - Prepare environment
    source ${PIPX_LOCAL_VENVS}/vox-box/bin/activate
    pip install vox-box==${VOXBOX_VERSION} \
        --disable-pip-version-check --no-cache-dir \
        && ln -vsf ${PIPX_LOCAL_VENVS}/vox-box/bin/vox-box /usr/local/bin/vox-box
    # Download dac weights used by audio models like Dia
    python3 -m dac download
    deactivate
EOF



COPY . /workspace/gpustack
RUN cd /workspace/gpustack && \
    make build

RUN <<EOF
    # - Create virtual environment to place gpustack
    python3 -m venv --system-site-packages ${PIPX_LOCAL_VENVS}/gpustack
    # - Prepare environment
    source ${PIPX_LOCAL_VENVS}/gpustack/bin/activate
    if [ "$TARGETPLATFORM" = "linux/amd64" ]; then
        # Install vllm dependencies for x86_64
        WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[vllm]";
    else
        WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)";
    fi
    pip install pipx
    pip install --disable-pip-version-check --no-cache-dir ${WHEEL_PACKAGE} \
        && ln -vsf ${PIPX_LOCAL_VENVS}/gpustack/bin/gpustack /usr/local/bin/gpustack \
        && ln -vsf ${PIPX_LOCAL_VENVS}/vox-box/bin/vox-box ${PIPX_LOCAL_VENVS}/gpustack/bin/vox-box
    pip cache purge
    rm -rf /workspace/gpustack
    deactivate
EOF

RUN gpustack download-tools

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
