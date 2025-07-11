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

RUN python3 -m pip install --no-cache-dir pipx

ENV PATH="/root/.local/bin:${PATH}"

ARG VOXBOX_VERSION=0.0.18
ARG TRANSFORMERS_VERSION=4.51.3

ENV VOXBOX_VERSION=${VOXBOX_VERSION}
ENV TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION}

RUN <<EOF
    pipx install vox-box==${VOXBOX_VERSION} --pip-args="transformers==${TRANSFORMERS_VERSION}"
    # Download dac weights used by audio models like Dia
    ${PIPX_LOCAL_VENVS}/vox-box/bin/python -m dac download
EOF



COPY . /workspace/gpustack
RUN cd /workspace/gpustack && \
    make build

RUN <<EOF
    if [ "$TARGETPLATFORM" = "linux/amd64" ]; then
        # Install vllm dependencies for x86_64
        WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[all]";
    else
        WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[audio]";
    fi
    pipx install ${WHEEL_PACKAGE} \
        && ln -vsf ${PIPX_LOCAL_VENVS}/gpustack/bin/gpustack /usr/local/bin/gpustack \
        && ln -vsf ${PIPX_LOCAL_VENVS}/vox-box/bin/vox-box ${PIPX_LOCAL_VENVS}/gpustack/bin/vox-box
    rm -rf /workspace/gpustack
EOF

RUN gpustack download-tools

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
