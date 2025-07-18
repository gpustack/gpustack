ARG CUDA_VERSION=12.4.1
ARG CUDA_TAG_SUFFIX=-cudnn-runtime-ubuntu22.04

FROM nvidia/cuda:${CUDA_VERSION}${CUDA_TAG_SUFFIX}

ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

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

COPY . /workspace/gpustack
RUN cd /workspace/gpustack && \
    make build

# Keep same FlashInfer version as vLLM https://github.com/vllm-project/vllm/blob/v0.9.2/docker/Dockerfile#L382
ARG FLASHINFER_GIT_REF=0.2.6.post1

RUN <<EOF
    if [ "$TARGETPLATFORM" = "linux/amd64" ]; then
        # Install vllm dependencies for x86_64
        WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[all]";
    else
        WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[audio]";
    fi
    pip install ${WHEEL_PACKAGE}

    if [ "$TARGETPLATFORM" = "linux/amd64" ] && command -v nvcc &> /dev/null; then
        # Install flashinfer-python JIT for vllm
        pip install flashinfer-python==${FLASHINFER_GIT_REF}
    fi
    rm -rf /workspace/gpustack
EOF

RUN gpustack download-tools

# Prepara variables for vox-box installation.
ENV PATH="/root/.local/bin:${PATH}"
ENV PIPX_LOCAL_VENVS=/root/.local/share/pipx/venvs
ARG VOXBOX_VERSION=0.0.19
ARG TRANSFORMERS_VERSION=4.51.3

RUN <<EOF
    pipx install vox-box==${VOXBOX_VERSION} --pip-args="transformers==${TRANSFORMERS_VERSION}"
    # Download dac weights used by audio models like Dia
    ${PIPX_LOCAL_VENVS}/vox-box/bin/python -m dac download

    # Create a symbolic link for vox-box to prevent version conflicts.
    GPUSTACK_PATH=$(dirname $(which gpustack))
    ln -vsf ${PIPX_LOCAL_VENVS}/vox-box/bin/vox-box ${GPUSTACK_PATH}/vox-box
EOF

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
