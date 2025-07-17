ARG CUDA_VERSION=12.4.1
ARG CUDA_TAG_SUFFIX=-cudnn-runtime-ubuntu22.04
ARG CUDA_TAG_SUFFIX_DEV=-cudnn-devel-ubuntu22.04

FROM nvidia/cuda:${CUDA_VERSION}${CUDA_TAG_SUFFIX_DEV} AS builder

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN <<EOF
    mkdir /whl
    git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
    cd flashinfer
    python3 -m pip install -v .

    # for development & contribution, install in editable mode
    python3 -m pip install --no-build-isolation -e . -v
    # Set target CUDA architectures
    export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a"
    # Build AOT kernels. Will produce AOT kernels in aot-ops/
    python -m flashinfer.aot
    # Build AOT wheel
    python -m build --no-isolation --wheel
    # Install AOT wheel
    cp dist/flashinfer-*.whl /whl/
EOF
ENTRYPOINT [ "python3" ]

FROM nvidia/cuda:${CUDA_VERSION}${CUDA_TAG_SUFFIX}

COPY --from=builder /whl /whl

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

    if [ -n "$(ls /whl/*.whl 2>/dev/null)" ]; then
        pip install --no-cache-dir /whl/*.whl;
        echo "FlashInfer installed";
    fi
    rm -rf /workspace/gpustack
EOF

RUN gpustack download-tools

# Prepara variables for vox-box installation.
ENV PATH="/root/.local/bin:${PATH}"
ENV PIPX_LOCAL_VENVS=/root/.local/share/pipx/venvs
ARG VOXBOX_VERSION=0.0.18
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
