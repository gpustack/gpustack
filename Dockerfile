# Package logic:
# 1. base target:
#    - Install/Upgrade tools, including Python.
# 2. build-base target:
#    - Install/Upgrade tools, including Python, CMake, Make, CCache, SCCache and dependencies.
# 2.1. (linux/amd64) flashinfer-build target:
#    - Install FlashInfer build dependencies, including PyTorch.
#    - Build FlashInfer wheel.
# 3. gpustack target(final):
#    - (linux/amd64) Install FlashInfer as a Python library for GPUStack.
#    - Install GPUStack.
#    - Install Vox-Box as an independent executor for GPUStack,
#      see https://github.com/gpustack/gpustack/pull/2473#issue-3222391256.
#    - Set up the entrypoint to start GPUStack.

# Arguments description:
# - CUDA_VERSION is the version of NVIDIA CUDA,
#   which is used to point to the base image for running.
# - CUDA_DEVEL_VERSION is the version of NVIDIA CUDA,
#   which is used to point to the base image for 3rdparty components building.
# - TORCH_VERSION is the version of PyTorch,
#   which should be compatible with the CUDA, vLLM and other components.
# - TORCH_CUDA_ARCH_LIST is the CUDA architecture list for PyTorch,
#   which is used to build the components that depend on PyTorch,
#   default is empty, which means it will be set automatically based on the CUDA version.
# - FLASHINFER_VERSION is the version of FlashInfer,
#   which is used to build the FlashInfer wheel.
# - PYTHON_VERSION is the version of Python,
#   which should be properly set, it must be 3.x.
ARG CUDA_VERSION=12.4.1
ARG CUDA_DEVEL_VERSION=12.6.3
ARG TORCH_VERSION=2.7.0
ARG TORCH_CUDA_ARCH_LIST=""
ARG FLASHINFER_VERSION=0.2.6.post1
ARG FLASHINFER_BUILD_MAX_JOBS=""
ARG PYTHON_VERSION=3.11

# Stage Base
#
# Example build command:
#   docker build --tag=gpustack/gpustack:cuda-base --file=Dockerfile --target=base --progress=plain .
#

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04 AS base
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

## Install Tools

ENV DEBIAN_FRONTEND=noninteractive

RUN <<EOF
    # Tools

    # Refresh
    apt-get update -y && apt-get install -y --no-install-recommends \
        software-properties-common apt-transport-https \
        ca-certificates gnupg2 lsb-release gnupg-agent \
      && apt-get update -y \
      && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
      && apt-get update -y

    # Install
    apt-get install -y --no-install-recommends \
        ca-certificates build-essential binutils bash openssl \
        curl wget aria2 \
        git git-lfs \
        unzip xz-utils \
        tzdata locales \
        iproute2 iputils-ping ifstat net-tools dnsutils pciutils ipmitool \
        procps sysstat htop \
        tini vim jq bc tree

    # Update locale
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

ENV LANG='en_US.UTF-8' \
    LANGUAGE='en_US:en' \
    LC_ALL='en_US.UTF-8'

## Install Python

ARG PYTHON_VERSION

ENV PYTHON_VERSION=${PYTHON_VERSION}

RUN <<EOF
    # Python

    # Add deadsnakes PPA for Python versions
    for i in 1 2 3; do
        add-apt-repository -y ppa:deadsnakes/ppa && break || { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }
    done
    apt-get update -y

    # Install
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils \
        python${PYTHON_VERSION}-lib2to3 \
        python${PYTHON_VERSION}-gdbm \
        python${PYTHON_VERSION}-tk \
        libibverbs-dev

    # Update alternatives
    if [ -f /etc/alternatives/python3 ]; then update-alternatives --remove-all python3; fi; update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
    if [ -f /etc/alternatives/python ]; then update-alternatives --remove-all python; fi; update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
    curl -sS "https://bootstrap.pypa.io/get-pip.py" | python${PYTHON_VERSION}
    if [ -f /etc/alternatives/2to3 ]; then update-alternatives --remove-all 2to3; fi; update-alternatives --install /usr/bin/2to3 2to3 /usr/bin/2to3${PYTHON_VERSION} 1 || true
    if [ -f /etc/alternatives/pydoc3 ]; then update-alternatives --remove-all pydoc3; fi; update-alternatives --install /usr/bin/pydoc3 pydoc3 /usr/bin/pydoc${PYTHON_VERSION} 1 || true
    if [ -f /etc/alternatives/idle3 ]; then update-alternatives --remove-all idle3; fi; update-alternatives --install /usr/bin/idle3 idle3 /usr/bin/idle${PYTHON_VERSION} 1 || true
    if [ -f /etc/alternatives/python3-config ]; then update-alternatives --remove-all python3-config; fi; update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python${PYTHON_VERSION}-config 1 || true

    # Install packages
    cat <<EOT >/tmp/requirements.txt
setuptools==80.7.1
pipx==1.7.1
EOT
    pip install --disable-pip-version-check --no-cache-dir --root-user-action ignore -r /tmp/requirements.txt

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Preset this to simplify configuration,
## it is the output of $(pipx environment --value PIPX_LOCAL_VENVS).
ENV PIPX_LOCAL_VENVS=/root/.local/share/pipx/venvs

#
# Stage Build Base
#
# Example build command:
#   docker build --tag=gpustack/gpustack:cuda-base-build --file=Dockerfile --target=build-base --progress=plain .
#

FROM nvidia/cuda:${CUDA_DEVEL_VERSION}-cudnn-devel-ubuntu22.04 AS base-build
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

## Install Tools

ENV DEBIAN_FRONTEND=noninteractive

RUN <<EOF
    # Tools

    # Refresh
    apt-get update -y && apt-get install -y --no-install-recommends \
        software-properties-common apt-transport-https \
        ca-certificates gnupg2 lsb-release gnupg-agent \
      && apt-get update -y \
      && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
      && apt-get update -y

    # Install
    apt-get install -y --no-install-recommends \
        ca-certificates build-essential binutils bash openssl \
        curl wget aria2 \
        git git-lfs \
        unzip xz-utils \
        tini vim jq bc tree

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Install CMake/Make/CCache/SCCACHE

RUN <<EOF
    # CMake/Make/CCache/SCCACHE

    # Install
    apt-get install -y --no-install-recommends \
        pkg-config make cmake ccache
    curl --retry 3 --retry-connrefused -fL "https://github.com/mozilla/sccache/releases/download/v0.10.0/sccache-v0.10.0-$(uname -m)-unknown-linux-musl.tar.gz" | tar -zx -C /usr/bin --strip-components 1

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Install Compile Dependencies

RUN <<EOF
    # Dependencies

    # Install
    apt-get install -y --no-install-recommends \
        zlib1g zlib1g-dev libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev \
        openssl libssl-dev libsqlite3-dev lcov libomp-dev \
        libblas-dev liblapack-dev libopenblas-dev libblas3 liblapack3 libhdf5-dev \
        libxml2 libxslt1-dev libgl1-mesa-glx libgmpxx4ldbl \
        libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
        liblzma-dev lzma lzma-dev tk-dev uuid-dev libmpdec-dev \
        libnuma-dev

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Install Python

ARG PYTHON_VERSION

ENV PYTHON_VERSION=${PYTHON_VERSION}

RUN <<EOF
    # Python

    # Install Python
    for i in 1 2 3; do
        add-apt-repository -y ppa:deadsnakes/ppa && break || { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }
    done
    apt-get update -y && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils \
        python${PYTHON_VERSION}-lib2to3 \
        python${PYTHON_VERSION}-gdbm \
        python${PYTHON_VERSION}-tk \
        libibverbs-dev

    # Update alternatives
    if [ -f /etc/alternatives/python3 ]; then update-alternatives --remove-all python3; fi; update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
    if [ -f /etc/alternatives/python ]; then update-alternatives --remove-all python; fi; update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
    curl -sS "https://bootstrap.pypa.io/get-pip.py" | python${PYTHON_VERSION}
    if [ -f /etc/alternatives/2to3 ]; then update-alternatives --remove-all 2to3; fi; update-alternatives --install /usr/bin/2to3 2to3 /usr/bin/2to3${PYTHON_VERSION} 1 || true
    if [ -f /etc/alternatives/pydoc3 ]; then update-alternatives --remove-all pydoc3; fi; update-alternatives --install /usr/bin/pydoc3 pydoc3 /usr/bin/pydoc${PYTHON_VERSION} 1 || true
    if [ -f /etc/alternatives/idle3 ]; then update-alternatives --remove-all idle3; fi; update-alternatives --install /usr/bin/idle3 idle3 /usr/bin/idle${PYTHON_VERSION} 1 || true
    if [ -f /etc/alternatives/python3-config ]; then update-alternatives --remove-all python3-config; fi; update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python${PYTHON_VERSION}-config 1 || true

    # Install packages
    cat <<EOT >/tmp/requirements.txt
build
ninja==1.11.1.4
packaging==25.0
pipx==1.7.1
setuptools==80.7.1
setuptools-scm==8.3.1
wheel==0.45.1
cmake==4.0.3
EOT
    pip install --disable-pip-version-check --no-cache-dir --root-user-action ignore -r /tmp/requirements.txt

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

#
# Stage FlashInfer Build (linux/amd64 only)
#
# Example build command:
#   docker build --platform=linux/amd64 --tag=gpustack/gpustack:cuda-flashinfer-build --file=Dockerfile --target=flashinfer-build --progress=plain .
#

FROM --platform=linux/amd64 base-build AS flashinfer-build

ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

ARG CUDA_VERSION
ARG CUDA_DEVEL_VERSION
ARG TORCH_VERSION
ARG TORCH_CUDA_ARCH_LIST
ARG FLASHINFER_VERSION
ARG FLASHINFER_BUILD_MAX_JOBS

ENV CUDA_VERSION=${CUDA_VERSION} \
    CUDA_DEVEL_VERSION=${CUDA_DEVEL_VERSION} \
    TORCH_VERSION=${TORCH_VERSION} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FLASHINFER_VERSION=${FLASHINFER_VERSION} \
    FLASHINFER_BUILD_MAX_JOBS=${FLASHINFER_BUILD_MAX_JOBS}

## Build FlashInfer

RUN <<EOF
    # FlashInfer

    mkdir -p /workspace/flashinfer

    if [[ "${TARGETARCH}" == "arm64" ]]; then
        echo "Skipping FlashInfer build for ${TARGETARCH}..."
        exit 0
    fi

    # Prepare
    IFS="." read -r CUDA_MAJOR CUDA_MINOR CUDA_PATCH <<< "${CUDA_DEVEL_VERSION}"
    IFS="." read -r TORCH_MAJOR TORCH_MINOR TORCH_PATCH <<< "${TORCH_VERSION}"
    ldconfig /usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}/compat/

    if [[ "${CUDA_MAJOR}.${CUDA_MINOR}" == "12.8" && "${TORCH_MAJOR}.${TORCH_MINOR}" == "2.7" ]]; then
        mkdir -p /workspace/flashinfer/dist
        if curl --retry 3 --retry-connrefused -fL https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-${FLASHINFER_VERSION}%2Bcu128torch2.7-cp39-abi3-linux_x86_64.whl \
            -o /workspace/flashinfer/dist/flashinfer_python-${FLASHINFER_VERSION}+cu128torch2.7-cp39-abi3-linux_x86_64.whl; then
            echo "Using prebuilt FlashInfer ${FLASHINFER_VERSION} wheel for CUDA ${CUDA_MAJOR}.${CUDA_MINOR} and PyTorch ${TORCH_MAJOR}.${TORCH_MINOR}..."
            exit 0
        fi
    fi

    echo "Building FlashInfer ${FLASHINFER_VERSION} wheel for CUDA ${CUDA_MAJOR}.${CUDA_MINOR} and PyTorch ${TORCH_MAJOR}.${TORCH_MINOR}..."

    # Install dependencies,
    # extract https://github.com/flashinfer-ai/flashinfer/blob/main/scripts/run-ci-build-wheel.sh.
    cat <<EOT >/tmp/requirements.txt
torch==${TORCH_VERSION}+cu${CUDA_MAJOR}${CUDA_MINOR}
torchaudio
torchvision
numpy
EOT
    pip install --disable-pip-version-check --no-cache-dir --root-user-action ignore -r /tmp/requirements.txt --index-url https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}

    # Download
    git clone --depth 1 --recursive --shallow-submodules \
        --branch v${FLASHINFER_VERSION} \
        https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer

    # Build
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
    if [[ -z "${TORCH_CUDA_ARCH_LIST}" ]]; then
        if (( $(echo "${CUDA_MAJOR} < 12" | bc -l) )); then
            export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9"
        elif (( $(echo "${CUDA_MAJOR}.${CUDA_MINOR} < 12.8" | bc -l) )); then
            export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0+PTX"
        else
            export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0+PTX 10.0+PTX 12.0"
        fi
    fi
    export FLASHINFER_LOCAL_VERSION="cu${CUDA_MAJOR}${CUDA_MINOR}torch${TORCH_MAJOR}.${TORCH_MINOR}"
    export MAX_JOBS="$(($(nproc) - 1))"
    if [[ -n "${FLASHINFER_BUILD_MAX_JOBS}" ]]; then
        export MAX_JOBS="${FLASHINFER_BUILD_MAX_JOBS}"
    fi
    echo "Building FlashInfer with the following environment variables:"
    echo "  TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
    echo "  FLASHINFER_LOCAL_VERSION: ${FLASHINFER_LOCAL_VERSION}"
    echo "  MAX_JOBS: ${MAX_JOBS}"
    pushd /tmp/flashinfer
    python -v -m flashinfer.aot
    python -v -m build --no-isolation --wheel
    popd

    # Archive
    mv /tmp/flashinfer/dist /workspace/flashinfer/

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && pip cache purge
EOF

#
# Stage GPUStack
#
# Example build command:
#   docker build --tag=gpustack/gpustack:cuda --file=Dockerfile --progress=plain .
#

FROM base AS gpustack

ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

ARG CUDA_VERSION
ARG CUDA_DEVEL_VERSION
ARG TORCH_VERSION

ENV CUDA_VERSION=${CUDA_VERSION} \
    CUDA_DEVEL_VERSION=${CUDA_DEVEL_VERSION} \
    TORCH_VERSION=${TORCH_VERSION}

## Install FlashInfer as a Python library for GPUStack (linux/amd64 only)

RUN --mount=type=bind,from=flashinfer-build,source=/workspace/flashinfer,target=/workspace/flashinfer,rw <<EOF
    # FlashInfer

    if [[ "${TARGETARCH}" == "arm64" ]]; then
        echo "Skipping FlashInfer installation for ${TARGETARCH}..."
        exit 0
    fi

    # Prepare
    IFS="." read -r CUDA_MAJOR CUDA_MINOR CUDA_PATCH <<< "${CUDA_DEVEL_VERSION}"

    # Install dependencies,
    # extract https://github.com/flashinfer-ai/flashinfer/blob/main/scripts/run-ci-build-wheel.sh.
    cat <<EOT >/tmp/requirements.txt
torch==${TORCH_VERSION}+cu${CUDA_MAJOR}${CUDA_MINOR}
torchaudio
torchvision
numpy
EOT
    pip install --disable-pip-version-check --no-cache-dir --root-user-action ignore -r /tmp/requirements.txt --index-url https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}

    # Install
    pip install --disable-pip-version-check --no-cache-dir --root-user-action ignore /workspace/flashinfer/dist/*.whl

    # Review
    pip freeze

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt \
        && pip cache purge
EOF

## Install GPUStack

RUN --mount=type=bind,target=/workspace/gpustack,rw <<EOF
    # GPUStack

    # Build GPUStack
    cd /workspace/gpustack \
        && git config --global --add safe.directory /workspace/gpustack \
        && make build

    # Install GPUStack.
    # FIXME: There is no linux/arm64 vLLM prebuilt wheel,
    #        so we only install the all wheel for linux/amd64.
    if [ "${TARGETARCH}" == "amd64" ]; then
        WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[all]";
    else
        WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[audio]";
    fi
    pip install --disable-pip-version-check --no-cache-dir --root-user-action ignore ${WHEEL_PACKAGE}

    # Download tools
    gpustack download-tools --device cuda

    # Set up environment
    mkdir -p /var/lib/gpustack \
        && chmod -R 0755 /var/lib/gpustack

    # Review
    pip freeze

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt \
        && rm -rf /workspace/gpustack/dist \
        && pip cache purge
EOF

## Install Vox-Box as an independent executor for GPUStack

RUN <<EOF
    # Vox-Box

    # Get version of Vox-Box from GPUStack
    VERSION=$(pip freeze | grep vox_box== | head -n 1 | cut -d'=' -f3)

    # Install dependencies
    apt-get install -y --no-install-recommends \
        ffmpeg

    # Pre process
    # - Create virtual environment to place vox-box
    python -m venv --system-site-packages ${PIPX_LOCAL_VENVS}/vox-box
    # - Prepare environment
    source ${PIPX_LOCAL_VENVS}/vox-box/bin/activate

    # Reinstall Vox-Box,
    # as we create a virtual environment which inherits system site packages,
    # so we need to reinstall Vox-Box to ensure it is installed in the virtual environment.
    # We also lock the transformers version here to fix https://github.com/gpustack/gpustack/pull/2473.
    cat <<EOT >/tmp/requirements.txt
transformers==4.51.3
vox-box==${VERSION}
EOT
    pip install --disable-pip-version-check --no-cache-dir --root-user-action ignore --force-reinstall --no-dependencies -r /tmp/requirements.txt \
        && ln -vsf ${PIPX_LOCAL_VENVS}/vox-box/bin/vox-box /usr/local/bin/vox-box

    # Download tools
    # - Download dac weights used by audio models like Dia.
    python -m dac download

    # Post process
    deactivate

    # Review
    pipx runpip vox-box freeze

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt \
        && pip cache purge
EOF

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
