# Package logic:
# 1. base target:
#    - Install tools.
#    - Upgrade GCC if needed.
#    - Install C buildkit.
#    - Upgrade Python if needed.
#    - Install Python buildkit.
#    - Install Torch.
#    - Install PostgreSQL.
#    - Install S6-overlay.
# 2. build-base target:
#    - Install tools.
#    - Upgrade GCC if needed.
#    - Install C buildkit.
#    - Upgrade Python if needed.
#    - Install Python buildkit.
#    - Install Torch.
# 2.1. flashinfer-build target:
#    - Build FlashInfer wheel.
# 3. gpustack target(final):
#    - Install FlashInfer as a Python library for GPUStack if existed.
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
ARG TORCH_VERSION=2.7.1
ARG TORCH_CUDA_ARCH_LIST=""
ARG FLASHINFER_VERSION=0.2.11
ARG FLASHINFER_BUILD_MAX_JOBS=""
ARG PYTHON_VERSION=3.11

# Stage Base
#
# Example build command:
#   docker build --tag=gpustack/gpustack:cuda-base --file=pack/Dockerfile --target=base --progress=plain .
#

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04 AS base
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    LANG='en_US.UTF-8' \
    LANGUAGE='en_US:en' \
    LC_ALL='en_US.UTF-8'

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
        vim jq bc tree

    # Update locale
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

    # Update timezone
    rm -f /etc/localtime \
        && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
        && echo "Asia/Shanghai" > /etc/timezone \
        && dpkg-reconfigure --frontend noninteractive tzdata

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Upgrade GCC if needed

RUN <<EOF
    # GCC

    # Upgrade GCC if the Ubuntu version is lower than 21.04.
    source /etc/os-release
    if (( $(echo "${VERSION_ID} >= 21.04" | bc -l) )); then
        echo "Skipping GCC upgrade for ${VERSION_ID}..."
        exit 0
    fi

    # Install
    apt-get install -y --no-install-recommends \
        gcc-11 g++-11 gfortran-11 gfortran

    # Update alternatives
    if [[ -f /etc/alternatives/gcov-dump ]]; then update-alternatives --remove-all gcov-dump; fi; update-alternatives --install /usr/bin/gcov-dump gcov-dump /usr/bin/gcov-dump-11 10
    if [[ -f /etc/alternatives/lto-dump ]]; then update-alternatives --remove-all lto-dump; fi; update-alternatives --install /usr/bin/lto-dump lto-dump /usr/bin/lto-dump-11 10
    if [[ -f /etc/alternatives/gcov ]]; then update-alternatives --remove-all gcov; fi; update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-11 10
    if [[ -f /etc/alternatives/gcc ]]; then update-alternatives --remove-all gcc; fi; update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 10
    if [[ -f /etc/alternatives/gcc-nm ]]; then update-alternatives --remove-all gcc-nm; fi; update-alternatives --install /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-11 10
    if [[ -f /etc/alternatives/cpp ]]; then update-alternatives --remove-all cpp; fi; update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-11 10
    if [[ -f /etc/alternatives/g++ ]]; then update-alternatives --remove-all g++; fi; update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 10
    if [[ -f /etc/alternatives/gcc-ar ]]; then update-alternatives --remove-all gcc-ar; fi; update-alternatives --install /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-11 10
    if [[ -f /etc/alternatives/gcov-tool ]]; then update-alternatives --remove-all gcov-tool; fi; update-alternatives --install /usr/bin/gcov-tool gcov-tool /usr/bin/gcov-tool-11 10
    if [[ -f /etc/alternatives/gcc-ranlib ]]; then update-alternatives --remove-all gcc-ranlib; fi; update-alternatives --install /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-11 10
    if [[ -f /etc/alternatives/gfortran ]]; then update-alternatives --remove-all gfortran; fi; update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-11 10

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Install C buildkit

RUN <<EOF
    # C buildkit

    # Install
    apt-get install -y --no-install-recommends \
        make ninja-build pkg-config ccache
    curl --retry 3 --retry-connrefused -fL "https://github.com/Kitware/CMake/releases/download/v3.31.7/cmake-3.31.7-linux-$(uname -m).tar.gz" | tar -zx -C /usr --strip-components 1

    # Install dependencies
    apt-get install -y --no-install-recommends \
        perl-openssl-defaults perl yasm \
        zlib1g zlib1g-dev libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev \
        openssl libssl-dev lcov libomp-dev \
        libblas-dev liblapack-dev libopenblas-dev libblas3 liblapack3 libhdf5-dev \
        libxml2 libxslt1-dev libgl1-mesa-glx libgmpxx4ldbl \
        libncurses5-dev libreadline6-dev \
        liblzma-dev lzma lzma-dev tk-dev uuid-dev libmpdec-dev \
        ffmpeg libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
        libnuma-dev libjemalloc-dev

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Upgrade Python if needed

ARG PYTHON_VERSION

ENV PYTHON_VERSION=${PYTHON_VERSION}

RUN <<EOF
    # Python

    if (( $(echo "$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2) == ${PYTHON_VERSION}" | bc -l) )); then
        echo "Skipping Python upgrade for ${PYTHON_VERSION}..."
        if [[ -z "$(ldconfig -v 2>/dev/null | grep libpython${PYTHON_VERSION})" ]]; then
            PYTHON_LIB_PREFIX=$(python3 -c "import sys; print(sys.base_prefix);")
            echo "${PYTHON_LIB_PREFIX}/lib" >> /etc/ld.so.conf.d/python3.conf
            echo "${PYTHON_LIB_PREFIX}/lib64" >> /etc/ld.so.conf.d/python3.conf
        fi
        exit 0
    fi

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
    if [[ -f /etc/alternatives/python3 ]]; then update-alternatives --remove-all python3; fi; update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
    if [[ -f /etc/alternatives/python ]]; then update-alternatives --remove-all python; fi; update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
    curl -sS "https://bootstrap.pypa.io/get-pip.py" | python${PYTHON_VERSION}
    if [[ -f /etc/alternatives/2to3 ]]; then update-alternatives --remove-all 2to3; fi; update-alternatives --install /usr/bin/2to3 2to3 /usr/bin/2to3${PYTHON_VERSION} 1 || true
    if [[ -f /etc/alternatives/pydoc3 ]]; then update-alternatives --remove-all pydoc3; fi; update-alternatives --install /usr/bin/pydoc3 pydoc3 /usr/bin/pydoc${PYTHON_VERSION} 1 || true
    if [[ -f /etc/alternatives/idle3 ]]; then update-alternatives --remove-all idle3; fi; update-alternatives --install /usr/bin/idle3 idle3 /usr/bin/idle${PYTHON_VERSION} 1 || true
    if [[ -f /etc/alternatives/python3-config ]]; then update-alternatives --remove-all python3-config; fi; update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python${PYTHON_VERSION}-config 1 || true

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Install Python buildkit

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN <<EOF
    # Buildkit

    cat <<EOT >/tmp/requirements.txt
build
cmake<4
ninja<1.11
setuptools<80
setuptools-scm
packaging<25
wheel
pybind11
Cython
psutil==7.0.0
pipx==1.7.1
EOT
    pip install -r /tmp/requirements.txt

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/*
EOF

## Preset this to simplify configuration,
## it is the output of $(pipx environment --value PIPX_LOCAL_VENVS).
ENV PIPX_HOME=/root/.local/share/pipx \
    PIPX_LOCAL_VENVS=/root/.local/share/pipx/venvs \
    USE_EMOJI="false"

## Install Torch

ARG CUDA_VERSION
ARG CUDA_DEVEL_VERSION
ARG TORCH_VERSION

ENV CUDA_HOME="/usr/local/cuda" \
    CUDA_VERSION=${CUDA_VERSION} \
    CUDA_DEVEL_VERSION=${CUDA_DEVEL_VERSION} \
    TORCH_VERSION=${TORCH_VERSION}

ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/compat:${LD_LIBRARY_PATH}"

RUN <<EOF
    # Torch

    # Install
    cat <<EOT >/tmp/requirements.txt
torch==${TORCH_VERSION}
torchvision
torchaudio
EOT
    IFS="." read -r CUDA_MAJOR CUDA_MINOR CUDA_PATCH <<< "${CUDA_DEVEL_VERSION}"
    if (( $(echo "${CUDA_MAJOR}.${CUDA_MINOR} > 12.7" | bc -l) )) || [[ "${TARGETARCH}" == "amd64" ]]; then
        pip install --index-url https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR} \
            -r /tmp/requirements.txt
    else
        pip install --extra-index-url https://download.pytorch.org/whl/cpu/ \
            -r /tmp/requirements.txt
    fi
    pip install \
        numpy scipy

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/*
EOF

## Install PostgreSQL

ENV PGDATA=/var/lib/postgresql/data
ENV PGCONFIG_FILE=/etc/postgresql/main/postgresql.conf
ENV POSTGRES_DB=gpustack

RUN set -eux; \
	groupadd -r postgres --gid=9999; \
	useradd -r -g postgres --uid=9999 --home-dir=/var/lib/postgresql --shell=/bin/bash postgres; \
	mkdir -p /var/lib/postgresql; \
	chown -R postgres:postgres /var/lib/postgresql

RUN <<EOF
    # PostgreSQL

    set -eux

    # Add PostgreSQL APT repository
    wget -O /tmp/ACCC4CF8.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        && gpg --dearmor /tmp/ACCC4CF8.asc \
        && mv /tmp/ACCC4CF8.asc.gpg /usr/share/keyrings/postgresql-archive-keyring.gpg \
        && echo "deb [signed-by=/usr/share/keyrings/postgresql-archive-keyring.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list
    
    # Install
    apt-get update -y && apt-get install -y --no-install-recommends \
        postgresql-17 \
        gosu

    # Prepare directories
    mkdir -p /var/log/postgresql
    chown postgres:postgres /var/log/postgresql
    chmod 0755 /var/log/postgresql

    # Create symlinks for PostgreSQL 17 to simplify usage
    ln -s /usr/lib/postgresql/17/bin /usr/lib/postgresql/bin \
        && ln -s /etc/postgresql/17/main /etc/postgresql/main \
        && ln -s /var/lib/postgresql/17/main /var/lib/postgresql/main \
        && ls -1 /usr/lib/postgresql/bin/ | xargs -I @ ln -sf /usr/lib/postgresql/bin/@ /usr/bin/@

    gosu postgres echo "listen_addresses='*'" >> "$PGCONFIG_FILE" \
        && gosu postgres echo "local all  postgres            peer" > /etc/postgresql/main/pg_hba.conf \
        && gosu postgres echo "host  all  root  127.0.0.1/32  trust" >> /etc/postgresql/main/pg_hba.conf \
        && gosu postgres echo "host  all  root  ::1/128       trust" >> /etc/postgresql/main/pg_hba.conf \
        && gosu postgres echo "host  all  all   0.0.0.0/0     scram-sha-256" >> /etc/postgresql/main/pg_hba.conf

    gosu postgres initdb -D "$PGDATA"
    gosu postgres pg_ctl -D "$PGDATA" -o "-c listen_addresses='localhost'" -w start
    gosu postgres psql --command "CREATE USER root WITH PASSWORD 'Root123';"
    gosu postgres psql --command "CREATE DATABASE gpustack OWNER root;"
    gosu postgres pg_ctl -D "$PGDATA" -m fast -w stop

    gosu postgres sed -i "s/^data_directory/#data_directory/" "$PGCONFIG_FILE" \
        && gosu postgres sed -i "s/^hba_file/#hba_file/" "$PGCONFIG_FILE" \
        && gosu postgres sed -i "s/^#log_destination/log_destination/" "$PGCONFIG_FILE" \
        && gosu postgres sed -i "s/^#log_min_messages = warning/log_min_messages = info/" "$PGCONFIG_FILE" \
        && gosu postgres sed -i "s/^#logging_collector = off/logging_collector = on/" "$PGCONFIG_FILE" \
        && gosu postgres sed -i 's/^#log_directory = '\''log'\''/log_directory = '\''\/var\/log\/postgresql'\''/g' "$PGCONFIG_FILE" \
        && gosu postgres sed -i "s/^#log_filename/log_filename/" "$PGCONFIG_FILE" \
        && gosu postgres sed -i "s/^#log_rotation_size/log_rotation_size/" "$PGCONFIG_FILE"

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt

EOF

## Install s6-overlay

ARG S6_OVERLAY_VERSION=3.2.1.0
RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64)   S6_ARCH="x86_64" ;; \
        arm64)   S6_ARCH="aarch64" ;; \
        arm/v7)  S6_ARCH="armhf" ;; \
        arm/v6)  S6_ARCH="arm" ;; \
        *) \
            echo >&2 "⚠️  Warning: Unknown TARGETARCH='${TARGETARCH}', defaulting to x86_64"; \
            S6_ARCH="x86_64"; \
        ;; \
    esac; \
    echo "Installing s6-overlay ${S6_OVERLAY_VERSION} for arch: ${S6_ARCH} (from TARGETARCH=${TARGETARCH})"; \
    base_url="https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}"; \
    for pkg in noarch ${S6_ARCH}; do \
        wget -q -O "/tmp/s6-overlay-${pkg}.tar.xz" "${base_url}/s6-overlay-${pkg}.tar.xz"; \
    done \
    && echo "📦  Extracting s6-overlay ..." \
    && tar -C / -Jxpf /tmp/s6-overlay-noarch.tar.xz \
    && tar -C / -Jxpf /tmp/s6-overlay-${S6_ARCH}.tar.xz \
    && rm -f /tmp/s6-overlay-*.tar.xz \
    && echo "Installed s6-overlay ${S6_OVERLAY_VERSION} successfully."

ENV S6_KEEP_ENV=1 \
    S6_BEHAVIOUR_IF_STAGE2_FAILS=2 \
    S6_SERVICES_GRACETIME=3000 \
    S6_KILL_GRACETIME=3000 \
    S6_VERBOSITY=5

#
# Stage Build Base
#
# Example build command:
#   docker build --tag=gpustack/gpustack:cuda-base-build --file=pack/Dockerfile --target=build-base --progress=plain .
#

FROM nvidia/cuda:${CUDA_DEVEL_VERSION}-cudnn-devel-ubuntu22.04 AS base-build
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    LANG='en_US.UTF-8' \
    LANGUAGE='en_US:en' \
    LC_ALL='en_US.UTF-8'

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
        vim jq bc tree

    # Update locale
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

    # Update timezone
    rm -f /etc/localtime \
        && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
        && echo "Asia/Shanghai" > /etc/timezone \
        && dpkg-reconfigure --frontend noninteractive tzdata

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Upgrade GCC if needed

RUN <<EOF
    # GCC

    # Upgrade GCC if the Ubuntu version is lower than 21.04.
    source /etc/os-release
    if (( $(echo "${VERSION_ID} >= 21.04" | bc -l) )); then
        echo "Skipping GCC upgrade for ${VERSION_ID}..."
        exit 0
    fi

    # Install
    apt-get install -y --no-install-recommends \
        gcc-11 g++-11 gfortran-11 gfortran

    # Update alternatives
    if [[ -f /etc/alternatives/gcov-dump ]]; then update-alternatives --remove-all gcov-dump; fi; update-alternatives --install /usr/bin/gcov-dump gcov-dump /usr/bin/gcov-dump-11 10
    if [[ -f /etc/alternatives/lto-dump ]]; then update-alternatives --remove-all lto-dump; fi; update-alternatives --install /usr/bin/lto-dump lto-dump /usr/bin/lto-dump-11 10
    if [[ -f /etc/alternatives/gcov ]]; then update-alternatives --remove-all gcov; fi; update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-11 10
    if [[ -f /etc/alternatives/gcc ]]; then update-alternatives --remove-all gcc; fi; update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 10
    if [[ -f /etc/alternatives/gcc-nm ]]; then update-alternatives --remove-all gcc-nm; fi; update-alternatives --install /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-11 10
    if [[ -f /etc/alternatives/cpp ]]; then update-alternatives --remove-all cpp; fi; update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-11 10
    if [[ -f /etc/alternatives/g++ ]]; then update-alternatives --remove-all g++; fi; update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 10
    if [[ -f /etc/alternatives/gcc-ar ]]; then update-alternatives --remove-all gcc-ar; fi; update-alternatives --install /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-11 10
    if [[ -f /etc/alternatives/gcov-tool ]]; then update-alternatives --remove-all gcov-tool; fi; update-alternatives --install /usr/bin/gcov-tool gcov-tool /usr/bin/gcov-tool-11 10
    if [[ -f /etc/alternatives/gcc-ranlib ]]; then update-alternatives --remove-all gcc-ranlib; fi; update-alternatives --install /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-11 10
    if [[ -f /etc/alternatives/gfortran ]]; then update-alternatives --remove-all gfortran; fi; update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-11 10

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Install C buildkit

RUN <<EOF
    # C buildkit

    # Install
    apt-get install -y --no-install-recommends \
        make ninja-build pkg-config ccache
    curl --retry 3 --retry-connrefused -fL "https://github.com/Kitware/CMake/releases/download/v3.31.7/cmake-3.31.7-linux-$(uname -m).tar.gz" | tar -zx -C /usr --strip-components 1

    # Install dependencies
    apt-get install -y --no-install-recommends \
        perl-openssl-defaults perl yasm \
        zlib1g zlib1g-dev libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev \
        openssl libssl-dev libsqlite3-dev lcov libomp-dev \
        libblas-dev liblapack-dev libopenblas-dev libblas3 liblapack3 libhdf5-dev \
        libxml2 libxslt1-dev libgl1-mesa-glx libgmpxx4ldbl \
        libncurses5-dev libreadline6-dev libsqlite3-dev \
        liblzma-dev lzma lzma-dev tk-dev uuid-dev libmpdec-dev \
        ffmpeg libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
        libnuma-dev libjemalloc-dev

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Upgrade Python if needed

ARG PYTHON_VERSION

ENV PYTHON_VERSION=${PYTHON_VERSION}

RUN <<EOF
    # Python

    if (( $(echo "$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2) == ${PYTHON_VERSION}" | bc -l) )); then
        echo "Skipping Python upgrade for ${PYTHON_VERSION}..."
        if [[ -z "$(ldconfig -v 2>/dev/null | grep libpython${PYTHON_VERSION})" ]]; then
            PYTHON_LIB_PREFIX=$(python3 -c "import sys; print(sys.base_prefix);")
            echo "${PYTHON_LIB_PREFIX}/lib" >> /etc/ld.so.conf.d/python3.conf
            echo "${PYTHON_LIB_PREFIX}/lib64" >> /etc/ld.so.conf.d/python3.conf
        fi
        exit 0
    fi

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
    if [[ -f /etc/alternatives/python3 ]]; then update-alternatives --remove-all python3; fi; update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
    if [[ -f /etc/alternatives/python ]]; then update-alternatives --remove-all python; fi; update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
    curl -sS "https://bootstrap.pypa.io/get-pip.py" | python${PYTHON_VERSION}
    if [[ -f /etc/alternatives/2to3 ]]; then update-alternatives --remove-all 2to3; fi; update-alternatives --install /usr/bin/2to3 2to3 /usr/bin/2to3${PYTHON_VERSION} 1 || true
    if [[ -f /etc/alternatives/pydoc3 ]]; then update-alternatives --remove-all pydoc3; fi; update-alternatives --install /usr/bin/pydoc3 pydoc3 /usr/bin/pydoc${PYTHON_VERSION} 1 || true
    if [[ -f /etc/alternatives/idle3 ]]; then update-alternatives --remove-all idle3; fi; update-alternatives --install /usr/bin/idle3 idle3 /usr/bin/idle${PYTHON_VERSION} 1 || true
    if [[ -f /etc/alternatives/python3-config ]]; then update-alternatives --remove-all python3-config; fi; update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python${PYTHON_VERSION}-config 1 || true

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Install Python buildkit

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN <<EOF
    # Buildkit

    cat <<EOT >/tmp/requirements.txt
build
cmake<4
ninja<1.11
setuptools<80
setuptools-scm
packaging<25
wheel
pybind11
Cython
psutil==7.0.0
pipx==1.7.1
EOT
    pip install -r /tmp/requirements.txt

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/*
EOF

## Preset this to simplify configuration,
## it is the output of $(pipx environment --value PIPX_LOCAL_VENVS).
ENV PIPX_HOME=/root/.local/share/pipx \
    PIPX_LOCAL_VENVS=/root/.local/share/pipx/venvs \
    USE_EMOJI="false"

## Install Torch

ARG CUDA_VERSION
ARG CUDA_DEVEL_VERSION
ARG TORCH_VERSION

ENV CUDA_HOME="/usr/local/cuda" \
    CUDA_VERSION=${CUDA_VERSION} \
    CUDA_DEVEL_VERSION=${CUDA_DEVEL_VERSION} \
    TORCH_VERSION=${TORCH_VERSION}

ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/compat:${LD_LIBRARY_PATH}"

RUN <<EOF
    # Torch

    # Install
    cat <<EOT >/tmp/requirements.txt
torch==${TORCH_VERSION}
torchvision
torchaudio
EOT
    IFS="." read -r CUDA_MAJOR CUDA_MINOR CUDA_PATCH <<< "${CUDA_DEVEL_VERSION}"
    if (( $(echo "${CUDA_MAJOR}.${CUDA_MINOR} > 12.7" | bc -l) )) || [[ "${TARGETARCH}" == "amd64" ]]; then
        pip install --index-url https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR} \
            -r /tmp/requirements.txt
    else
        pip install --extra-index-url https://download.pytorch.org/whl/cpu/ \
            -r /tmp/requirements.txt
    fi
    pip install \
        numpy scipy

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/*
EOF

#
# Stage FlashInfer Build (linux/amd64 only)
#
# Example build command:
#   docker build --platform=linux/amd64 --tag=gpustack/gpustack:cuda-flashinfer-build --file=pack/Dockerfile --target=flashinfer-build --progress=plain .
#

FROM base-build AS flashinfer-build

ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

ARG TORCH_CUDA_ARCH_LIST
ARG FLASHINFER_VERSION
ARG FLASHINFER_BUILD_MAX_JOBS

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FLASHINFER_VERSION=${FLASHINFER_VERSION} \
    FLASHINFER_BUILD_MAX_JOBS=${FLASHINFER_BUILD_MAX_JOBS}

## Build FlashInfer

RUN <<EOF
    # FlashInfer

    if [[ "${TARGETARCH}" == "arm64" ]]; then
        echo "Skipping FlashInfer build for ${TARGETARCH}..."
        exit 0
    fi

    # Install Dependencies
    cat <<EOT >/tmp/requirements.txt
requests==2.32.5
pynvml
cuda-python
einops
nvidia-nvshmem-cu12
EOT
    pip install \
        -r /tmp/requirements.txt

    # Prepare
    IFS="." read -r CUDA_MAJOR CUDA_MINOR CUDA_PATCH <<< "${CUDA_DEVEL_VERSION}"
    IFS="." read -r TORCH_MAJOR TORCH_MINOR TORCH_PATCH <<< "${TORCH_VERSION}"
    echo "Building FlashInfer ${FLASHINFER_VERSION} wheel for CUDA ${CUDA_MAJOR}.${CUDA_MINOR} and PyTorch ${TORCH_MAJOR}.${TORCH_MINOR}..."

    # Download
    git clone --depth 1 --recursive --shallow-submodules \
        --branch v${FLASHINFER_VERSION} \
        https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer

    # Build
    export MAX_JOBS="${FLASHINFER_BUILD_MAX_JOBS}"
    if [[ -z "${MAX_JOBS}" ]]; then
        export MAX_JOBS="$(nproc)"
    fi
    if (( $(echo "${MAX_JOBS} > 4" | bc -l) )); then
        export MAX_JOBS="4"
    fi
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
    export LD_PRELOAD="${CUDA_HOME}/lib64/libcudart.so:${LD_PRELOAD}"  # Ensure CUDA runtime is preloaded
    echo "Building FlashInfer with the following environment variables:"
    echo "  MAX_JOBS: ${MAX_JOBS}"
    echo "  TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
    echo "  FLASHINFER_LOCAL_VERSION: ${FLASHINFER_LOCAL_VERSION}"
    echo "  LD_PRELOAD=${LD_PRELOAD}"
    pushd /tmp/flashinfer \
      && python -v -m flashinfer.aot \
      && python -v -m build --no-isolation --wheel \
      && tree -hs /tmp/flashinfer/dist \
      && mv /tmp/flashinfer/dist /workspace

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/*
EOF

#
# Stage GPUStack
#
# Example build command:
#   docker build --tag=gpustack/gpustack:cuda --file=pack/Dockerfile --progress=plain .
#

FROM base AS gpustack
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

ARG CUDA_VERSION
ARG CUDA_DEVEL_VERSION
ARG TORCH_VERSION

ENV CUDA_VERSION=${CUDA_VERSION} \
    CUDA_DEVEL_VERSION=${CUDA_DEVEL_VERSION} \
    TORCH_VERSION=${TORCH_VERSION}

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
    IFS="." read -r CUDA_MAJOR CUDA_MINOR CUDA_PATCH <<< "${CUDA_DEVEL_VERSION}"
    pip install ${WHEEL_PACKAGE} --extra-index-url https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}

    if [[ "${TARGETARCH}" == "amd64" ]]; then
        # vllm v0.10.0+ requires xformers==0.0.31, which lacks support for NVIDIA 5090 GPUs when running VL models.
        # Apply downgrade as a temporary solution.
        # See https://github.com/gpustack/gpustack/issues/2769
        pip install --force-reinstall --no-dependencies xformers==0.0.30
    fi

    # Download tools
    gpustack download-tools --device cuda
    tree -hs "$(pip show gpustack | grep Location: | head -n 1 | cut -d" " -f 2)/gpustack/third_party"

    # Set up environment
    mkdir -p /var/lib/gpustack \
        && chmod -R 0755 /var/lib/gpustack

    # Review
    pip freeze

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /workspace/gpustack/dist
EOF

## Install FlashInfer as a Python library for GPUStack if existed

RUN --mount=type=bind,from=flashinfer-build,source=/,target=/flashinfer,rw <<EOF
    # FlashInfer

    if [[ ! -d /flashinfer/workspace ]]; then
        echo "Skipping FlashInfer installation for ${TARGETARCH}..."
        exit 0
    fi

    # Install
    pip install /flashinfer/workspace/*.whl --force-reinstall --no-dependencies

    # Review
    pip freeze

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/*
EOF

# Directory for built-in backend venvs.
ENV LOCAL_VENVS=/opt/venvs

## Install Vox-Box as an independent executor for GPUStack

RUN <<EOF
    # Vox-Box

    # Get version of Vox-Box from GPUStack
    VERSION=$(pip freeze | grep vox_box== | head -n 1 | cut -d'=' -f3)

    # Pre process
    # - Create virtual environment to place vox-box
    python -m venv --system-site-packages ${LOCAL_VENVS}/vox-box
    # - Prepare environment
    source ${LOCAL_VENVS}/vox-box/bin/activate

    # Reinstall Vox-Box,
    # as we create a virtual environment which inherits system site packages,
    # so we need to reinstall Vox-Box to ensure it is installed in the virtual environment.
    # We also lock the transformers version here to fix https://github.com/gpustack/gpustack/pull/2473.
    cat <<EOT >/tmp/requirements.txt
transformers==4.51.3
vox-box==${VERSION}
EOT
    pip install --force-reinstall --no-dependencies -r /tmp/requirements.txt \
        && ln -vsf ${LOCAL_VENVS}/vox-box/bin/vox-box /usr/local/bin/vox-box

    if [[ "${TARGETARCH}" == "amd64" ]]; then
        # Since no compatible version exists for arm64, triton installation is restricted to amd64 architectures,
        # aligning with pyproject.toml constraints.
        pip install --force-reinstall --no-dependencies triton==3.3.1
    fi

    # Download tools
    # - Download dac weights used by audio models like Dia.
    python -m dac download

    # Review
    pip freeze

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/*
EOF

## Setup environment

RUN <<EOF
    # Export pip extra index URL
    IFS="." read -r CUDA_MAJOR CUDA_MINOR CUDA_PATCH <<< "${CUDA_DEVEL_VERSION}"
    EXPORT_PIP_EXTRA_INDEX_URL="export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}"
    echo "${EXPORT_PIP_EXTRA_INDEX_URL}" >> /etc/profile
    echo "${EXPORT_PIP_EXTRA_INDEX_URL}" >> ~/.bashrc
EOF

# 1. Remove cuda paths from LD_LIBRARY_PATH
# 2. Persist pipx venvs in the data directory
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64" \
    PIPX_HOME=/var/lib/gpustack/pipx \
    PIPX_LOCAL_VENVS=/var/lib/gpustack/pipx/venvs \
    PIPX_BIN_DIR=/var/lib/gpustack/bin

ENV NVIDIA_DISABLE_REQUIRE="true" \
    NVIDIA_VISIBLE_DEVICES="all"

COPY pack/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
