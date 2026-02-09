# Package logic:
# 1. base target:
#    - Install tools.
#    - Upgrade GCC if needed.
#    - Install C buildkit.
#    - Upgrade Python if needed.
#    - Install Python buildkit.
#    - Install Platform toolkit.
#    - Install S6-overlay.
# 2. gpustack target.
#    - Install PostgreSQL.
#    - Install Higress standalone components.
#    - Install gpustack package from the mounted source code.
#    - Setup entrypoint to gpustack command.

# Argument usage:
# - PYTHON_VERSION: Version of Python to use.
# - GPUSTACK_BASE_IMAGE: Base image for the gpustack stage.
# - GPUSTACK_RUNTIME_ROCM_VERSION: Version of ROCm detection library for gpustack-runtime, update this if project dependencies has changed.
# - GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS: Semicolon-separated list of labels to filter mirrored images when deploying mirrored deployment.
# - HIGRESS_VERSION: Version of Higress to use.
# - HIGRESS_APISERVER_VERSION: Version of Higress API server to use.
ARG PYTHON_VERSION=3.11
ARG GPUSTACK_BASE_IMAGE=base
ARG GPUSTACK_RUNTIME_ROCM_VERSION=7.0.2
ARG GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS
ARG HIGRESS_VERSION=2.1.8
ARG HIGRESS_APISERVER_VERSION=0.0.25
ARG PROMETHEUS_VERSION=3.5.1
ARG GRAFANA_VERSION=12.2.4

# Stage Base
#
# Example build command:
#   docker build --tag=gpustack/gpustack:base --file=pack/Dockerfile --target=base --progress=plain .
#

FROM higress-registry.cn-hangzhou.cr.aliyuncs.com/higress/api-server:${HIGRESS_APISERVER_VERSION} AS apiserver
FROM higress-registry.cn-hangzhou.cr.aliyuncs.com/higress/higress:${HIGRESS_VERSION} AS controller
FROM higress-registry.cn-hangzhou.cr.aliyuncs.com/higress/pilot:${HIGRESS_VERSION} AS pilot
FROM higress-registry.cn-hangzhou.cr.aliyuncs.com/higress/gateway:${HIGRESS_VERSION} AS gateway
FROM ubuntu:22.04@sha256:3c61d3759c2639d4b836d32a2d3c83fa0214e36f195a3421018dbaaf79cbe37f AS base
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

## Install Tools

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
        vim jq bc tree \
        logrotate cron netcat

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
            ldconfig -v
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
    PIP_ROOT_USER_ACTION=ignore \
    PIPX_HOME=/root/.local/share/pipx \
    PIPX_LOCAL_VENVS=/root/.local/share/pipx/venvs \
    POETRY_NO_CACHE=1 \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY="unsafe-best-match"

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
pybind11<3
Cython
psutil
pipx
uv
yq
hatchling
py-spy
poetry
EOT
    pip install -r /tmp/requirements.txt

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/*
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
    S6_VERBOSITY=1

#
# Stage GPUStack
#
# Example build command:
#   docker build --tag=gpustack/gpustack:main --file=pack/Dockerfile --progress=plain .
#

# Vendor ROCm libraries from ROCm base image,
# now only linux/amd64 is supported.
# Must build on linux/amd64 platform.
FROM --platform=${BUILDPLATFORM} rocm/dev-ubuntu-22.04:${GPUSTACK_RUNTIME_ROCM_VERSION} AS rocm-base

FROM ${GPUSTACK_BASE_IMAGE} AS gpustack
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

## Configure data volume
VOLUME /var/lib/gpustack
ENV GPUSTACK_RUNTIME_DOCKER_EPHEMERAL_FILES_DIR="/var/lib/gpustack/cache/gpustack-runtime"

## Install PostgreSQL

ENV PGCONFIG_FILE=/etc/postgresql/main/postgresql.conf \
    POSTGRES_DB=gpustack

RUN set -eux; \
	groupadd -r postgres --gid=9999; \
	useradd -r -g postgres --uid=9999 --home-dir=/var/lib/postgresql --shell=/bin/bash postgres; \
	mkdir -p /var/lib/postgresql; \
	chown -R postgres:postgres /var/lib/postgresql

RUN <<EOF
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

    gosu postgres sed -i "s/^data_directory/#data_directory/" "$PGCONFIG_FILE" \
        && gosu postgres sed -i "s/^hba_file/#hba_file/" "$PGCONFIG_FILE" \
        && gosu postgres sed -i "s/^#log_destination/log_destination/" "$PGCONFIG_FILE" \
        && gosu postgres sed -i "s/^#log_min_messages = warning/log_min_messages = info/" "$PGCONFIG_FILE" \
        && gosu postgres sed -i "s/^#logging_collector = off/logging_collector = on/" "$PGCONFIG_FILE" \
        && gosu postgres sed -i "s/^#log_filename/log_filename/" "$PGCONFIG_FILE" \
        && gosu postgres sed -i "s/^#log_rotation_size/log_rotation_size/" "$PGCONFIG_FILE"

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt

EOF


## Install Higress standalone components

RUN --mount=type=bind,from=apiserver,source=/apiserver,dst=/mnt/apiserver,rw \
    --mount=type=bind,from=controller,source=/usr/local/bin/higress,dst=/mnt/higress,rw \
    --mount=type=bind,from=pilot,source=/usr/local/bin,dst=/mnt/pilot,rw \
    --mount=type=bind,from=gateway,source=/,dst=/mnt/gateway,rw <<EOF
    # Prepare Higress standalone components

    set -eux;
    # Install API server
    cp /mnt/apiserver /usr/local/bin/apiserver;

    # Install controller
    cp /mnt/higress /usr/local/bin/higress;

    # Install pilot
    cp /mnt/pilot/pilot-discovery /usr/local/bin/pilot-discovery;
    cp /mnt/pilot/higress-pilot-start.sh /usr/local/bin/higress-pilot-start.sh;

    # Install gateway
    mkdir -p /var/lib/istio/envoy/
    cp /mnt/gateway/var/lib/istio/envoy/*.json /var/lib/istio/envoy/;
    cp /mnt/gateway/var/lib/istio/envoy/*.so /var/lib/istio/envoy/;
    cp /mnt/gateway/usr/local/bin/pilot-agent /usr/local/bin/pilot-agent;
    cp /mnt/gateway/usr/local/bin/envoy /usr/local/bin/envoy;
    cp /mnt/gateway/usr/local/bin/higress-proxy-container-init.sh /usr/local/bin/;
    cp /mnt/gateway/usr/local/bin/supercronic-linux-${TARGETARCH} /usr/local/bin/;
    ln -s supercronic-linux-${TARGETARCH} supercronic && mv supercronic /usr/local/bin/;

    # Prepare Higress
    chmod a+x /usr/local/bin/higress-proxy-container-init.sh; \
    sed -i 's/1337/0/g' /usr/local/bin/higress-proxy-container-init.sh; \
    /usr/local/bin/higress-proxy-container-init.sh

EOF
# Initialize configurations
COPY pack/rootfs/ /
COPY docker-compose/grafana/grafana_dashboards/ /etc/dashboards/
ENV GPUSTACK_HIGRESS_PLUGIN_DIR=/usr/local/share/gpustack-higress/plugins
## END Install Higress standalone components

## Install Skopeo

ARG GOLANG_INSTALL_MIRROR="https://go.dev/dl"
ARG GOPROXY="https://proxy.golang.org"
RUN <<EOF
    # Skopeo

    REQUIRED_SKOPEO_VERSION=1.13.3
    # Install Skopeo for images mirroring
    apt-get update -y && apt-get install -y --no-install-recommends \
        skopeo

    IFS="." read -r MAJOR MINOR PATCH <<< "$(skopeo --version | awk '{print $3}' | cut -d'-' -f1)"
    if (( MAJOR < 1 || ( MAJOR == 1 && MINOR < 14 ) )); then
        GOLANG_VERSION=1.18.10
        echo "Skopeo version ${MAJOR}.${MINOR} is lower than required ${REQUIRED_SKOPEO_VERSION}, upgrading from source..."

        apt-get install -y --no-install-recommends \
            libgpgme-dev \
            libdevmapper-dev

        # Install Go
        curl --retry 3 --retry-connrefused -fL "${GOLANG_INSTALL_MIRROR}/go${GOLANG_VERSION}.${TARGETOS}-${TARGETARCH}.tar.gz" | tar -zx -C /usr/local
        export PATH="/usr/local/go/bin:${PATH}"

        # Install Skopeo from source
        git -C /tmp clone --recursive --shallow-submodules \
            --depth 1 --branch v${REQUIRED_SKOPEO_VERSION} --single-branch \
            https://github.com/containers/skopeo.git skopeo

        # Build and install
        pushd /tmp/skopeo \
            && sed -i "s#export GOPROXY=https://proxy.golang.org#export GOPROXY=${GOPROXY}#g" Makefile\
            && make vendor \
            && DISABLE_DOCS=1 PREFIX=/usr make install-binary

        # Review
        skopeo --version

        # Cleanup
        go clean -cache -modcache -testcache \
            && rm -rf /usr/local/go \
            && rm -rf /root/.cache/go-build
    fi

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Install Prometheus and Grafana

ARG PROMETHEUS_VERSION
ARG GRAFANA_VERSION
RUN <<EOF
    set -eux

    apt-get update -y && apt-get install -y --no-install-recommends \
        fontconfig

    case "${TARGETARCH}" in
        amd64)
            PROM_ARCH="amd64"
            GRAFANA_ARCH="amd64"
            ;;
        arm64)
            PROM_ARCH="arm64"
            GRAFANA_ARCH="arm64"
            ;;
        *)
            echo "Unsupported TARGETARCH: ${TARGETARCH}"
            exit 1
            ;;
    esac

    curl --retry 3 --retry-connrefused -fL \
        "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-${PROM_ARCH}.tar.gz" \
        | tar -zx -C /opt
    mv "/opt/prometheus-${PROMETHEUS_VERSION}.linux-${PROM_ARCH}" /opt/prometheus
    ln -s /opt/prometheus/prometheus /usr/local/bin/prometheus
    ln -s /opt/prometheus/promtool /usr/local/bin/promtool

    curl --retry 3 --retry-connrefused -fL \
        "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-${GRAFANA_ARCH}.tar.gz" \
        | tar -zx -C /opt
    mv "/opt/grafana-${GRAFANA_VERSION}" /opt/grafana
    ln -s /opt/grafana/bin/grafana-server /usr/local/bin/grafana-server
    ln -s /opt/grafana/bin/grafana-cli /usr/local/bin/grafana-cli

    mkdir -p /etc/grafana
    cp /opt/grafana/conf/sample.ini /etc/grafana/grafana.ini.sample

    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Install GPUStack

RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,target=/workspace/gpustack,rw \
    --mount=type=bind,from=ghcr.io/oras-project/oras:v1.3.0,source=/bin/oras,dst=/bin/oras <<EOF
    # Install GPUStack

    export POETRY_NO_CACHE=0
    export UV_NO_CACHE=0
    export UV_SYSTEM_PYTHON=1
    export UV_LINK_MODE=copy

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

    uv pip install --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cpu/ \
        ${WHEEL_PACKAGE}

    # Download tools
    gpustack download-tools
    tree -hs "$(pip show gpustack | grep Location: | head -n 1 | cut -d" " -f 2)/gpustack/third_party"

    # Set up environment
    mkdir -p /var/lib/gpustack \
        && chmod -R 0755 /var/lib/gpustack

    # Review
    uv pip tree \
        --package gpustack
    gpustack version

    # Download wasm-plugins
    export ORAS_CACHE=/root/cache/oras
    python gpustack/gateway/pull_plugins.py

    # Try to update PCI IDs
    if ! update-pciids; then
        curl -o /usr/share/misc/pci.ids https://pci-ids.ucw.cz/v2.2/pci.ids || true
    fi

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /workspace/gpustack/dist
EOF

## Entrypoint

## Active all AMD devices detection,
## works with (default) ROCm container runtime and privileged mode.
## See https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-py-api.html.
## Runs:
## - With container runtime installed:
##   + If installed AMD contaienr runtime as default runtime, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged ...
##   + If there are mulitple container runtimes installed, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged --runtime amd ...
##   + If failed to detect devices' name, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged -v /usr/share:/usr/share:ro ...
##   + If want to detect the correct host ROCm version, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged -v /opt/rocm:/opt/rocm:ro ...
##   + Disallowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock ...
## - Without container runtime installed:
##   + Allowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged -v /opt/rocm:/opt/rocm:ro ...
##   + Disallowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --security-opt seccomp=unconfined -v /dev:/dev:ro --group-add video -v /opt/rocm:/opt/rocm:ro ...
RUN --mount=type=bind,from=rocm-base,source=/opt/rocm/share,target=/opt/rocm/share,rw <<EOF
    # Reinstall amd-smi

    export UV_SYSTEM_PYTHON=1
    export UV_PRERELEASE=allow
    uv pip install --no-build-isolation \
        /opt/rocm/share/amd_smi
    uv pip tree

    # Hack to avoid: Fail to open libdrm_amdgpu.so: libdrm_amdgpu.so: cannot open shared object file: No such file or directory
    TARGET_DIR="/usr/lib/$(uname -m)-linux-gnu"
    TARGET_LIB="libdrm_amdgpu.so.1.0.0"
    TARGET_LINK="libdrm_amdgpu.so"
    TARGET_LIB_EXISTED="true"
    if [[ ! -e "${TARGET_DIR}/${TARGET_LIB}" ]]; then
        TARGET_LIB_EXISTED="false"
        touch "${TARGET_DIR}/${TARGET_LIB}"
    fi
    pushd "${TARGET_DIR}" \
        && ln -sf "${TARGET_LIB}" "${TARGET_LINK}"
    if [[ "${TARGET_LIB_EXISTED}" == "false" ]]; then
        rm -f "${TARGET_DIR}/${TARGET_LIB}"
    fi
EOF
ENV AMD_VISIBLE_DEVICES="all" \
    GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES="/opt/rocm"

## Active all Ascend devices detection,
## works with (default) Ascend container runtime and privileged mode.
## See https://gitcode.com/Ascend/mind-cluster/blob/master/component/ascend-common/devmanager/dcmi/dcmi_interface_api.h.
## Runs:
## - With container runtime installed:
##   + If installed Ascend container runtime as default runtime, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged -e "ASCEND_VISIBLE_DEVICES=$(npu-smi info -m | tail -n 1 | awk '{print $1}')" ...
##   + If there are mulitple container runtimes installed, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged -e "ASCEND_VISIBLE_DEVICES=$(npu-smi info -m | tail -n 1 | awk '{print $1}')" --runtime ascend ...
##   + If want to detect the correct host CANN version and SoC name, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged -e "ASCEND_VISIBLE_DEVICES=$(npu-smi info -m | tail -n 1 | awk '{print $1}')" -v /usr/local/Ascend/ascend-toolkit:/usr/local/Ascend/ascend-toolkit:ro ...
##   + Disallowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock -e "ASCEND_VISIBLE_DEVICES=$(npu-smi info -m | grep -v mcu | awk '{if(NR>1){print $1}}' | uniq | paste -sd ',')" ...
## - Without container runtime installed:
##   + Allowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged -v /usr/local/dcmi:/usr/local/dcmi:ro -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /etc/hccn.conf:/etc/hccn.conf:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro ...
##   + Disallowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --security-opt seccomp=unconfined -v /dev:/dev:ro -v /usr/local/dcmi:/usr/local/dcmi:ro -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /etc/hccn.conf:/etc/hccn.conf:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro ...
ENV ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest" \
    LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:${LD_LIBRARY_PATH}" \
    GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES="/usr/local/Ascend/ascend-toolkit;${GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES}"

## Active all Cambricon devices detection,
## works with (default) Cambricon container runtime and privileged mode.
## See https://github.com/Cambricon/cambricon-k8s-device-plugin/blob/master/device-plugin/pkg/cndev/include/cndev.h,
##     https://github.com/Cambricon/cambricon-k8s-device-plugin/blob/master/device-plugin/pkg/cntopo/include/cntopo.h.
## Runs:
## - With container runtime installed:
##   [TODO, TBD]
## - Without container runtime installed:
##   + Allowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged -v /usr/local/neuware:/usr/local/neuware:ro -v /usr/bin/cnmon:/usr/bin/cnmon ...
##   + Disallowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --security-opt seccomp=unconfined -v /dev:/dev:ro -v /usr/local/neuware:/usr/local/neuware:ro -v /usr/bin/cnmon:/usr/bin/cnmon ...
ENV CAMBRICON_VISIBLE_DEVICES="all" \
    NEUWARE_HOME="/usr/local/neuware" \
    LD_LIBRARY_PATH="/usr/local/neuware/lib64:${LD_LIBRARY_PATH}" \
    GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES="/usr/local/neuware;${GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES}"

## Active all Hygon devices detection,
## works with (default) Hygon container runtime and privileged mode.
## See https://github.com/Project-HAMi/dcu-dcgm/blob/master/pkg/dcgm/include/rocm_smi.h.
## Runs:
## - With container runtime installed:
##   [TODO, TBD]
## - Without container runtime installed:
##   + Allowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged -v /opt/hyhal:/opt/hyhal:ro -v /opt/dtk:/opt/dtk:ro -e ROCM_SMI_LIB_PATH=/opt/hyhal/lib -e ROCM_PATH=/opt/dtk ...
##   + Disallowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --security-opt seccomp=unconfined -v /dev:/dev:ro --group-add video -v /opt/hyhal:/opt/hyhal:ro -v /opt/dtk:/opt/dtk:ro -e ROCM_SMI_LIB_PATH=/opt/hyhal/lib -e ROCM_PATH=/opt/dtk ...
ENV HYGON_VISIBLE_DEVICES="all" \
    GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES="/opt/dtk;${GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES}"

## Active all Iluvatar devices detection,
## works with (default) Iluvatar container runtime and privileged mode.
## See https://github.com/Deep-Spark/ix-container-toolkit.
## Runs:
## - With container runtime installed:
##   + If installed Iluvatar container runtime as default runtime, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged ...
##   + If there are mulitple container runtimes installed, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged --runtime iluvatar ...
##   + Disallowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock ...
## - Without container runtime installed:
##   + Allowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged -v /usr/local/corex:/usr/local/corex:ro ...
##   + Disallowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --security-opt seccomp=unconfined -v /dev:/dev:ro -v /usr/local/corex:/usr/local/corex:ro ...
ENV IX_VISIBLE_DEVICES="all" \
    COREX_HOME="/usr/local/corex" \
    LD_LIBRARY_PATH="/usr/local/corex/lib64:${LD_LIBRARY_PATH}" \
    GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES="/usr/local/corex;${GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES}"

## Active all MetaX devices detection,
## works with (default) MetaX container runtime and privileged mode.
## See https://developer.metax-tech.com/api/client/document/preview/626/k8s/03_component.html#container-runtime.
## Runs:
## - With container runtime installed:
##   [TODO, TBD]
## - Without container runtime installed:
##   + Allowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged -v /opt/mxdriver:/opt/mxdriver:ro -v /opt/maca:/opt/maca:ro ...
##   + Disallowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --security-opt seccomp=unconfined -v /dev:/dev:ro -v /opt/mxdriver:/opt/mxdriver:ro -v /opt/maca:/opt/maca:ro ...
ENV LD_LIBRARY_PATH="/opt/maca/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH}" \
    GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES="/opt/maca;${GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES}"

## Active all MThreads devices detection,
## works with (default) MThreads container runtime and privileged mode.
## See https://docs.mthreads.com/cloud-native/cloud-native-doc-online/install_guide.
## Runs:
## - With container runtime installed:
##   + If installed MThreads container runtime as default runtime, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged ...
##   + If there are mulitple container runtimes installed, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged --runtime mthreads ...
##   + Disallowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock ...
## - Without container runtime installed:
##   [TODO, TBD]
ENV MTHREADS_VISIBLE_DEVICES="all" \
    MTHREADS_DRIVER_CAPABILITIES="compute,utility"

## Active all NVIDIA devices detection,
## works with (default) NVIDIA container runtime and privileged mode.
## See https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html#nvml-api-reference.
## Runs:
## - With container runtime installed:
##   + If installed NVIDIA container runtime as default runtime, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged ...
##   + If there are mulitple container runtimes installed, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --privileged --runtime nvidia ...
##   + Disallowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock ...
## - Without container runtime installed:
##   [TODO, TBD]
ENV NVIDIA_DISABLE_REQUIRE="true" \
    NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility"

## Active all T-Head devices detection,
## works with (default) T-Head container runtime and privileged mode.
## See https://help.aliyun.com/document_detail/2996754.html.
## Runs:
## - With container runtime installed:
##   [TODO, TBD]
## - Without container runtime installed:
##   + Allowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock -e GPUSTACK_RUNTIME_DOCKER_RESOURCE_INJECTION_POLICY=CDI -v /var/run/cdi:/var/run/cdi --privileged -v /usr/local/PPU_SDK:/usr/local/PPU_SDK:ro ...
##   + Disallowing privileged, try with:
##     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock -e GPUSTACK_RUNTIME_DOCKER_RESOURCE_INJECTION_POLICY=CDI -v /var/run/cdi:/var/run/cdi --security-opt seccomp=unconfined -v /dev:/dev:ro -v /usr/local/PPU_SDK:/usr/local/PPU_SDK:ro ...
ENV PPU_HOME="/usr/local/PPU_SDK" \
    LD_LIBRARY_PATH="/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH}" \
    GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES="/usr/local/PPU_SDK;${GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES}"

## Active GPUStack runtime mirrored deployment mode,
## if getting an error like, "Found multiple Containers with the same hostname ...",
## please use `--env GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME=...` to specify the exact container name.
##
ARG GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS

## GPUSTACK_RUNTIME_LOG_EXCEPTION=false: Disable logging exceptions from gpustack-runtime.
## GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY=/var/run/cdi: Set CDI specs directory.
## GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT=true: Enable mirrored deployment mode.
## GPUSTACK_RUNTIME_DEPLOY_CORRECT_RUNNER_IMAGE=false: Disable auto correction of runner images.
## GPUSTACK_RUNTIME_DOCKER_IMAGE_NO_PULL_VISUALIZATION=true: Disable visualizating image pull progress, instead using simple logs.
## GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS: Configure filter labels for mirrored deployment.
## GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES: Declare volumes to be ignored during mirrored deployment.
ENV GPUSTACK_RUNTIME_LOG_EXCEPTION="false" \
    GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY="/var/run/cdi" \
    GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT="true" \
    GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES="/var/run/cdi;${GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES}" \
    GPUSTACK_RUNTIME_DEPLOY_CORRECT_RUNNER_IMAGE="false" \
    GPUSTACK_RUNTIME_DOCKER_IMAGE_NO_PULL_VISUALIZATION="true" \
    GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS="${GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS}"

COPY --chmod=755 pack/entrypoint.sh /usr/bin/entrypoint.sh

WORKDIR /
ENTRYPOINT [ "/usr/bin/entrypoint.sh" ]
