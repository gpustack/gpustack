ARG CUDA_VERSION=12.4.1

FROM nvidia/cuda:$CUDA_VERSION-runtime-ubuntu22.04

ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    tzdata \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Gets and unpack OpenFst source for linux arm64.
ENV FST_VERSION "1.8.3"
ENV FST_DOWNLOAD_PREFIX "https://www.openfst.org/twiki/pub/FST/FstDownload"
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        cd /tmp && \
        wget -q --no-check-certificate "${FST_DOWNLOAD_PREFIX}/openfst-${FST_VERSION}.tar.gz" && \
        tar -xzf "openfst-${FST_VERSION}.tar.gz" && \
        rm "openfst-${FST_VERSION}.tar.gz"; \
    fi

# Compiles OpenFst for linux arm64..
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        cd "/tmp/openfst-${FST_VERSION}" && \
        ./configure --enable-grm && \
        make install && \
        rm -rd "/tmp/openfst-${FST_VERSION}"; \
    fi

# Gets and unpacks Pynini source.
ENV PYNINI_VERSION "2.1.6"
ENV PYNINI_DOWNLOAD_PREFIX "https://www.opengrm.org/twiki/pub/GRM/PyniniDownload"
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        cd /tmp && \
        wget -q --no-check-certificate "${PYNINI_DOWNLOAD_PREFIX}/pynini-${PYNINI_VERSION}.tar.gz" && \
        tar -xzf "pynini-${PYNINI_VERSION}.tar.gz" && \
        rm "pynini-${PYNINI_VERSION}.tar.gz"; \
    fi

# Installs requirements in all our Pythons.
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        pip install --verbose --upgrade \
        pip -r "/tmp/pynini-${PYNINI_VERSION}/requirements.txt" || exit; \
    fi

# Compiles the wheels to a temporary directory.
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        pip wheel -v "/tmp/pynini-${PYNINI_VERSION}" -w /tmp/wheelhouse/ || exit; \
    fi

# Installs the Pynini.
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        PYNINI_WHEEL_PACKAGE="$(ls /tmp/wheelhouse/*.whl)" && \
        pip install $PYNINI_WHEEL_PACKAGE; \
    fi

# Installs the wetextprocessing.
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        pip install wetextprocessing==1.0.4.1; \
    fi

COPY . /workspace/gpustack
RUN cd /workspace/gpustack && \
    make build

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
    # Install vllm and audio dependencies for x86_64
    WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[all]"; \
    else  \
    WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[audio]"; \
    fi && \
    pip install $WHEEL_PACKAGE &&\
    pip cache purge && \
    rm -rf /workspace/gpustack

RUN gpustack download-tools

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

ENTRYPOINT [ "gpustack", "start" ]
