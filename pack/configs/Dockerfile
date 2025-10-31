ARG HUB=higress-registry.cn-hangzhou.cr.aliyuncs.com/higress
ARG BASE_VERSION=2022-10-27T19-02-22
ARG CORE_VERSION=2.1.8
ARG APISERVER_VERSION=0.0.25

FROM ${HUB}/api-server:${APISERVER_VERSION} AS apiserver
FROM ${HUB}/higress:${CORE_VERSION} AS controller
FROM ${HUB}/pilot:${CORE_VERSION} AS pilot
FROM ${HUB}/gateway:${CORE_VERSION} AS gateway
FROM ${HUB}/base:${BASE_VERSION}

# Install API server
COPY --from=apiserver /apiserver /usr/local/bin/apiserver

# Install controller
COPY --from=controller /usr/local/bin/higress /usr/local/bin/higress

# Install pilot
COPY --from=pilot /usr/local/bin/pilot-discovery /usr/local/bin/pilot-discovery
COPY --from=pilot /usr/local/bin/higress-pilot-start.sh /usr/local/bin/higress-pilot-start.sh

# Install gateway
COPY --from=gateway /var/lib/istio/envoy/*.json /var/lib/istio/envoy/
COPY --from=gateway /var/lib/istio/envoy/*.so /var/lib/istio/envoy/
COPY --from=gateway /usr/local/bin/pilot-agent /usr/local/bin/pilot-agent
COPY --from=gateway /usr/local/bin/envoy /usr/local/bin/envoy
COPY --from=gateway /usr/local/bin/higress-proxy-*.sh /usr/local/bin/
RUN chmod a+x /usr/local/bin/higress-proxy-container-init.sh; \
  sed -i 's/1337/0/g' /usr/local/bin/higress-proxy-container-init.sh; \
  /usr/local/bin/higress-proxy-container-init.sh
COPY --from=gateway /usr/local/bin/supercronic* /usr/local/bin/

# Install supervisord, logrotate, cron and initialize related folders
RUN arch="$(dpkg --print-architecture)"; arch="${arch##*-}"; \
  apt-get update --allow-unauthenticated; \
  apt-get install --no-install-recommends -y --allow-unauthenticated \
    wget supervisor logrotate cron; \
  apt-get upgrade -y --allow-unauthenticated; \
  apt-get clean; \
  rm -rf /var/log/*log /var/lib/apt/lists/* /var/log/apt/* /var/lib/dpkg/*-old /var/cache/debconf/*-old; \
  wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_$arch -O /usr/local/bin/yq && chmod +x /usr/local/bin/yq; \
  mkdir -p /var/log/higress; \
  mkdir /var/lib/gpustack;

COPY ./supervisord/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Initialize configurations
COPY ./meshConfig /etc/istio/config
COPY ./gateway/podinfo /etc/istio/pod
COPY ./scripts /usr/local/bin
COPY ./config /opt/data/defaultConfig

EXPOSE 80 443

ENTRYPOINT ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
