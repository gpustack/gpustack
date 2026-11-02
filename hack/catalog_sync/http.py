"""Bounded, credential-free acquisition of public research evidence."""

import ipaddress
import json
import os
import re
import socket
from typing import Set
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from hack.catalog_sync.models import require

FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")


def fake_ip_hosts() -> Set[str]:
    """Read explicit local proxy exceptions, restricted to exact DNS hostnames."""
    value = os.environ.get("CATALOG_FAKE_IP_HOSTS", "").strip()
    if not value:
        return set()
    require(
        os.environ.get("GITHUB_ACTIONS") != "true",
        "CATALOG_FAKE_IP_HOSTS is for local testing only; unset it in GitHub Actions",
    )
    hosts = {host.strip().lower().rstrip(".") for host in value.split(",")}
    for host in hosts:
        require(
            bool(
                re.fullmatch(r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}", host)
            ),
            "CATALOG_FAKE_IP_HOSTS requires comma-separated exact hostnames, "
            "without wildcards, IP literals, URLs or ports",
        )
    return hosts


def public_url(url: str) -> None:
    parsed = urlparse(url)
    require(
        parsed.scheme == "https"
        and parsed.hostname
        and not parsed.username
        and not parsed.password
        and parsed.port in (None, 443),
        "Evidence must use a public HTTPS URL",
    )
    allow_fake_ip = parsed.hostname.lower().rstrip(".") in fake_ip_hosts()
    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    ips = [ipaddress.ip_address(row[4][0]) for row in addresses]
    require(
        bool(ips)
        and all(
            ip.is_global or (allow_fake_ip and ip in FAKE_IP_NETWORK) for ip in ips
        ),
        f"Evidence URL resolves to a non-public address: {parsed.hostname}. "
        "For a trusted local fake-IP proxy, add this exact hostname to "
        "CATALOG_FAKE_IP_HOSTS; only 198.18.0.0/15 is supported",
    )


class PublicClient:
    """Read public metadata without forwarding ambient credentials or cookies."""

    def __init__(self):
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=2,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                    respect_retry_after_header=False,
                )
            ),
        )
        self.cache = {}

    def get(self, url: str, limit: int = 4 * 1024 * 1024) -> bytes:
        if url in self.cache:
            data = self.cache[url]
            require(len(data) <= limit, "Evidence exceeds size limit")
            return data
        original = url
        for _ in range(6):
            public_url(url)
            self.session.cookies.clear()
            with self.session.get(
                url, timeout=(10, 45), stream=True, allow_redirects=False
            ) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers["Location"])
                    continue
                response.raise_for_status()
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    require(size <= limit, "Evidence exceeds size limit")
                    chunks.append(chunk)
                data = b"".join(chunks)
                self.cache[original] = data
                return data
        raise ValueError("Too many evidence redirects")

    def json(self, url: str, limit: int = 4 * 1024 * 1024) -> dict:
        return json.loads(self.get(url, limit=limit))
