"""
Authenticator for server-to-server federation.
"""

import hmac
import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Dict
from fastapi import WebSocket


class Authenticator(ABC):
    """Abstract base class for authenticator implementations."""

    @abstractmethod
    def inject_headers(
        self,
        headers: Dict[str, str],
    ) -> None:
        """Inject auth headers into the headers dict for an outgoing connection."""

    @abstractmethod
    async def authenticate(self, websocket: WebSocket) -> bool:
        """Verify the signature from an incoming connection."""


class HMACAuthenticator(Authenticator):
    """HMAC-SHA256 based authenticator."""

    def __init__(
        self,
        key: str,
        header_key: str = 'x-server-id',
        signature_header: str = 'x-auth-signature',
    ) -> None:
        self.key = key
        self.header_key = header_key
        self.signature_header = signature_header

    def inject_headers(
        self,
        headers: Dict[str, str],
    ) -> None:
        """Inject HMAC auth signature into headers."""
        server_id_str = headers.get(self.header_key, '')
        if server_id_str == '':
            raise ValueError("Missing server ID in headers for HMAC authentication")
        signature = hmac.new(
            self.key.encode(), server_id_str.encode(), hashlib.sha256
        ).hexdigest()
        headers[self.signature_header] = signature

    async def authenticate(self, websocket: WebSocket) -> bool:
        """Verify the signature from an incoming connection."""
        headers = websocket.headers
        provided = headers.get(self.signature_header, '')
        if not provided:
            return False
        server_id_str = headers.get(self.header_key, '')
        if not server_id_str:
            return False
        expected = hmac.new(
            self.key.encode(), server_id_str.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(provided, expected)


class NoOpAuthenticator(Authenticator):
    """Authenticator that accepts all connections (no auth)."""

    def inject_headers(
        self,
        _headers: Dict[str, str],
    ) -> None:
        """No-op: does not inject any headers."""
        pass

    async def authenticate(self, _websocket: WebSocket) -> bool:
        return True


def create_authenticator(key: Optional[str]) -> Authenticator:
    """Factory to create an authenticator based on whether a key is provided."""
    if key:
        return HMACAuthenticator(key)
    return NoOpAuthenticator()
