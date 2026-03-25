"""
Tests for the authenticator module.
"""

import pytest
import uuid
from unittest.mock import MagicMock

from gpustack.websocket_proxy.authenticator import (
    Authenticator,
    HMACAuthenticator,
    NoOpAuthenticator,
    create_authenticator,
)


def create_mock_websocket(headers: dict) -> MagicMock:
    """Create a mock WebSocket with the given headers."""
    mock_ws = MagicMock()
    mock_ws.headers = headers
    return mock_ws


class TestAuthenticatorClass:
    """Tests for Authenticator class"""

    @pytest.mark.asyncio
    async def test_authenticate_valid_signature(self):
        """Test that authenticate returns True for valid signature"""
        auth = HMACAuthenticator("secret-key")
        server_id = uuid.uuid4()

        # Generate headers with signature
        headers = {'x-server-id': str(server_id)}
        auth.inject_headers(headers)

        # Authenticate should succeed
        mock_ws = create_mock_websocket(headers)
        assert await auth.authenticate(mock_ws) is True

    @pytest.mark.asyncio
    async def test_authenticate_invalid_signature(self):
        """Test that authenticate returns False for invalid signature"""
        auth = HMACAuthenticator("secret-key")
        server_id = uuid.uuid4()
        headers = {
            'x-server-id': str(server_id),
            'x-auth-signature': 'invalid-signature',
        }

        mock_ws = create_mock_websocket(headers)
        assert await auth.authenticate(mock_ws) is False

    @pytest.mark.asyncio
    async def test_authenticate_missing_signature(self):
        """Test that authenticate returns False when signature is missing"""
        auth = HMACAuthenticator("secret-key")
        server_id = uuid.uuid4()
        headers = {
            'x-server-id': str(server_id),
        }

        mock_ws = create_mock_websocket(headers)
        assert await auth.authenticate(mock_ws) is False

    @pytest.mark.asyncio
    async def test_authenticate_wrong_key(self):
        """Test that authenticate fails with wrong key"""
        auth1 = HMACAuthenticator("secret-key")
        auth2 = HMACAuthenticator("wrong-key")
        server_id = uuid.uuid4()

        headers = {'x-server-id': str(server_id)}
        auth1.inject_headers(headers)

        mock_ws = create_mock_websocket(headers)
        # auth2 should reject headers signed by auth1
        assert await auth2.authenticate(mock_ws) is False

    @pytest.mark.asyncio
    async def test_authenticate_wrong_server_id(self):
        """Test that authenticate fails when server_id header is tampered"""
        auth = HMACAuthenticator("secret-key")
        server_id_a = uuid.uuid4()
        server_id_b = uuid.uuid4()
        headers = {'x-server-id': str(server_id_a)}
        auth.inject_headers(headers)

        # Tamper with x-server-id header
        headers['x-server-id'] = str(server_id_b)

        mock_ws = create_mock_websocket(headers)
        # Should fail since signature was computed for different server_id
        assert await auth.authenticate(mock_ws) is False


class TestNoOpAuthenticator:
    """Tests for NoOpAuthenticator class"""

    def test_inject_headers_no_signature(self):
        """Test that NoOpAuthenticator injects no headers (auth headers should be injected by caller)."""
        auth = NoOpAuthenticator()
        server_id = uuid.uuid4()
        headers = {'x-server-id': str(server_id)}
        auth.inject_headers(headers)

        # No auth headers should be added
        assert headers == {'x-server-id': str(server_id)}

    @pytest.mark.asyncio
    async def test_authenticate_always_true(self):
        """Test that NoOpAuthenticator always returns True"""
        auth = NoOpAuthenticator()

        assert await auth.authenticate(create_mock_websocket({})) is True
        assert (
            await auth.authenticate(
                create_mock_websocket({'x-auth-signature': 'anything'})
            )
            is True
        )
        assert (
            await auth.authenticate(create_mock_websocket({'x-server-id': 'server'}))
            is True
        )


class TestCreateAuthenticator:
    """Tests for create_authenticator factory"""

    def test_with_secret_returns_authenticator(self):
        """Test that create_authenticator returns Authenticator when secret is provided"""
        auth = create_authenticator("my-secret")
        assert isinstance(auth, Authenticator)

    def test_without_secret_returns_noop(self):
        """Test that create_authenticator returns NoOpAuthenticator when secret is None"""
        auth = create_authenticator(None)
        assert isinstance(auth, NoOpAuthenticator)

    def test_with_empty_secret_returns_noop(self):
        """Test that create_authenticator returns NoOpAuthenticator when secret is empty"""
        auth = create_authenticator("")
        assert isinstance(auth, NoOpAuthenticator)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
