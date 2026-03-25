#!/usr/bin/env python3
"""
Tests for patricia_trie.py - CIDR Registry using py-radix
"""

import uuid
from gpustack.websocket_proxy.patricia_trie import CIDRRegistry


class TestCIDRRegistry:
    """Tests for CIDRRegistry class."""

    def test_insert_and_find_exact_match(self):
        """Test basic insert and find for a single CIDR."""
        registry = CIDRRegistry()
        client_id = uuid.uuid4()

        registry.insert("10.0.0.0/8", client_id)
        result = registry.find_best_match("10.0.0.1")

        assert result == client_id

    def test_longest_prefix_match(self):
        """Test that more specific CIDRs take precedence over less specific ones."""
        registry = CIDRRegistry()
        client_class_a = uuid.uuid4()
        client_class_b = uuid.uuid4()

        registry.insert("10.0.0.0/8", client_class_a)
        registry.insert("10.1.0.0/16", client_class_b)

        # 10.1.x.x should match /16 (more specific)
        assert registry.find_best_match("10.1.0.1") == client_class_b
        # 10.0.x.x should match /8
        assert registry.find_best_match("10.0.0.1") == client_class_a
        # 10.5.x.x should match /8
        assert registry.find_best_match("10.5.5.5") == client_class_a

    def test_no_match(self):
        """Test that unmatched IPs return None."""
        registry = CIDRRegistry()
        client_id = uuid.uuid4()

        registry.insert("10.0.0.0/8", client_id)

        assert registry.find_best_match("192.168.1.1") is None
        assert registry.find_best_match("172.16.0.1") is None

    def test_default_route(self):
        """Test that 0.0.0.0/0 matches any IP."""
        registry = CIDRRegistry()
        default_client = uuid.uuid4()
        specific_client = uuid.uuid4()

        registry.insert("0.0.0.0/0", default_client)
        registry.insert("10.0.0.0/8", specific_client)

        assert registry.find_best_match("10.0.0.1") == specific_client
        assert registry.find_best_match("192.168.1.1") == default_client
        assert registry.find_best_match("8.8.8.8") == default_client

    def test_multiple_cidrs_same_client(self):
        """Test that the same client can have multiple CIDRs."""
        registry = CIDRRegistry()
        client_id = uuid.uuid4()

        registry.insert("10.0.0.0/8", client_id)
        registry.insert("172.16.0.0/12", client_id)

        assert registry.find_best_match("10.5.5.5") == client_id
        assert registry.find_best_match("172.16.0.1") == client_id
        assert registry.find_best_match("192.168.1.1") is None

    def test_exact_host_match(self):
        """Test /32 exact host match takes precedence over /24."""
        registry = CIDRRegistry()
        class_c_client = uuid.uuid4()
        host_client = uuid.uuid4()

        registry.insert("192.168.1.0/24", class_c_client)
        registry.insert("192.168.1.100/32", host_client)

        assert registry.find_best_match("192.168.1.100") == host_client
        assert registry.find_best_match("192.168.1.99") == class_c_client
        assert registry.find_best_match("192.168.1.101") == class_c_client

    def test_ipv6_support(self):
        """Test IPv6 CIDR matching."""
        registry = CIDRRegistry()
        client1 = uuid.uuid4()
        client2 = uuid.uuid4()

        registry.insert("2001:db8::/32", client1)
        registry.insert("2001:db8:1::/48", client2)

        assert registry.find_best_match("2001:db8::1") == client1
        assert registry.find_best_match("2001:db8:ffff::1") == client1
        assert registry.find_best_match("2001:db8:1::1") == client2
        assert registry.find_best_match("2001:db8:2::1") == client1
        assert registry.find_best_match("2001:dead::1") is None

    def test_remove_client(self):
        """Test removing a client's all CIDRs."""
        registry = CIDRRegistry()
        client1 = uuid.uuid4()
        client2 = uuid.uuid4()

        registry.insert("10.0.0.0/8", client1)
        registry.insert("10.1.0.0/16", client1)
        registry.insert("192.168.0.0/16", client2)

        # Verify both client1 CIDRs work
        assert registry.find_best_match("10.0.0.1") == client1
        assert registry.find_best_match("10.1.0.1") == client1

        # Remove client1
        registry.remove_client(client1)

        # client1's CIDRs should no longer match
        assert registry.find_best_match("10.0.0.1") is None
        assert registry.find_best_match("10.1.0.1") is None
        # client2 should still work
        assert registry.find_best_match("192.168.1.1") == client2

    def test_update_client(self):
        """Test updating a client's CIDRs."""
        registry = CIDRRegistry()
        client_id = uuid.uuid4()

        registry.insert("10.0.0.0/8", client_id)
        assert registry.find_best_match("10.0.0.1") == client_id
        assert registry.find_best_match("172.16.0.1") is None

        # Update client's CIDRs
        registry.update_client(client_id, ["172.16.0.0/12"])

        # Old CIDR should not match anymore
        assert registry.find_best_match("10.0.0.1") is None
        # New CIDR should match
        assert registry.find_best_match("172.16.0.1") == client_id

    def test_empty_registry(self):
        """Test that empty registry returns None for any IP."""
        registry = CIDRRegistry()

        assert registry.find_best_match("10.0.0.1") is None
        assert registry.find_best_match("192.168.1.1") is None
        assert registry.find_best_match("::1") is None

    def test_invalid_ip(self):
        """Test that invalid IP returns None."""
        registry = CIDRRegistry()
        client_id = uuid.uuid4()

        registry.insert("10.0.0.0/8", client_id)

        assert registry.find_best_match("not-an-ip") is None
        assert registry.find_best_match("") is None

    def test_complex_overlapping_cidrs(self):
        """Test complex overlapping CIDR scenarios."""
        registry = CIDRRegistry()
        c1 = uuid.uuid4()
        c2 = uuid.uuid4()
        c3 = uuid.uuid4()
        c4 = uuid.uuid4()

        registry.insert("0.0.0.0/0", c1)
        registry.insert("10.0.0.0/8", c2)
        registry.insert("10.1.0.0/16", c3)
        registry.insert("10.1.1.0/24", c4)

        tests = [
            ("1.1.1.1", c1),
            ("9.9.9.9", c1),
            ("10.0.0.1", c2),
            ("10.0.255.255", c2),
            ("10.1.0.1", c3),
            ("10.1.0.255", c3),
            ("10.1.1.0", c4),
            ("10.1.1.1", c4),
            ("10.1.1.255", c4),
            ("10.1.2.0", c3),
            ("10.2.0.0", c2),
        ]

        for ip, expected in tests:
            result = registry.find_best_match(ip)
            assert result == expected, f"IP {ip}: expected {expected}, got {result}"
