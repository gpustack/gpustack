#!/usr/bin/env python3
"""
CIDR Registry using py-radix for efficient longest prefix match (LPM) lookups.

Provides O(k) lookup where k = address bits (32 for IPv4, 128 for IPv6),
using the py-radix library for production-ready Patricia Trie implementation.

Radix Tree Organization (Patricia Trie):
=========================================

A radix tree is a compressed prefix tree that stores network prefixes.
Each node represents a bit position in the binary representation of an IP address.

Structure:
-----------
- Root node: represents the start of all addresses
- Each edge: labeled with 0 or 1 (a single bit)
- Each leaf node: represents a complete network prefix (CIDR)

Example - inserting 10.0.0.0/8 and 10.1.0.0/16:
------------------------------------------------

              [root]
             /      \\
           0         1
            \\        \\
         [10.x.x.x]  [other]
              \\
             ... (compressed path for /8)
              \\
            [node at bit 16, prefix=10.1.x.x, client_id=client2]
                              \\
                            ... (compressed path for /16)

Key Properties:
----------------
1. Longest Prefix Match (LPM): When searching for an IP, the tree traversal
   continues until no matching child exists. The last node with a valid
   prefix that matches the search key is the best match.

2. Compression: Patricia trie compresses chains of single-child nodes into
   single nodes, reducing space complexity from O(k*n) to O(k) where k is
   address bits and n is number of prefixes.

3. search_best(): Traverses from root following bits of the IP address.
   Returns the most specific (longest) matching prefix.

Lookup Example for IP 10.1.5.5:
--------------------------------
- Binary of 10.1.5.5: 00001010 00000001 00000101 00000101
- Inserted prefixes: 10.0.0.0/8, 10.1.0.0/16

1. Start at root
2. Follow bit 0 (first bit of 10) -> child exists
3. Continue following bits 0,0,0,0,1,0,1,0 (first 8 bits = /8)
   - At position 8, /8 node has client_id=client1, but we continue...
4. Continue with bits for second octet (00000001 = 0,0,0,0,0,0,0,1)
5. At position 16, /16 node has client_id=client2 (more specific!)
6. Try to follow bit at position 16, but /16 is exact match, stop
7. Return client2 (the longest matching prefix)

Memory Layout:
---------------
RadixNode {
    prefix: str      # e.g., "10.0.0.0/8"
    prefixlen: int   # e.g., 8
    packed: bytes   # binary representation of network address
    family: int     # 2 for IPv4, 10 for IPv6
    data: dict       # user data ({"client_id": uuid})
    children: dict   # {0: child_node, 1: child_node}
    parent: node     # pointer to parent node
}
"""

import radix
import uuid
from typing import Optional, Dict, List


class CIDRRegistry:
    """
    Registry that maps CIDR ranges to client IDs using py-radix.

    This provides efficient longest-prefix-match lookups for IP addresses.
    """

    def __init__(self):
        self._tree = radix.Radix()
        # Track all CIDRs per client for rebuild purposes
        self._client_cidrs: Dict[uuid.UUID, List[str]] = {}

    def insert(self, cidr: str, client_id: uuid.UUID) -> None:
        """Insert a CIDR for a client."""
        node = self._tree.add(cidr)
        node.data["client_id"] = client_id

        if client_id not in self._client_cidrs:
            self._client_cidrs[client_id] = []
        if cidr not in self._client_cidrs[client_id]:
            self._client_cidrs[client_id].append(cidr)

    def remove_client(self, client_id: uuid.UUID) -> None:
        """Remove all CIDRs associated with a client."""
        if client_id in self._client_cidrs:
            del self._client_cidrs[client_id]
        self._rebuild()

    def update_client(self, client_id: uuid.UUID, cidrs: List[str]) -> None:
        """Update all CIDRs for a client."""
        self._client_cidrs[client_id] = list(cidrs)
        self._rebuild()

    def find_best_match(self, ip: str) -> Optional[uuid.UUID]:
        """Find the best matching client for an IP address."""
        try:
            node = self._tree.search_best(ip)
            if node:
                return node.data.get("client_id")
        except (ValueError, OSError):
            # Invalid IP format
            pass
        return None

    def _rebuild(self) -> None:
        """Rebuild the tree from the client_cidrs mapping."""
        self._tree = radix.Radix()
        for client_id, cidrs in self._client_cidrs.items():
            for cidr in cidrs:
                node = self._tree.add(cidr)
                node.data["client_id"] = client_id
