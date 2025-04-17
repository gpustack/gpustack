"""
Ensure the local gpustack package is imported before any installed ones.
"""

import os
import sys

# Prepend the repository root to sys.path so that the local gpustack module is used
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
