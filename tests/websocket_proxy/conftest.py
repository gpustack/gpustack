"""
Initialize logging for websocket_proxy tests.
"""

import sys
import os
import logging
from gpustack.logging import TRACE_LEVEL

# Ensure local gpustack package is used
repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


logging.addLevelName(TRACE_LEVEL, "TRACE")


def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)


logging.Logger.trace = trace
