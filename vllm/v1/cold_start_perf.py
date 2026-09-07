# SPDX-License-Identifier: Apache-2.0
"""Startup-only gate for branch-local cold-start performance diagnostics.

This module has no LMCache or device-runtime dependency. Keep the false values
aligned with LMCache and vllm-ascend so disabling one knob disables the full
cross-process diagnostic path. Configure it before starting the server.
"""

import os

COLD_START_PERF_ENABLED = os.environ.get(
    "LMCACHE_COLD_START_PERF", "0"
).strip().lower() not in ("", "0", "false", "no", "off")
