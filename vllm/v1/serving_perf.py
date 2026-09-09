# SPDX-License-Identifier: Apache-2.0
"""Startup-only gate for PD serving performance diagnostics.

This module has no LMCache or device-runtime dependency. Keep the false values
aligned with LMCache and vllm-ascend so disabling one knob disables the full
cross-process diagnostic path. Configure PD_SERVING_PERF before starting the
server. Host timing is enabled by 1, detail and device; device timing remains
explicitly opt-in in components that implement it. Content diagnostics and
operational errors are independent of this switch.
"""

import os

SERVING_PERF_ENABLED = os.environ.get("PD_SERVING_PERF", "0").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
    "off",
)
