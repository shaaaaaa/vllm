# SPDX-License-Identifier: Apache-2.0
import json
import os
import time
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_SOURCE_KEY = "ascend_live_split_source_v1"


def log_live_source_handoff(
    event: str,
    request_id: str,
    params: dict[str, Any] | None,
    **fields: Any,
) -> None:
    if os.environ.get("LMCACHE_COLD_START_PERF", "0").lower() in (
        "", "0", "false", "no", "off"
    ) or not isinstance(params, dict):
        return
    source = params.get(_SOURCE_KEY)
    capabilities = params.get("live_split_capabilities", ())
    if (
        source is None
        and not capabilities
        and not params.get("do_remote_decode")
        and not params.get("request_live_split")
    ):
        return
    descriptors = source.get("descriptors", ()) if isinstance(source, dict) else ()
    logger.info(
        "[LMCACHE_COLD_PERF] %s",
        json.dumps(
            {
                "schema": 1,
                "event": event,
                "pid": os.getpid(),
                "monotonic_ms": round(time.perf_counter() * 1000, 3),
                "req_id": request_id,
                "source_present": source is not None,
                "descriptor_count": len(descriptors),
                "segment_count": sum(
                    len(item.get("segments", ()))
                    for item in descriptors
                    if isinstance(item, dict)
                ),
                "compact_layer_count": sum(
                    len(item.get("compact_layout", {}).get("layers", ()))
                    for item in descriptors
                    if isinstance(item, dict)
                ),
                "compact_run_count": sum(
                    len(item.get("compact_layout", {}).get("runs", ()))
                    for item in descriptors
                    if isinstance(item, dict)
                ),
                "transfer_param_keys": sorted(params),
                **fields,
            },
            separators=(",", ":"),
        ),
    )
