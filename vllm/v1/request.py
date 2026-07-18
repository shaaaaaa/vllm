# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import binascii
import enum
import hashlib
import hmac
import sys
import time
from array import array
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash

logger = init_logger(__name__)


_FINAL_HIDDEN_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
_MAX_FINAL_HIDDEN_BYTES = 16 * 1024 * 1024
_MAX_FINAL_HIDDEN_BASE64_CHARS = 4 * ((_MAX_FINAL_HIDDEN_BYTES + 2) // 3)


def validate_final_hidden_payload(
    payload: object, expected_hidden_size: int | None = None
) -> bool:
    """Validate the bounded, bit-exact final-hidden transport payload."""
    if not isinstance(payload, dict):
        return False
    dtype_size = _FINAL_HIDDEN_DTYPE_BYTES.get(payload.get("dtype"))
    shape = payload.get("shape")
    data = payload.get("data")
    if (
        payload.get("version") != 1
        or payload.get("encoding") != "base64"
        or dtype_size is None
        or not isinstance(shape, list)
        or len(shape) != 1
        or type(shape[0]) is not int
        or shape[0] <= 0
        or (expected_hidden_size is not None and shape[0] != expected_hidden_size)
        or not isinstance(data, str)
        or len(data) > _MAX_FINAL_HIDDEN_BASE64_CHARS
        or not isinstance(payload.get("data_sha256"), str)
    ):
        return False
    expected_bytes = shape[0] * dtype_size
    if expected_bytes > _MAX_FINAL_HIDDEN_BYTES:
        return False
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError, TypeError):
        return False
    return len(raw) == expected_bytes and hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), payload["data_sha256"]
    )


def compute_prompt_token_fingerprint(token_ids: list[int]) -> str:
    """Return a stable hash binding a hidden-state artifact to its prompt."""
    packed = array("q", token_ids)
    if sys.byteorder != "little":
        packed.byteswap()
    digest = hashlib.sha256()
    digest.update(b"vllm-final-hidden-prompt-v1\0")
    digest.update(len(token_ids).to_bytes(8, "little"))
    digest.update(packed.tobytes())
    return digest.hexdigest()


@dataclass
class StreamingUpdate:
    """Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None

    @classmethod
    def from_request(cls, request: "Request") -> "StreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
        )


class Request:
    # Final-hidden handoff state is intentionally lazy. Ordinary requests use
    # these immutable class defaults and do not grow their instance __dict__.
    capture_final_hidden = False
    bootstrap_final_hidden: dict[str, Any] | None = None
    bootstrap_sample_pending = False
    dsa_compact_allocated = False
    captured_final_hidden: dict[str, Any] | None = None
    final_hidden_prompt_fingerprint: str | None = None

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: "LoRARequest | None" = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        resumable: bool = False,
        reasoning_ended: bool | None = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        if self.structured_output_request is not None:
            self.structured_output_request.reasoning_ended = reasoning_ended
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None
        capture_final_hidden = False
        bootstrap_final_hidden: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_FSM

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
                if self.kv_transfer_params is not None:
                    capture_final_hidden = bool(
                        self.kv_transfer_params.get("ret_final_hidden", False)
                    )
                    final_hidden = self.kv_transfer_params.get(
                        "bootstrap_final_hidden"
                    )
                    if validate_final_hidden_payload(final_hidden):
                        bootstrap_final_hidden = final_hidden
                        decoder_engine_received_unix_ns = time.time_ns()
                        producer_ready_unix_ns = final_hidden.get(
                            "producer_ready_unix_ns"
                        )
                        proxy_decoder_send_unix_ns = final_hidden.get(
                            "proxy_decoder_send_unix_ns"
                        )
                        final_hidden["decoder_engine_received_unix_ns"] = (
                            decoder_engine_received_unix_ns
                        )
                        producer_to_engine_ms = (
                            (decoder_engine_received_unix_ns - producer_ready_unix_ns)
                            / 1e6
                            if isinstance(producer_ready_unix_ns, int)
                            else None
                        )
                        proxy_to_engine_ms = (
                            (
                                decoder_engine_received_unix_ns
                                - proxy_decoder_send_unix_ns
                            )
                            / 1e6
                            if isinstance(proxy_decoder_send_unix_ns, int)
                            else None
                        )
                        logger.info(
                            "[FINAL_HIDDEN_REQUEST_ARTIFACT] req=%s "
                            "basic_validation=ok dtype=%s shape=%s "
                            "prompt_length=%s checksum=%s "
                            "producer_to_engine_ms=%s proxy_to_engine_ms=%s "
                            "clock_sync_required=true "
                            "proxy_to_engine_includes=http_upload_json_"
                            "tokenization_ipc",
                            request_id,
                            final_hidden.get("dtype"),
                            final_hidden.get("shape"),
                            final_hidden.get("prompt_length"),
                            str(final_hidden.get("data_sha256", ""))[:16],
                            f"{producer_to_engine_ms:.3f}"
                            if producer_to_engine_ms is not None
                            else "unknown",
                            f"{proxy_to_engine_ms:.3f}"
                            if proxy_to_engine_ms is not None
                            else "unknown",
                        )
                    elif final_hidden is not None:
                        logger.warning(
                            "[FINAL_HIDDEN_REQUEST_ARTIFACT] req=%s "
                            "basic_validation=failed action=normal_path",
                            request_id,
                        )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        # Cache per-block prompt-embed hashes to avoid rehashing the same
        # tensor slices when generating extra keys.
        self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {}
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        handoff_requested = (
            capture_final_hidden or bootstrap_final_hidden is not None
        )
        if handoff_requested:
            handoff_inputs_supported = (
                prompt_token_ids is not None
                and prompt_embeds is None
                and not mm_features
                and lora_request is None
                and not resumable
                and (
                    sampling_params is None
                    or sampling_params.prompt_logprobs is None
                )
            )
            if not handoff_inputs_supported:
                logger.warning(
                    "[FINAL_HIDDEN_REQUEST_UNSUPPORTED] req=%s "
                    "prompt_token_ids=%s prompt_embeds=%s multimodal=%s "
                    "lora=%s resumable=%s prompt_logprobs=%s "
                    "action=disable_handoff",
                    request_id,
                    prompt_token_ids is not None,
                    prompt_embeds is not None,
                    bool(mm_features),
                    lora_request is not None,
                    resumable,
                    sampling_params.prompt_logprobs
                    if sampling_params is not None
                    else None,
                )
                capture_final_hidden = False
                bootstrap_final_hidden = None
                handoff_requested = False

        final_hidden_prompt_fingerprint: str | None = None
        if handoff_requested and prompt_token_ids is not None:
            final_hidden_prompt_fingerprint = compute_prompt_token_fingerprint(
                prompt_token_ids
            )
        if capture_final_hidden:
            self.capture_final_hidden = True
            self.final_hidden_prompt_fingerprint = final_hidden_prompt_fingerprint

        if capture_final_hidden:
            logger.info(
                "[FINAL_HIDDEN_REQUEST_CAPTURE] req=%s enabled=true "
                "prompt_tokens=%d prompt_hash=%s max_tokens=%d",
                request_id,
                self.num_prompt_tokens,
                self.final_hidden_prompt_fingerprint[:16]
                if self.final_hidden_prompt_fingerprint is not None
                else None,
                self.max_tokens,
            )
        if bootstrap_final_hidden is not None:
            payload_matches_prompt = (
                final_hidden_prompt_fingerprint is not None
                and bootstrap_final_hidden.get("prompt_length")
                == self.num_prompt_tokens
                and bootstrap_final_hidden.get("prompt_sha256")
                == final_hidden_prompt_fingerprint
            )
            if payload_matches_prompt:
                self.final_hidden_prompt_fingerprint = (
                    final_hidden_prompt_fingerprint
                )
                self.bootstrap_final_hidden = bootstrap_final_hidden
                self.bootstrap_sample_pending = True
                logger.info(
                    "[FINAL_HIDDEN_REQUEST_ACCEPTED] req=%s prompt_tokens=%d "
                    "prompt_hash=%s bootstrap_pending=true",
                    request_id,
                    self.num_prompt_tokens,
                    final_hidden_prompt_fingerprint[:16]
                    if final_hidden_prompt_fingerprint is not None
                    else None,
                )
            else:
                logger.warning(
                    "[FINAL_HIDDEN_REQUEST_REJECTED] req=%s reason=prompt_mismatch "
                    "request_prompt_tokens=%d artifact_prompt_tokens=%s "
                    "request_hash=%s artifact_hash=%s action=normal_prefill",
                    request_id,
                    self.num_prompt_tokens,
                    bootstrap_final_hidden.get("prompt_length"),
                    final_hidden_prompt_fingerprint[:16]
                    if final_hidden_prompt_fingerprint is not None
                    else None,
                    str(bootstrap_final_hidden.get("prompt_sha256", ""))[:16],
                )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )

        # Used in async scheduling.
        self.num_output_placeholders = 0
        # Used in forced preemption (reset_prefix_cache) with async scheduling.
        self.discard_latest_async_tokens = False

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        # The number of tokens that have been computed remotely.
        self.num_external_computed_tokens = 0

        self.block_hashes: list[BlockHash] = []
        # Store the block hasher without binding self to avoid creating a
        # reference cycle (Request -> partial -> Request) that prevents
        # immediate garbage collection via reference counting.
        self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
        self.update_block_hashes()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        # Used for streaming
        self.resumable = resumable
        # None entry in the queue means finished.
        self.streaming_queue: deque[StreamingUpdate | None] | None = None

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            resumable=request.resumable,
            reasoning_ended=request.reasoning_ended,
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_features)

    @property
    def has_encoder_inputs(self) -> bool:
        return self.num_encoder_inputs > 0

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        return self.mm_features[input_id].mm_position.get_num_embeds()

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, arrival time, and request ID.
        Used in priority scheduling.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}
