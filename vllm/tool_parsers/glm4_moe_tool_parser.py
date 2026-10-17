# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GLM-4 Tool Call Parser with incremental string streaming support.

This parser fixes the streaming issue reported in Issue #32829 where long string
parameters (e.g., file content with 4000+ characters of code) are buffered until
complete, causing multi-second delays before the user sees any content.

The fix streams string values incrementally as they arrive, providing a true
streaming experience for long content.
"""

import ast
import json
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.tool_parsers.utils import partial_tag_overlap

logger = init_logger(__name__)


class Glm4MoeModelToolParser(ToolParser):
    """Tool parser for GLM-4 models with incremental string streaming.

    This parser emits tool-call deltas incrementally as arguments arrive.
    For string-type parameters, content is streamed character-by-character
    rather than waiting for the complete </arg_value> tag.
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        # Stateful streaming fields
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict[str, Any]] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.arg_key_start: str = "<arg_key>"
        self.arg_key_end: str = "</arg_key>"
        self.arg_val_start: str = "<arg_value>"
        self.arg_val_end: str = "</arg_value>"

        self.tool_calls_start_token = self.tool_call_start_token

        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(
            r"<tool_call>([^\n]*)\n(.*)</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        # Streaming state for re-parse-and-diff.  The serving layer can
        # deliver several model tokens in one delta (MTP or stream_interval),
        # so parsing only the new delta can leave complete tags stranded in a
        # buffer at the end of generation.
        self._streaming_text: str = ""
        self._sent_content_idx: int = 0
        self._tool_call_ids: list[str] = []

    @staticmethod
    def _deserialize(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        return value

    @staticmethod
    def _json_escape_string_content(s: str) -> str:
        """JSON-escape string content for incremental streaming.

        This escapes the content that goes INSIDE a JSON string (between quotes),
        not including the surrounding quotes themselves.
        """
        if not s:
            return ""
        return json.dumps(s, ensure_ascii=False)[1:-1]

    @staticmethod
    def _is_string_type(
        tool_name: str,
        arg_name: str,
        tools: list[ChatCompletionToolsParam] | None,
    ) -> bool:
        if tools is None:
            return False
        for tool in tools:
            if tool.function.name != tool_name:
                continue
            if tool.function.parameters is None:
                return False
            arg_type = (
                tool.function.parameters.get("properties", {})
                .get(arg_name, {})
                .get("type", None)
            )
            return arg_type == "string"
        logger.debug("No tool named '%s'.", tool_name)
        return False

    @staticmethod
    def _tools_enabled(request: ChatCompletionRequest) -> bool:
        """Return whether tool parsing should be applied for this request."""
        try:
            tools = getattr(request, "tools", None)
            tool_choice = getattr(request, "tool_choice", None)
            return bool(tools) and tool_choice != "none"
        except Exception:
            logger.exception("Failed to determine if tools are enabled.")
            return False

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust request parameters for tool call token handling."""
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Ensure tool call tokens (<tool_call>, </tool_call>) are not skipped
            # during decoding. Even though they are not marked as special tokens,
            # setting skip_special_tokens=False ensures proper handling in
            # transformers 5.x where decoding behavior may have changed.
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        matched_tool_calls = self.func_call_regex.findall(model_output)
        logger.debug("model_output: %s", model_output)
        try:
            tool_calls: list[ToolCall] = []
            for match in matched_tool_calls:
                tc_detail = self.func_detail_regex.search(match)
                if not tc_detail:
                    logger.warning(
                        "Failed to parse tool call details from: %s",
                        match,
                    )
                    continue
                tc_name = tc_detail.group(1).strip()
                tc_args = tc_detail.group(2)
                pairs = self.func_arg_regex.findall(tc_args) if tc_args else []
                arg_dct: dict[str, Any] = {}
                for key, value in pairs:
                    arg_key = key.strip()
                    arg_val = value.strip()
                    if not self._is_string_type(tc_name, arg_key, request.tools):
                        arg_val = self._deserialize(arg_val)
                    logger.debug("arg_key = %s, arg_val = %s", arg_key, arg_val)
                    arg_dct[arg_key] = arg_val
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=tc_name,
                            arguments=json.dumps(arg_dct, ensure_ascii=False),
                        ),
                    )
                )
        except Exception:
            logger.exception("Failed to extract tool call spec")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        else:
            if len(tool_calls) > 0:
                content = model_output[: model_output.find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True, tool_calls=tool_calls, content=content
                )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _update_streaming_text(self, current_text: str, delta_text: str) -> str:
        """Maintain cumulative text for callers that do not provide it.

        ``current_text`` is authoritative in the serving path.  The fallback
        to ``delta_text`` keeps the parser compatible with older direct unit
        callers that did not provide cumulative text.
        """
        if current_text:
            if self._streaming_text and not current_text.startswith(
                self._streaming_text
            ):
                self._reset_reparse_state()
            self._streaming_text = current_text
        elif delta_text:
            self._streaming_text += delta_text
        return self._streaming_text

    def _reset_reparse_state(self) -> None:
        """Reset all state used by one re-parse-and-diff stream."""
        self._streaming_text = ""
        self._sent_content_idx = 0
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()
        self._tool_call_ids.clear()

    def _extract_content(self, text: str) -> str | None:
        """Return newly available text outside tool-call regions."""
        content_segments: list[str] = []
        position = self._sent_content_idx

        while position < len(text):
            start = text.find(self.tool_call_start_token, position)
            if start == -1:
                tail = text[position:]
                overlap = partial_tag_overlap(tail, self.tool_call_start_token)
                if overlap:
                    tail = tail[:-overlap]
                    position = len(text) - overlap
                else:
                    position = len(text)
                if tail:
                    content_segments.append(tail)
                break

            if start > position:
                content_segments.append(text[position:start])

            end = text.find(self.tool_call_end_token, start)
            if end == -1:
                position = start
                break
            position = end + len(self.tool_call_end_token)

        if position > self._sent_content_idx:
            self._sent_content_idx = position
        return "".join(content_segments) or None

    def _extract_tool_call_regions(self, text: str) -> list[tuple[str, bool]]:
        """Extract tool-call bodies and whether their closing tag arrived."""
        regions: list[tuple[str, bool]] = []
        position = 0
        while True:
            start = text.find(self.tool_call_start_token, position)
            if start == -1:
                break
            body_start = start + len(self.tool_call_start_token)
            end = text.find(self.tool_call_end_token, body_start)
            if end == -1:
                body = text[body_start:]
                overlap = partial_tag_overlap(body, self.tool_call_end_token)
                if overlap:
                    body = body[:-overlap]
                regions.append((body, False))
                break
            regions.append((text[body_start:end], True))
            position = end + len(self.tool_call_end_token)
        return regions

    def _extract_tool_name_from_region(
        self, body: str, is_complete: bool
    ) -> str | None:
        newline = body.find("\n")
        key_start = body.find(self.arg_key_start)
        positions = [position for position in (newline, key_start) if position != -1]
        if not positions:
            if is_complete:
                return body.strip() or None
            return None
        name = body[: min(positions)].strip()
        return name or None

    def _build_args_json_so_far(
        self,
        tool_name: str,
        body: str,
        is_complete: bool,
        request: ChatCompletionRequest,
    ) -> str:
        """Build the JSON argument prefix represented by a tool-call body."""
        parts: list[str] = []
        pairs = self.func_arg_regex.findall(body)
        for key, value in pairs:
            key = key.strip()
            if self._is_string_type(tool_name, key, request.tools):
                value_json = json.dumps(value, ensure_ascii=False)
            else:
                value_json = json.dumps(
                    self._deserialize(value.strip()), ensure_ascii=False
                )
            parts.append(f"{json.dumps(key, ensure_ascii=False)}: {value_json}")

        last_value_start = body.rfind(self.arg_val_start)
        last_value_end = body.rfind(self.arg_val_end)
        has_partial_value = last_value_start != -1 and (
            last_value_end == -1 or last_value_end < last_value_start
        )

        if has_partial_value:
            key_match = None
            for match in re.finditer(
                re.escape(self.arg_key_start) + r"(.*?)" + re.escape(self.arg_key_end),
                body[:last_value_start],
                re.DOTALL,
            ):
                key_match = match

            if key_match:
                key = key_match.group(1).strip()
                value = body[last_value_start + len(self.arg_val_start) :]
                overlap = partial_tag_overlap(value, self.arg_val_end)
                if overlap:
                    value = value[:-overlap]
                key_json = json.dumps(key, ensure_ascii=False)

                if is_complete:
                    if self._is_string_type(tool_name, key, request.tools):
                        value_json = json.dumps(value, ensure_ascii=False)
                    else:
                        value_json = json.dumps(
                            self._deserialize(value.strip()), ensure_ascii=False
                        )
                    parts.append(f"{key_json}: {value_json}")
                elif self._is_string_type(tool_name, key, request.tools):
                    escaped = self._json_escape_string_content(value)
                    parts.append(f'{key_json}: "{escaped}')
                else:
                    parts.append(f"{key_json}: {value}")

        if not parts:
            return "{}" if is_complete else ""

        result = "{" + ", ".join(parts)
        return result + "}" if is_complete else result

    def _ensure_tool_state_for(self, index: int) -> None:
        while len(self._tool_call_ids) <= index:
            self._tool_call_ids.append(
                make_tool_call_id(id_type="random", func_name=None, idx=None)
            )
        while len(self.streamed_args_for_tool) <= index:
            self.streamed_args_for_tool.append("")
        while len(self.prev_tool_call_arr) <= index:
            self.prev_tool_call_arr.append({})

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if not self._tools_enabled(request):
            return DeltaMessage(content=delta_text) if delta_text else None

        text = self._update_streaming_text(current_text, delta_text)
        content = self._extract_content(text)
        regions = self._extract_tool_call_regions(text)
        tool_call_deltas: list[DeltaToolCall] = []

        for index, (body, is_complete) in enumerate(regions):
            tool_name = self._extract_tool_name_from_region(body, is_complete)
            if not tool_name:
                continue

            self._ensure_tool_state_for(index)
            if self.prev_tool_call_arr[index].get("name") != tool_name:
                self.prev_tool_call_arr[index]["name"] = tool_name
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=index,
                        id=self._tool_call_ids[index],
                        type="function",
                        function=DeltaFunctionCall(
                            name=tool_name, arguments=""
                        ).model_dump(exclude_none=True),
                    )
                )

            args_so_far = self._build_args_json_so_far(
                tool_name, body, is_complete, request
            )
            previous_args = self.streamed_args_for_tool[index]
            if args_so_far and args_so_far.startswith(previous_args):
                args_delta = args_so_far[len(previous_args) :]
                self.streamed_args_for_tool[index] = args_so_far
                self.prev_tool_call_arr[index]["arguments"] = args_so_far
                if args_delta:
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=index,
                            function=DeltaFunctionCall(arguments=args_delta).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

        if self.prev_tool_call_arr:
            self.current_tool_id = len(self.prev_tool_call_arr) - 1
            self.current_tool_name_sent = True

        if tool_call_deltas:
            return DeltaMessage(content=content, tool_calls=tool_call_deltas)
        if content:
            # Leave tool_calls unset so exclude_unset serialization does not
            # signal a tool-call transition to protocol-converting gateways.
            return DeltaMessage(content=content)
        return None
