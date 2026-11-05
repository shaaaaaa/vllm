# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import Mock

import pytest
from jsonschema.validators import validator_for

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.glm4_moe_tool_parser import (
    Glm4MoeModelToolParser,
)
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

# Use a common model that is likely to be available
MODEL = "zai-org/GLM-4.5"


@pytest.fixture(scope="module")
def glm4_moe_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def glm4_moe_tool_parser(glm4_moe_tokenizer):
    return Glm4MoeModelToolParser(glm4_moe_tokenizer)


@pytest.fixture
def mock_request() -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = [  # GLM45 parser needs this attribute to enable tool parsing.
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
        ),
    ]
    return request


def assert_tool_calls(
    actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) > 0

        assert actual_tool_call.type == "function"
        assert actual_tool_call.function.name == expected_tool_call.function.name
        # Compare arguments as JSON objects to handle formatting differences
        actual_args = json.loads(actual_tool_call.function.arguments)
        expected_args = json.loads(expected_tool_call.function.arguments)
        assert actual_args == expected_args


def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, mock_request):
    model_output = "This is a test"
    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "single_tool_call",
        "multiple_tool_calls",
        "tool_call_with_content_before",
        "tool_call_with_mixed_args",
        "tool_call_with_chinese_content",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """<tool_call>get_current_weather
    <arg_key>city</arg_key>
    <arg_value>Dallas</arg_value>
    <arg_key>state</arg_key>
    <arg_value>TX</arg_value>
    <arg_key>unit</arg_key>
    <arg_value>fahrenheit</arg_value>
    </tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Dallas",
                                "state": "TX",
                                "unit": "fahrenheit",
                            }
                        ),
                    )
                )
            ],
            "",
        ),
        (
            """<tool_call>get_current_weather
    <arg_key>city</arg_key>
    <arg_value>Dallas</arg_value>
    <arg_key>state</arg_key>
    <arg_value>TX</arg_value>
    <arg_key>unit</arg_key>
    <arg_value>fahrenheit</arg_value>
    </tool_call>
    <tool_call>get_current_weather
    <arg_key>city</arg_key>
    <arg_value>Orlando</arg_value>
    <arg_key>state</arg_key>
    <arg_value>FL</arg_value>
    <arg_key>unit</arg_key>
    <arg_value>fahrenheit</arg_value>
    </tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Dallas",
                                "state": "TX",
                                "unit": "fahrenheit",
                            }
                        ),
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Orlando",
                                "state": "FL",
                                "unit": "fahrenheit",
                            }
                        ),
                    )
                ),
            ],
            "",
        ),
        (
            """I'll help you check the weather. <tool_call>get_current_weather
    <arg_key>city</arg_key>
    <arg_value>Seattle</arg_value>
    <arg_key>state</arg_key>
    <arg_value>WA</arg_value>
    <arg_key>unit</arg_key>
    <arg_value>celsius</arg_value>
    </tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Seattle",
                                "state": "WA",
                                "unit": "celsius",
                            }
                        ),
                    )
                )
            ],
            "I'll help you check the weather. ",
        ),
        (
            """<tool_call>get_current_weather
    <arg_key>city</arg_key>
    <arg_value>New York</arg_value>
    <arg_key>state</arg_key>
    <arg_value>NY</arg_value>
    <arg_key>unit</arg_key>
    <arg_value>celsius</arg_value>
    </tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "New York",
                                "state": "NY",
                                "unit": "celsius",
                            }
                        ),
                    )
                )
            ],
            "",
        ),
        (
            """I will help you get the weather.<tool_call>get_weather
    <arg_key>city</arg_key>
    <arg_value>Beijing</arg_value>
    <arg_key>date</arg_key>
    <arg_value>2025-08-01</arg_value>
    </tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "city": "Beijing",
                                "date": "2025-08-01",
                            }
                        ),
                    )
                )
            ],
            "I will help you get the weather.",
        ),
    ],
)
def test_extract_tool_calls(
    glm4_moe_tool_parser,
    mock_request,
    model_output,
    expected_tool_calls,
    expected_content,
):
    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called
    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_with_thinking_tags(glm4_moe_tool_parser, mock_request):
    """Test tool extraction when thinking tags are present."""
    model_output = """<think>I want to get the weather.</think>

I will help you get the weather.
<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
<arg_key>date</arg_key>
<arg_value>2025-08-01</arg_value>
</tool_call>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "get_weather"

    expected_content = """<think>I want to get the weather.</think>

I will help you get the weather.
"""
    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_malformed_xml(glm4_moe_tool_parser, mock_request):
    """Test that malformed XML is handled gracefully."""
    model_output = """<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Seattle</arg_value>
<arg_key>incomplete_arg
<arg_value>value</arg_value>
</tool_call>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    # Should handle malformed XML gracefully
    # The parser should either extract what it can or return no tool calls
    # depending on how robust we want the parsing to be
    assert isinstance(extracted_tool_calls.tools_called, bool)
    assert isinstance(extracted_tool_calls.tool_calls, list)


def test_extract_tool_calls_empty_arguments(glm4_moe_tool_parser, mock_request):
    """Test tool calls with no arguments."""
    model_output = """<tool_call>get_current_time
</tool_call>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "get_current_time"
    # Empty arguments should result in empty JSON object
    assert extracted_tool_calls.tool_calls[0].function.arguments == "{}"


def test_extract_tool_calls_mixed_content(glm4_moe_tool_parser, mock_request):
    """Test extraction with mixed content and multiple tool calls."""
    model_output = """I will help you get the weather info.

<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
<arg_key>date</arg_key>
<arg_value>2025-08-01</arg_value>
</tool_call>

meaningwhile, I will also check the weather in Shanghai.

<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Shanghai</arg_value>
<arg_key>date</arg_key>
<arg_value>2025-08-01</arg_value>
</tool_call>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 2

    # Check first tool call
    assert extracted_tool_calls.tool_calls[0].function.name == "get_weather"
    args1 = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert args1["city"] == "Beijing"
    assert args1["date"] == "2025-08-01"

    # Check second tool call
    assert extracted_tool_calls.tool_calls[1].function.name == "get_weather"
    args2 = json.loads(extracted_tool_calls.tool_calls[1].function.arguments)
    assert args2["city"] == "Shanghai"
    assert args2["date"] == "2025-08-01"

    # Content should be everything before the first tool call
    assert extracted_tool_calls.content == "I will help you get the weather info.\n\n"


def test_streaming_basic_functionality(glm4_moe_tool_parser, mock_request):
    """Test basic streaming functionality."""
    # Reset streaming state
    glm4_moe_tool_parser.current_tool_name_sent = False
    glm4_moe_tool_parser.prev_tool_call_arr = []
    glm4_moe_tool_parser.current_tool_id = -1
    glm4_moe_tool_parser.streamed_args_for_tool = []

    # Test with a simple tool call
    current_text = """<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
</tool_call>"""

    # Mock token IDs for testing
    tool_call_start_id = glm4_moe_tool_parser.tool_call_start_token_id or 12345
    tool_call_end_id = glm4_moe_tool_parser.tool_call_end_token_id or 12346

    result = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text="</tool_call>",
        previous_token_ids=[],
        current_token_ids=[tool_call_start_id, tool_call_end_id],
        delta_token_ids=[tool_call_end_id],
        request=mock_request,
    )

    # The result behavior depends on the streaming state
    # This test mainly ensures no exceptions are thrown
    assert result is None or hasattr(result, "tool_calls") or hasattr(result, "content")


def test_streaming_no_tool_calls(glm4_moe_tool_parser, mock_request):
    """Test streaming when there are no tool calls."""
    current_text = "This is just regular text without any tool calls."

    result = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text=current_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    # The re-parse parser returns the newly available cumulative content.
    assert result is not None
    assert hasattr(result, "content")
    assert result.content == current_text
    assert json.loads(result.model_dump_json(exclude_unset=True)) == {
        "content": current_text
    }


@pytest.mark.parametrize(
    "parser_cls", [Glm4MoeModelToolParser, Glm47MoeModelToolParser]
)
@pytest.mark.parametrize("chunk_size", [1, 4, 256])
def test_streaming_text_omits_empty_tool_calls(
    parser_cls, chunk_size, glm4_moe_tokenizer, mock_request
):
    """Plain text must not signal a tool transition to protocol gateways."""
    parser = parser_cls(glm4_moe_tokenizer)
    text = "我是 Claude Code，底层模型是 **GLM-5.3**。\n\n服务 IP 需查看配置。"
    received_text = []
    for start in range(0, len(text), chunk_size):
        delta_text = text[start : start + chunk_size]
        delta = parser.extract_tool_calls_streaming(
            previous_text=text[:start],
            current_text=text[: start + chunk_size],
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=mock_request,
        )
        assert delta is not None
        response = ChatCompletionStreamResponse(
            id="chatcmpl-text-stream",
            created=0,
            model=MODEL,
            choices=[ChatCompletionResponseStreamChoice(index=0, delta=delta)],
        )
        # Match the serving layer's serialization, including nested fields.
        payload = json.loads(response.model_dump_json(exclude_unset=True))
        wire_delta = payload["choices"][0]["delta"]
        assert wire_delta == {"content": delta_text}
        received_text.append(wire_delta["content"])

    assert "".join(received_text) == text


@pytest.mark.parametrize(
    "parser_cls", [Glm4MoeModelToolParser, Glm47MoeModelToolParser]
)
def test_streaming_mixed_text_and_tool_call_preserves_arguments(
    parser_cls, glm4_moe_tokenizer, mock_request
):
    """Omitting empty tool lists must preserve real calls in an MTP chunk."""
    parser = parser_cls(glm4_moe_tokenizer)
    text = (
        "先查天气。<tool_call>get_weather\n"
        "<arg_key>city</arg_key><arg_value>北京</arg_value></tool_call>"
    )
    delta = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=text,
        delta_text=text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )
    assert delta is not None
    payload = json.loads(delta.model_dump_json(exclude_unset=True))
    assert payload["content"] == "先查天气。"
    calls = payload["tool_calls"]
    assert calls[0]["id"]
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    assert all(call["index"] == 0 for call in calls)
    arguments = "".join(call["function"].get("arguments", "") for call in calls)
    assert json.loads(arguments) == {"city": "北京"}

    # A completed tool call must not leak an empty tool list into later text.
    suffix = "\n查询完成。"
    delta = parser.extract_tool_calls_streaming(
        previous_text=text,
        current_text=text + suffix,
        delta_text=suffix,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )
    assert delta is not None
    assert json.loads(delta.model_dump_json(exclude_unset=True)) == {"content": suffix}


def test_streaming_with_content_before_tool_calls(glm4_moe_tool_parser, mock_request):
    """Test streaming when there's content before tool calls."""
    # Reset streaming state
    glm4_moe_tool_parser.current_tool_name_sent = False
    glm4_moe_tool_parser.prev_tool_call_arr = []
    glm4_moe_tool_parser.current_tool_id = -1
    glm4_moe_tool_parser.streamed_args_for_tool = []

    current_text = "I will help you get the weather<tool_call>"

    result = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text=current_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    # The incomplete opening tag is held back while preceding content is sent.
    assert result is not None
    assert hasattr(result, "content")
    assert result.content == "I will help you get the weather"
    assert json.loads(result.model_dump_json(exclude_unset=True)) == {
        "content": "I will help you get the weather"
    }


def test_extract_tool_calls_special_characters(glm4_moe_tool_parser, mock_request):
    """Test tool calls with special characters and unicode."""
    model_output = """<tool_call>send_message
<arg_key>recipient</arg_key>
<arg_value>Amy</arg_value>
<arg_key>message</arg_key>
<arg_value>It is a nice day</arg_value>
<arg_key>priority</arg_key>
<arg_value>high</arg_value>
</tool_call>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "send_message"

    args = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert args["recipient"] == "Amy"
    assert args["message"] == "It is a nice day"
    assert args["priority"] == "high"


def test_extract_tool_calls_incomplete_tool_call(glm4_moe_tool_parser, mock_request):
    """Test incomplete tool calls (missing closing tag)."""
    model_output = """<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
<arg_key>date</arg_key>
<arg_value>2025-08-01</arg_value>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    # Incomplete tool calls should not be extracted
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


def _reset_streaming_state(parser):
    """Helper to reset parser streaming state."""
    parser._streaming_text = ""
    parser._sent_content_idx = 0
    parser.current_tool_name_sent = False
    parser.prev_tool_call_arr = []
    parser.current_tool_id = -1
    parser.streamed_args_for_tool = []
    parser._tool_call_ids = []


def test_streaming_incremental_string_value(glm4_moe_tool_parser, mock_request):
    """Test incremental streaming of string argument values."""
    _reset_streaming_state(glm4_moe_tool_parser)

    # Simulate streaming a tool call character by character
    chunks = [
        "<tool_call>",
        "get_weather\n",
        "<arg_key>city</arg_key>",
        "<arg_value>",
        "Bei",
        "jing",
        "</arg_value>",
        "</tool_call>",
    ]

    collected_fragments = []
    for chunk in chunks:
        result = glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=mock_request,
        )
        if result is not None and hasattr(result, "tool_calls") and result.tool_calls:
            for tc in result.tool_calls:
                if hasattr(tc, "function") and tc.function:
                    func = tc.function
                    if isinstance(func, dict):
                        if func.get("arguments"):
                            collected_fragments.append(func["arguments"])
                        if func.get("name"):
                            collected_fragments.append(f"name:{func['name']}")
                    else:
                        if func.arguments:
                            collected_fragments.append(func.arguments)
                        if func.name:
                            collected_fragments.append(f"name:{func.name}")

    # Verify we got incremental streaming of the argument value
    assert len(collected_fragments) > 0
    # The fragments should include the tool name and argument pieces
    combined = "".join(collected_fragments)
    assert "get_weather" in combined or "name:get_weather" in combined


def test_streaming_empty_tool_call(glm4_moe_tool_parser, mock_request):
    """Test that empty tool calls don't cause infinite loops."""
    _reset_streaming_state(glm4_moe_tool_parser)

    # Empty tool call should be handled gracefully
    result = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="",
        delta_text="<tool_call></tool_call>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    # Should not hang and should return something (None or content)
    # The key is that this completes without hanging
    assert result is None or hasattr(result, "content") or hasattr(result, "tool_calls")
    # State should be properly reset
    assert glm4_moe_tool_parser.current_tool_id == -1


def test_streaming_no_args_tool_call(glm4_moe_tool_parser, mock_request):
    """A complete tool call without an argument block emits an empty object."""
    _reset_streaming_state(glm4_moe_tool_parser)
    text = "<tool_call>get_current_date</tool_call>"

    result = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=text,
        delta_text=text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    assert result is not None
    assert result.tool_calls
    assert result.tool_calls[0].function is not None
    assert result.tool_calls[0].function.name == "get_current_date"
    arguments = "".join(
        call.function.arguments or ""
        for call in result.tool_calls
        if call.function is not None
    )
    assert json.loads(arguments) == {}


def test_streaming_prev_tool_call_arr_updates(glm4_moe_tool_parser, mock_request):
    """Test that parser state tracks the re-parsed JSON prefix."""
    _reset_streaming_state(glm4_moe_tool_parser)

    chunks = [
        "<tool_call>get_weather\n",
        "<arg_key>city</arg_key>",
        "<arg_value>Beijing</arg_value>",
        "</tool_call>",
    ]

    current_text = ""
    for chunk in chunks:
        current_text += chunk
        glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=mock_request,
        )

    assert len(glm4_moe_tool_parser.prev_tool_call_arr) == 1
    tool_entry = glm4_moe_tool_parser.prev_tool_call_arr[0]
    assert tool_entry.get("name") == "get_weather"
    args = tool_entry.get("arguments")
    assert isinstance(args, str)
    assert json.loads(args) == {"city": "Beijing"}
    assert glm4_moe_tool_parser.streamed_args_for_tool[0] == args


def test_streaming_multiple_tool_calls_sequential(glm4_moe_tool_parser, mock_request):
    """Test streaming multiple sequential tool calls."""
    _reset_streaming_state(glm4_moe_tool_parser)

    # Stream two tool calls
    chunks = [
        "<tool_call>get_weather\n",
        "<arg_key>city</arg_key>",
        "<arg_value>Beijing</arg_value>",
        "</tool_call>",
        "<tool_call>get_weather\n",
        "<arg_key>city</arg_key>",
        "<arg_value>Shanghai</arg_value>",
        "</tool_call>",
    ]

    current_text = ""
    for chunk in chunks:
        current_text += chunk
        glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=mock_request,
        )

    # Should have two tool calls in prev_tool_call_arr
    assert len(glm4_moe_tool_parser.prev_tool_call_arr) == 2
    args0 = json.loads(glm4_moe_tool_parser.prev_tool_call_arr[0]["arguments"])
    args1 = json.loads(glm4_moe_tool_parser.prev_tool_call_arr[1]["arguments"])
    assert args0["city"] == "Beijing"
    assert args1["city"] == "Shanghai"


def test_streaming_json_escape_in_string(glm4_moe_tool_parser, mock_request):
    """Test that special characters in string values are properly escaped."""
    _reset_streaming_state(glm4_moe_tool_parser)

    # String with characters that need JSON escaping
    chunks = [
        "<tool_call>send_message\n",
        "<arg_key>message</arg_key>",
        '<arg_value>Hello "world"\nNew line</arg_value>',
        "</tool_call>",
    ]

    for chunk in chunks:
        glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=mock_request,
        )

    # The streamed_args_for_tool should contain valid JSON
    assert len(glm4_moe_tool_parser.streamed_args_for_tool) == 1
    args_json = glm4_moe_tool_parser.streamed_args_for_tool[0]
    # Should be parseable as JSON
    parsed = json.loads(args_json)
    assert "message" in parsed
    # The value should preserve the special characters
    assert '"' in parsed["message"] or "world" in parsed["message"]


def test_streaming_long_content_incremental(glm4_moe_tool_parser):
    """Test incremental streaming of long content (Issue #32829).

    This is the core fix: for long string values like code (4000+ chars),
    the parser should stream incrementally rather than buffering until
    complete. This test verifies we get many fragments, not just 1-3.
    """
    _reset_streaming_state(glm4_moe_tool_parser)

    # Bubble sort example from Issue #32829 - realistic long content
    bubble_sort_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bubble Sort Implementation
"""

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

if __name__ == "__main__":
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original: {test_arr}")
    sorted_arr = bubble_sort(test_arr.copy())
    print(f"Sorted: {sorted_arr}")'''

    # Create a request with tool schema to enable string type detection
    # This is required for incremental streaming of string values
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "write_to_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
            }
        ],
    )  # type: ignore

    # Simulate token-based streaming (special tags as single tokens)
    chunks = [
        "<tool_call>",
        "write_to_file\n",
        "<arg_key>file_path</arg_key>",
        "<arg_value>/tmp/bubble_sort.py</arg_value>",
        "<arg_key>content</arg_key>",
        "<arg_value>",
    ]
    # Add content line by line (realistic token streaming)
    for line in bubble_sort_code.split("\n"):
        chunks.append(line + "\n")
    chunks.append("</arg_value>")
    chunks.append("</tool_call>")

    # Count argument fragments
    fragment_count = 0
    for chunk in chunks:
        result = glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if result is not None and hasattr(result, "tool_calls") and result.tool_calls:
            for tc in result.tool_calls:
                if hasattr(tc, "function") and tc.function:
                    func = tc.function
                    args = (
                        func.get("arguments")
                        if isinstance(func, dict)
                        else getattr(func, "arguments", None)
                    )
                    if args:
                        fragment_count += 1

    # For true incremental streaming, we expect many fragments (10+)
    # Old buffered implementation would give only 1-3 fragments
    assert fragment_count >= 10, (
        f"Expected >=10 fragments for incremental streaming, got {fragment_count}"
    )

    # Verify final result is valid JSON
    assert len(glm4_moe_tool_parser.streamed_args_for_tool) == 1
    args_json = glm4_moe_tool_parser.streamed_args_for_tool[0]
    parsed = json.loads(args_json)
    assert parsed["file_path"] == "/tmp/bubble_sort.py"
    assert "def bubble_sort" in parsed["content"]


def test_extract_tool_calls_numeric_deserialization(glm4_moe_tool_parser, mock_request):
    """Test that numeric arguments are deserialized as numbers, not strings."""
    model_output = """<tool_call>calculate
<arg_key>operation</arg_key>
<arg_value>add</arg_value>
<arg_key>a</arg_key>
<arg_value>42</arg_value>
<arg_key>b</arg_key>
<arg_value>3.14</arg_value>
<arg_key>enabled</arg_key>
<arg_value>true</arg_value>
</tool_call>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1

    args = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)

    # String should remain string
    assert args["operation"] == "add"
    assert isinstance(args["operation"], str)

    # Integer should be deserialized as int
    assert args["a"] == 42
    assert isinstance(args["a"], int)

    # Float should be deserialized as float
    assert args["b"] == 3.14
    assert isinstance(args["b"], float)

    # Boolean should be deserialized as bool
    assert args["enabled"] is True
    assert isinstance(args["enabled"], bool)


def test_streaming_multi_token_chunk(glm4_moe_tool_parser, mock_request):
    """A complete argument can arrive across a multi-token delta."""
    _reset_streaming_state(glm4_moe_tool_parser)
    chunks = [
        "<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Bei",
        "jing</arg_value></tool_call>",
    ]

    current_text = ""
    deltas = []
    for chunk in chunks:
        current_text += chunk
        delta = glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=mock_request,
        )
        if delta is not None:
            deltas.append(delta)

    assert len(glm4_moe_tool_parser.prev_tool_call_arr) == 1
    assert json.loads(glm4_moe_tool_parser.prev_tool_call_arr[0]["arguments"]) == {
        "city": "Beijing"
    }
    streamed_args = "".join(
        tc.function.arguments
        for delta in deltas
        for tc in delta.tool_calls
        if tc.function and tc.function.arguments
    )
    assert json.loads(streamed_args) == {"city": "Beijing"}


def _stream_argument_fragments(parser, request, chunks):
    previous_text = ""
    fragments = []
    for chunk in chunks:
        current_text = previous_text + chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        previous_text = current_text
        if delta is None:
            continue
        payload = json.loads(delta.model_dump_json(exclude_unset=True))
        for call in payload.get("tool_calls", []):
            assert call["index"] == 0
            arguments = call.get("function", {}).get("arguments")
            if arguments:
                fragments.append(arguments)
    return fragments


@pytest.mark.parametrize(
    "parser_cls", [Glm4MoeModelToolParser, Glm47MoeModelToolParser]
)
@pytest.mark.parametrize("chunk_size", [1, 4, 16, 64, 256])
@pytest.mark.parametrize("task_id_first", [False, True])
def test_streaming_task_update_preserves_required_id(
    parser_cls, glm4_moe_tokenizer, chunk_size, task_id_first
):
    """An anyOf status must not truncate JSON or swallow the required task ID."""
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=[
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="TaskUpdate",
                    parameters={
                        "type": "object",
                        "properties": {
                            "taskId": {"type": "string"},
                            "status": {
                                "anyOf": [
                                    {
                                        "type": "string",
                                        "enum": [
                                            "pending",
                                            "in_progress",
                                            "interrupt",
                                            "completed",
                                        ],
                                    },
                                    {"type": "string", "const": "deleted"},
                                ]
                            },
                        },
                        "required": ["taskId"],
                    },
                )
            )
        ],
    )
    status = "<arg_key>status</arg_key><arg_value>in_progress</arg_value>"
    task_id = "<arg_key>taskId</arg_key><arg_value>1</arg_value>"
    body = task_id + status if task_id_first else status + task_id
    text = f"<tool_call>TaskUpdate\n{body}</tool_call>"
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    fragments = _stream_argument_fragments(
        parser_cls(glm4_moe_tokenizer), request, chunks
    )
    assert json.loads("".join(fragments)) == {
        "status": "in_progress",
        "taskId": "1",
    }


@pytest.mark.parametrize(
    "parser_cls", [Glm4MoeModelToolParser, Glm47MoeModelToolParser]
)
@pytest.mark.parametrize("chunk_size", [1, 4, 256])
@pytest.mark.parametrize(
    "schema, raw_value, expected",
    [
        pytest.param({}, "in_progress", "in_progress", id="untyped-string"),
        pytest.param({"type": "integer"}, " 42 ", 42, id="integer-whitespace"),
        pytest.param({"type": "number"}, "1e3", 1000.0, id="exponent"),
        pytest.param({"type": "boolean"}, "True", True, id="python-boolean"),
        pytest.param({"type": "array"}, "[1,2]", [1, 2], id="array-spacing"),
        pytest.param(
            {"type": "object"},
            '{"ok":true,"values":[1,2]}',
            {"ok": True, "values": [1, 2]},
            id="nested-object",
        ),
        pytest.param(
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "null",
            None,
            id="nullable-string",
        ),
        pytest.param(
            {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            "42",
            42,
            id="mixed-union",
        ),
    ],
)
def test_streaming_serialized_values_preserve_following_args(
    parser_cls, glm4_moe_tokenizer, chunk_size, schema, raw_value, expected
):
    """Partial values must not commit bytes that deserialization will change."""
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=[
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="update",
                    parameters={
                        "type": "object",
                        "properties": {
                            "value": schema,
                            "taskId": {"type": "string"},
                        },
                    },
                )
            )
        ],
    )
    text = (
        "<tool_call>update\n"
        f"<arg_key>value</arg_key><arg_value>{raw_value}</arg_value>"
        "<arg_key>taskId</arg_key><arg_value>1</arg_value></tool_call>"
    )
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    fragments = _stream_argument_fragments(
        parser_cls(glm4_moe_tokenizer), request, chunks
    )
    arguments = json.loads("".join(fragments))
    assert arguments == {"value": expected, "taskId": "1"}
    assert type(arguments["value"]) is type(expected)


@pytest.mark.parametrize(
    "parser_cls", [Glm4MoeModelToolParser, Glm47MoeModelToolParser]
)
@pytest.mark.parametrize("union_keyword", ["anyOf", "oneOf"])
@pytest.mark.parametrize("raw_value", ["123", "false", "null"])
def test_string_union_values_keep_their_type_and_stream_incrementally(
    parser_cls, glm4_moe_tokenizer, union_keyword, raw_value
):
    """String-only unions preserve JSON-looking strings in both API modes."""
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=[
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="update",
                    parameters={
                        "type": "object",
                        "properties": {
                            "value": {
                                union_keyword: [
                                    {
                                        "type": "string",
                                        "enum": ["123", "false", "null"],
                                    },
                                    {"type": "string", "const": "deleted"},
                                ]
                            }
                        },
                    },
                )
            )
        ],
    )
    text = (
        "<tool_call>update\n"
        f"<arg_key>value</arg_key><arg_value>{raw_value}</arg_value></tool_call>"
    )
    parser = parser_cls(glm4_moe_tokenizer)
    extracted = parser.extract_tool_calls(text, request)
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "value": raw_value
    }
    fragments = _stream_argument_fragments(parser, request, text)
    assert json.loads("".join(fragments)) == {"value": raw_value}
    assert len(fragments) > len(raw_value)


def _schema_case(name, schema, raw, expected, incremental=False, **root):
    parameters = {
        "type": "object",
        "properties": {"value": schema, "taskId": {"type": "string"}},
        "required": ["value", "taskId"],
        **root,
    }
    return pytest.param(parameters, raw, expected, incremental, id=name)


@pytest.mark.parametrize(
    "parameters, raw, expected, incremental",
    [
        _schema_case("string-type-array", {"type": ["string"]}, "123", "123", True),
        _schema_case("nullable-type-array", {"type": ["string", "null"]}, "123", "123"),
        _schema_case("nullable-false", {"type": ["string", "null"]}, "false", "false"),
        _schema_case("nullable-null", {"type": ["string", "null"]}, "null", None),
        _schema_case(
            "nullable-quoted-null", {"type": ["string", "null"]}, '"null"', "null"
        ),
        _schema_case("nullable-empty", {"type": ["string", "null"]}, "", ""),
        _schema_case(
            "nullable-whitespace", {"type": ["string", "null"]}, "  text\n", "  text\n"
        ),
        _schema_case("nullable-number", {"type": ["number", "null"]}, "1e3", 1000.0),
        _schema_case(
            "nullable-object", {"type": ["object", "null"]}, '{"id":"1"}', {"id": "1"}
        ),
        _schema_case("nullable-array", {"type": ["array", "null"]}, "null", None),
        _schema_case(
            "nullable-anyof-string",
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "123",
            "123",
        ),
        _schema_case(
            "nullable-anyof-null",
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "null",
            None,
        ),
        _schema_case(
            "nullable-oneof-string",
            {"oneOf": [{"type": "null"}, {"type": "string"}]},
            "false",
            "false",
        ),
        _schema_case(
            "nullable-oneof-null",
            {"oneOf": [{"type": "null"}, {"type": "string"}]},
            "null",
            None,
        ),
        _schema_case(
            "nullable-constrained",
            {"type": ["string", "null"], "enum": ["null", "123"]},
            "null",
            "null",
            True,
        ),
        _schema_case(
            "nullable-not-null",
            {"type": ["string", "null"], "not": {"type": "null"}},
            "null",
            "null",
        ),
        _schema_case(
            "string-whitespace", {"type": "string"}, "  text\n", "  text\n", True
        ),
        _schema_case(
            "string-literal-quotes", {"type": "string"}, '"null"', '"null"', True
        ),
        _schema_case(
            "enum-string", {"enum": ["123", "false", "null"]}, "false", "false", True
        ),
        _schema_case("enum-null", {"enum": ["123", None]}, "null", None),
        _schema_case("const-string", {"const": "false"}, "false", "false", True),
        _schema_case("const-null", {"const": None}, "null", None),
        _schema_case(
            "local-defs",
            {"$ref": "#/$defs/ID"},
            "123",
            "123",
            True,
            **{"$defs": {"ID": {"type": "string"}}},
        ),
        _schema_case(
            "legacy-definitions",
            {"$ref": "#/definitions/ID"},
            "123",
            "123",
            True,
            definitions={"ID": {"type": "string"}},
        ),
        _schema_case(
            "escaped-pointer",
            {"$ref": "#/$defs/a~1b~0c"},
            "123",
            "123",
            True,
            **{"$defs": {"a/b~c": {"type": "string"}}},
        ),
        _schema_case(
            "percent-encoded-pointer",
            {"$ref": "#/$defs/space%20name"},
            "123",
            "123",
            True,
            **{"$defs": {"space name": {"type": "string"}}},
        ),
        _schema_case(
            "anchor",
            {"$ref": "#identifier"},
            "123",
            "123",
            True,
            **{"$defs": {"ID": {"$anchor": "identifier", "type": "string"}}},
        ),
        _schema_case(
            "ref-chain",
            {"$ref": "#/$defs/A"},
            "123",
            "123",
            True,
            **{"$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"type": "string"}}},
        ),
        _schema_case(
            "nullable-ref-string",
            {"$ref": "#/$defs/ID"},
            "123",
            "123",
            **{"$defs": {"ID": {"type": ["string", "null"]}}},
        ),
        _schema_case(
            "nullable-ref-null",
            {"$ref": "#/$defs/ID"},
            "null",
            None,
            **{"$defs": {"ID": {"type": ["string", "null"]}}},
        ),
        _schema_case(
            "ref-sibling-intersection",
            {"$ref": "#/$defs/ID", "type": "string"},
            "123",
            "123",
            True,
            **{"$defs": {"ID": {"type": ["string", "integer"]}}},
        ),
        _schema_case(
            "draft7-ref-siblings-ignored",
            {"$ref": "#/definitions/ID", "type": "integer"},
            "123",
            "123",
            True,
            **{
                "$schema": "http://json-schema.org/draft-07/schema#",
                "definitions": {"ID": {"type": "string"}},
            },
        ),
        _schema_case(
            "draft4-unknown-const",
            {"const": "123"},
            "123",
            123,
            **{"$schema": "http://json-schema.org/draft-04/schema#"},
        ),
        _schema_case(
            "root-ref-literal-percent",
            {},
            "123",
            "123",
            True,
            **{
                "$ref": "#/$defs/Args%2520",
                "$defs": {"Args%20": {"properties": {"value": {"type": "string"}}}},
            },
        ),
        _schema_case(
            "allof-string-constraint",
            {"allOf": [{"type": "string"}, {"pattern": "^[0-9]+$"}]},
            "123",
            "123",
            True,
        ),
        _schema_case(
            "allof-intersection",
            {"allOf": [{"type": ["string", "null"]}, {"type": ["string", "integer"]}]},
            "123",
            "123",
            True,
        ),
        _schema_case(
            "allof-ref",
            {"allOf": [{"$ref": "#/$defs/ID"}, {"minLength": 3}]},
            "123",
            "123",
            True,
            **{"$defs": {"ID": {"type": "string"}}},
        ),
        _schema_case(
            "nested-compositions",
            {
                "allOf": [
                    {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    {"oneOf": [{"enum": ["123"]}, {"const": "false"}]},
                ]
            },
            "123",
            "123",
            True,
        ),
        _schema_case(
            "anyof-constraint-selects-string",
            {
                "anyOf": [
                    {"type": "integer", "minimum": 200},
                    {"type": "string", "pattern": "^123$"},
                ]
            },
            "123",
            "123",
        ),
        _schema_case(
            "oneof-exclusive-selects-string",
            {"oneOf": [{"type": "integer"}, {"type": "number"}, {"type": "string"}]},
            "123",
            "123",
        ),
        _schema_case(
            "allof-not-selects-string",
            {"allOf": [{"type": ["string", "integer"]}, {"not": {"type": "integer"}}]},
            "123",
            "123",
        ),
        _schema_case(
            "mixed-union-keeps-integer", {"type": ["string", "integer"]}, "123", 123
        ),
        _schema_case(
            "mixed-union-quoted-string", {"type": ["string", "integer"]}, '"123"', "123"
        ),
        _schema_case(
            "anyof-unconstrained-branch",
            {"anyOf": [{"type": "string"}, {}]},
            "123",
            123,
        ),
        _schema_case(
            "allof-number-integer",
            {"allOf": [{"type": "number"}, {"type": "integer"}]},
            "123",
            123,
        ),
        _schema_case("true-schema", True, "123", 123),
        _schema_case(
            "root-allof",
            {},
            "123",
            "123",
            True,
            allOf=[{"properties": {"value": {"type": "string"}}}],
        ),
        _schema_case(
            "root-anyof",
            {},
            "123",
            "123",
            True,
            anyOf=[
                {"properties": {"value": {"const": "123"}}},
                {"properties": {"value": {"const": "false"}}},
            ],
        ),
        _schema_case(
            "root-oneof",
            {},
            "123",
            "123",
            True,
            oneOf=[
                {"properties": {"value": {"const": "123"}}},
                {"properties": {"value": {"const": "false"}}},
            ],
        ),
        _schema_case(
            "root-ref",
            {},
            "123",
            "123",
            True,
            **{
                "$ref": "#/$defs/Args",
                "$defs": {"Args": {"properties": {"value": {"type": "string"}}}},
            },
        ),
        _schema_case(
            "pattern-property",
            {},
            "123",
            "123",
            True,
            patternProperties={"^val": {"type": "string"}},
        ),
        _schema_case(
            "additional-property",
            {},
            "123",
            "123",
            True,
            properties={"taskId": {"type": "string"}},
            additionalProperties={"type": "string"},
        ),
        _schema_case(
            "property-pattern-intersection",
            {"type": ["string", "integer"]},
            "123",
            "123",
            True,
            patternProperties={"^value$": {"type": "string"}},
        ),
        _schema_case(
            "additional-does-not-override-property",
            {"type": "integer"},
            "123",
            123,
            additionalProperties={"type": "string"},
        ),
        _schema_case(
            "recursive-object",
            {"$ref": "#/$defs/Node"},
            '{"id":"1","child":{"id":"2"}}',
            {"id": "1", "child": {"id": "2"}},
            **{
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "child": {"$ref": "#/$defs/Node"},
                        },
                    }
                }
            },
        ),
        _schema_case(
            "embedded-resource",
            {"$ref": "#/$defs/Inner"},
            "123",
            "123",
            True,
            **{
                "$id": "https://example.test/root",
                "$defs": {
                    "Inner": {
                        "$id": "inner/",
                        "$defs": {"ID": {"type": "string"}},
                        "$ref": "#/$defs/ID",
                    }
                },
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "parser_cls", [Glm4MoeModelToolParser, Glm47MoeModelToolParser]
)
@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
def test_schema_types_match_in_streaming_and_nonstreaming(
    parser_cls, glm4_moe_tokenizer, parameters, raw, expected, incremental, chunk_size
):
    validator = validator_for(parameters)(parameters)
    validator.check_schema(parameters)
    expected_args = {"value": expected, "taskId": "1"}
    validator.validate(expected_args)
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=[
            ChatCompletionToolsParam(
                function=FunctionDefinition(name="update", parameters=parameters)
            )
        ],
    )
    prefix = "<tool_call>update\n<arg_key>value</arg_key><arg_value>" + raw
    suffix = "</arg_value><arg_key>taskId</arg_key><arg_value>1</arg_value></tool_call>"
    text = prefix + suffix
    parser = parser_cls(glm4_moe_tokenizer)
    result = parser.extract_tool_calls(text, request)
    nonstreamed = json.loads(result.tool_calls[0].function.arguments)
    assert nonstreamed == expected_args
    assert type(nonstreamed["value"]) is type(expected)
    validator.validate(nonstreamed)
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    arguments = json.loads("".join(_stream_argument_fragments(parser, request, chunks)))
    assert arguments == expected_args
    assert type(arguments["value"]) is type(expected)
    validator.validate(arguments)
    if chunk_size == 1:
        # An ambiguous value must not emit irreversible raw bytes before its
        # closing tag; a string-only schema must still stream long content.
        partial = _stream_argument_fragments(
            parser_cls(glm4_moe_tokenizer), request, [prefix]
        )
        assert bool(partial) is incremental


@pytest.mark.parametrize(
    "parser_cls", [Glm4MoeModelToolParser, Glm47MoeModelToolParser]
)
@pytest.mark.parametrize(
    "reference", ["#/$defs/missing", "https://example.invalid/schema", "#/$defs/Cycle"]
)
def test_unresolvable_schema_keeps_following_arguments(
    parser_cls, glm4_moe_tokenizer, reference
):
    parameters = {
        "type": "object",
        "properties": {"value": {"$ref": reference}, "taskId": {"type": "string"}},
        "$defs": {"Cycle": {"$ref": "#/$defs/Cycle"}},
    }
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=[
            ChatCompletionToolsParam(
                function=FunctionDefinition(name="update", parameters=parameters)
            )
        ],
    )
    text = (
        "<tool_call>update\n<arg_key>value</arg_key><arg_value>123</arg_value>"
        "<arg_key>taskId</arg_key><arg_value>1</arg_value></tool_call>"
    )
    parser = parser_cls(glm4_moe_tokenizer)
    arguments = "".join(_stream_argument_fragments(parser, request, text))
    assert json.loads(arguments) == {"value": 123, "taskId": "1"}
    result = parser.extract_tool_calls(text, request)
    assert json.loads(result.tool_calls[0].function.arguments) == json.loads(arguments)


def test_streaming_content_between_tool_calls(glm4_moe_tool_parser, mock_request):
    """Text between completed tool calls is emitted exactly once."""
    _reset_streaming_state(glm4_moe_tool_parser)
    text = (
        "Before\n"
        "<tool_call>get_weather\n<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value></tool_call>"
        "\nBetween\n"
        "<tool_call>get_weather\n<arg_key>city</arg_key>"
        "<arg_value>Shanghai</arg_value></tool_call>"
    )

    delta = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=text,
        delta_text=text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    assert delta is not None
    assert delta.content == "Before\n\nBetween\n"
    assert len(glm4_moe_tool_parser.prev_tool_call_arr) == 2
    assert (
        json.loads(glm4_moe_tool_parser.prev_tool_call_arr[0]["arguments"])["city"]
        == "Beijing"
    )
    assert (
        json.loads(glm4_moe_tool_parser.prev_tool_call_arr[1]["arguments"])["city"]
        == "Shanghai"
    )


def test_remaining_unstreamed_args_uses_a_strict_prefix(glm4_moe_tool_parser):
    """Final flush returns only a suffix and rejects non-prefix state."""
    glm4_moe_tool_parser.prev_tool_call_arr = [
        {"arguments": '{"city":"Beijing","unit":"celsius"}'},
    ]
    glm4_moe_tool_parser.streamed_args_for_tool = ['{"city":"Beijing"']
    assert glm4_moe_tool_parser.get_remaining_unstreamed_args() == ',"unit":"celsius"}'

    glm4_moe_tool_parser.streamed_args_for_tool = ['{"city":"Beijing"}junk']
    assert glm4_moe_tool_parser.get_remaining_unstreamed_args() == ""


def test_remaining_unstreamed_args_serializes_dict_once(glm4_moe_tool_parser):
    """Dict parser state is serialized once, without double JSON encoding."""
    glm4_moe_tool_parser.prev_tool_call_arr = [
        {"arguments": {"city": "Beijing", "temperature": 20}},
    ]
    glm4_moe_tool_parser.streamed_args_for_tool = ['{"city": "Beijing"']
    remaining = glm4_moe_tool_parser.get_remaining_unstreamed_args()
    assert remaining == ', "temperature": 20}'


def test_glm47_reparse_streaming_uses_special_argument_regex(glm4_moe_tokenizer):
    parser = Glm47MoeModelToolParser(glm4_moe_tokenizer)
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=[
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                )
            )
        ],
    )
    text = (
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>\n"
        "<arg_value>Beijing</arg_value></tool_call>"
    )

    result = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=text,
        delta_text=text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=request,
    )

    assert result is not None
    assert json.loads(parser.prev_tool_call_arr[0]["arguments"]) == {"city": "Beijing"}


def test_reparse_state_resets_when_current_text_restarts(
    glm4_moe_tokenizer, mock_request
):
    parser = Glm4MoeModelToolParser(glm4_moe_tokenizer)
    parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="old content",
        delta_text="old content",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    text = "<tool_call>get_weather\n<arg_key>city</arg_key>"
    result = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=text,
        delta_text=text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    assert result is not None
    assert len(parser.prev_tool_call_arr) == 1
    assert parser.current_tool_id == 0


def test_parser_instances_do_not_share_reparse_state(glm4_moe_tokenizer, mock_request):
    parser_a = Glm4MoeModelToolParser(glm4_moe_tokenizer)
    parser_b = Glm4MoeModelToolParser(glm4_moe_tokenizer)
    text_a = "<tool_call>get_weather\n<arg_key>city</arg_key>"
    text_b = "<tool_call>get_weather\n<arg_key>city</arg_key>"

    parser_a.extract_tool_calls_streaming(
        previous_text="",
        current_text=text_a,
        delta_text=text_a,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )
    parser_b.extract_tool_calls_streaming(
        previous_text="",
        current_text=text_b,
        delta_text=text_b,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    assert parser_a.prev_tool_call_arr is not parser_b.prev_tool_call_arr
    assert parser_a.streamed_args_for_tool is not parser_b.streamed_args_for_tool
    assert parser_a.current_tool_id == parser_b.current_tool_id == 0
