"""OpenRouter API client wrapper with tool integration."""

import json
import os
from typing import override

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.shared_params.function_definition import FunctionDefinition

from ..tools.base import Tool, ToolCall
from ..utils.config import ModelParameters
from .base_client import BaseLLMClient
from .llm_basics import LLMMessage, LLMResponse, LLMUsage
from .retry_utils import retry_with


class OpenRouterClient(BaseLLMClient):
    """OpenRouter client wrapper with tool schema generation."""

    def __init__(self, model_parameters: ModelParameters):
        super().__init__(model_parameters)

        # Use OpenAI SDK with OpenRouter's base URL
        self.client: openai.OpenAI = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.message_history: list[ChatCompletionMessageParam] = []

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    def _create_openrouter_response(
        self,
        model_parameters: ModelParameters,
        tool_schemas: list[ChatCompletionToolParam] | None,
        extra_headers: dict[str, str] | None = None,
    ) -> ChatCompletion:
        """Create a response using OpenRouter API. This method will be decorated with retry logic."""
        return self.client.chat.completions.create(
            model=model_parameters.model,
            messages=self.message_history,
            tools=tool_schemas if tool_schemas else openai.NOT_GIVEN,
            temperature=model_parameters.temperature,
            top_p=model_parameters.top_p,
            max_tokens=model_parameters.max_tokens,
            extra_headers=extra_headers if extra_headers else None,
            n=1,
        )

    @override
    def chat(
        self,
        messages: list[LLMMessage],
        model_parameters: ModelParameters,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages to OpenRouter with optional tool support."""
        openrouter_messages = self.parse_messages(messages)
        if reuse_history:
            self.message_history = self.message_history + openrouter_messages
        else:
            self.message_history = openrouter_messages

        tool_schemas = None
        # Add tools if provided
        if tools:
            tool_schemas = [
                ChatCompletionToolParam(
                    function=FunctionDefinition(
                        name=tool.get_name(),
                        description=tool.get_description(),
                        parameters=tool.get_input_schema(),
                    ),
                    type="function",
                )
                for tool in tools
            ]

        # Set up extra headers for OpenRouter
        extra_headers: dict[str, str] = {}

        openrouter_site_url = os.getenv("OPENROUTER_SITE_URL")
        if openrouter_site_url:
            extra_headers["HTTP-Referer"] = openrouter_site_url
        openrouter_size_name = os.getenv("OPENROUTER_SITE_NAME")
        if openrouter_size_name:
            extra_headers["X-Title"] = openrouter_size_name

        # Apply retry decorator to the API call
        retry_decorator = retry_with(
            func=self._create_openrouter_response,
            service_name="OpenRouter",
            max_retries=model_parameters.max_retries,
            provider_name="openrouter",
        )
        response = retry_decorator(model_parameters, tool_schemas, extra_headers)

        choice = response.choices[0]

        tool_calls: list[ToolCall] | None = None
        if choice.message.tool_calls:
            tool_calls = []
            for tool_call in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tool_call.function.name,
                        call_id=tool_call.id,
                        arguments=(
                            json.loads(tool_call.function.arguments)
                            if tool_call.function.arguments
                            else {}
                        ),
                    )
                )

        llm_response = LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model=response.model,
            usage=(
                LLMUsage(
                    input_tokens=response.usage.prompt_tokens or 0,
                    output_tokens=response.usage.completion_tokens or 0,
                )
                if response.usage
                else None
            ),
        )

        # update message history
        if llm_response.tool_calls:
            self.message_history.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=llm_response.content,
                    tool_calls=[
                        ChatCompletionMessageToolCallParam(
                            id=tool_call.call_id,
                            function=Function(
                                name=tool_call.name,
                                arguments=json.dumps(tool_call.arguments),
                            ),
                            type="function",
                        )
                        for tool_call in llm_response.tool_calls
                    ],
                )
            )
        elif llm_response.content:
            self.message_history.append(
                ChatCompletionAssistantMessageParam(content=llm_response.content, role="assistant")
            )

        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="openrouter",
                model=model_parameters.model,
                tools=tools,
            )

        return llm_response

    @override
    def supports_tool_calling(self, model_parameters: ModelParameters) -> bool:
        """Check if the current model supports tool calling."""
        # Most modern models on OpenRouter support tool calling
        # We'll be conservative and check for known capable models
        tool_capable_patterns = [
            "gpt-4",
            "gpt-3.5-turbo",
            "claude-3",
            "claude-2",
            "gemini",
            "mistral",
            "llama-3",
            "command-r",
        ]
        return any(pattern in model_parameters.model.lower() for pattern in tool_capable_patterns)

    def parse_messages(self, messages: list[LLMMessage]) -> list[ChatCompletionMessageParam]:
        openrouter_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            match msg:
                case msg if msg.tool_call is not None:
                    _msg_tool_call_handler(openrouter_messages, msg)
                case msg if msg.tool_result is not None:
                    _msg_tool_result_handler(openrouter_messages, msg)
                case msg if msg.role is not None:
                    _msg_role_handler(openrouter_messages, msg)
                case _:
                    raise ValueError(f"Invalid message: {msg}")

        return openrouter_messages


def _msg_tool_call_handler(messages: list[ChatCompletionMessageParam], msg: LLMMessage) -> None:
    if msg.tool_call:
        messages.append(
            ChatCompletionFunctionMessageParam(
                content=json.dumps(
                    {
                        "name": msg.tool_call.name,
                        "arguments": msg.tool_call.arguments,
                    }
                ),
                role="function",
                name=msg.tool_call.name,
            )
        )


def _msg_tool_result_handler(messages: list[ChatCompletionMessageParam], msg: LLMMessage) -> None:
    if msg.tool_result:
        result: str = ""
        if msg.tool_result.result:
            result = result + msg.tool_result.result + "\n"
        if msg.tool_result.error:
            result += "Tool call failed with error:\n"
            result += msg.tool_result.error
        result = result.strip()
        messages.append(
            ChatCompletionToolMessageParam(
                content=result,
                role="tool",
                tool_call_id=msg.tool_result.call_id,
            )
        )


def _msg_role_handler(messages: list[ChatCompletionMessageParam], msg: LLMMessage) -> None:
    if msg.role:
        match msg.role:
            case "system":
                if not msg.content:
                    raise ValueError("System message content is required")
                messages.append(
                    ChatCompletionSystemMessageParam(content=msg.content, role="system")
                )
            case "user":
                if not msg.content:
                    raise ValueError("User message content is required")
                messages.append(ChatCompletionUserMessageParam(content=msg.content, role="user"))
            case "assistant":
                if not msg.content:
                    raise ValueError("Assistant message content is required")
                messages.append(
                    ChatCompletionAssistantMessageParam(content=msg.content, role="assistant")
                )
            case _:
                raise ValueError(f"Invalid message role: {msg.role}")