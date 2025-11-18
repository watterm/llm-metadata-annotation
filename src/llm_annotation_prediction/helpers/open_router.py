"""OpenRouter module.

This modules defines all data transfer objects for the OpenRouter API and contains
helper functions.

The OpenRouter Docs unfortunately do not always contain complete
information about responses, therefore the types here throw errors on any
unrecognized payload fields, so we do not miss them. If you encounter missing fields,
that are not documented on OpenRouter.ai, it's worth looking at the OpenAI defintions
to get a fast fix:
https://github.com/openai/openai-python/tree/main/src/openai/types/chat
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# Can be used to check OpenRouter API deviation from these classes
class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------
# Definitions from https://openrouter.ai/docs/limits
# --------------------------------------------------------
class KeyInfo(BaseModel):
    label: str
    usage: float
    limit: float | None
    limit_remaining: float | None  # Undocumented
    is_free_tier: bool
    is_provisioning_key: bool  # Undocumented


class KeyInfoWrap(BaseModel):
    """
    API key information from OpenRouter. See https://openrouter.ai/docs/limits
    """

    data: KeyInfo


# --------------------------------------------------------
# Plugins
# --------------------------------------------------------

# Web Search Plugin
# https://openrouter.ai/docs/features/web-search
WebSearchEngine = Literal["exa", "native"]


class WebPlugin(BaseModel):
    id: Literal["web"] = "web"
    engine: WebSearchEngine | None = None
    max_results: int
    search_prompt: str | None = None


SearchContextSize = Literal["low", "medium", "high"]


# This is used in the request DTO
class WebSearchOptions(BaseModel):
    search_context_size: SearchContextSize = "medium"


# These will appear in the response
class UrlCitation(BaseModel):
    url: str
    start_index: int | None = None
    end_index: int | None = None
    title: str | None = None
    content: str | None = None


# PDF file attachment Plugin
# https://openrouter.ai/docs/features/multimodal/pdfs
PdfEngine = Literal["mistral-ocr", "pdf-text", "native"]


class PdfPlugin(BaseModel):
    engine: PdfEngine


class FileParserPlugin(BaseModel):
    id: Literal["file-parser"] = "file-parser"
    pdf: PdfPlugin


class FileAnnotation(BaseModel):
    hash: str
    name: str
    content: list[ContentPart]


# Generic Plugin classes
# Plugins seem to add annotations to the response messages.
# They are not well documented, some is extracted from the responses.
class Annotation(BaseModel):
    type: Literal["url_citation", "file"] = "url_citation"
    url_citation: UrlCitation | None = None
    file: FileAnnotation | None = None


# Use union of all plugins for request type
Plugins = WebPlugin | FileParserPlugin


# --------------------------------------------------------
# Definitions from https://openrouter.ai/docs/requests
# --------------------------------------------------------


# Content parts for multimodal messages
class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrl(BaseModel):
    url: str
    # URL or base64 encoded image data
    detail: str | None = None  # Optional, defaults to 'auto'


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"  # URL or base64 encoded image data
    image_url: ImageUrl  # Optional, defaults to 'auto'


# Not described in the chat endpoint, just in https://openrouter.ai/docs/features/multimodal/pdfs
class FileContentData(BaseModel):
    filename: str
    file_data: str  # Can be a URL or base64 encoded file data


class FileContent(BaseModel):
    type: Literal["file"] = "file"
    file: FileContentData


ContentPart = TextContent | ImageContent | FileContent
Role = Literal["user", "assistant", "system"]


# Message types
class UserMessage(BaseModel):
    role: Role

    # ContentParts are only for the 'user' role:
    content: str | list[ContentPart] | None

    # If "name" is included, it will be prepended like this
    # for non-OpenAI models: `{name}: {content}`
    name: str | None = None

    # Undocumented fields
    # The assistant can send tool call requests, which would be in here.
    tool_calls: list[ToolCall] | None = None

    # This is part of the response when web search citations are used
    annotations: list[Annotation] | None = None


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str
    name: str | None = None


Message = Annotated[UserMessage | ToolMessage, BaseModel]


class FunctionDescription(BaseModel):
    description: str | None = None
    name: str
    parameters: dict[str, Any]  # JSON Schema object


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDescription


class FunctionName(BaseModel):
    name: str


# This definition is only needed for enforcing a single tool
class Function(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionName


ToolChoice = Literal["none", "auto", "required"] | Function


class Prediction(BaseModel):
    type: Literal["content"] = "content"
    content: str


class ResponseFormat(BaseModel):
    type: Literal["json_schema"] = "json_schema"
    json_schema: dict[str, Any]


# https://openrouter.ai/docs/use-cases/reasoning-tokens
class Reasoning(BaseModel):
    # OpenRouter says exactly one of these two is required
    effort: Literal["low", "medium", "high"] | None = "medium"
    max_tokens: int | None = None

    # Set to true to exclude reasoning tokens from response
    exclude: bool | None = None

    # Default: inferred from `effort` or `max_tokens`
    enabled: bool | None = None


class RequestDto(BaseModel):
    """
    Data transfer object for OpenRouter requests. For more comments on the
    properties, see https://openrouter.ai/docs/requests
    """

    # Either "messages" or "prompt" is required
    messages: list[Message] | None = None
    prompt: str | None = None

    # See https://openrouter.ai/docs/models
    model: str | None = None

    # Can be used for structure outputs following a JSON schema
    # See https://openrouter.ai/docs/structured-outputs
    response_format: ResponseFormat | None = None

    stop: str | list[str] | None = None
    stream: bool | None = None

    # LLM Parameters (https://openrouter.ai/docs/parameters)
    max_tokens: Annotated[int, Field(ge=1)] | None = None
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    top_p: Annotated[float, Field(gt=0, le=1)] | None = None
    top_k: Annotated[int, Field(gt=1)] | None = None
    frequency_penalty: Annotated[float, Field(ge=-2, le=2)] | None = None
    presence_penalty: Annotated[float, Field(ge=-2, le=2)] | None = None
    repetition_penalty: Annotated[float, Field(gt=0, le=2)] | None = None
    seed: int | None = None
    logit_bias: dict[int, float] | None = None
    top_logprobs: int | None = None  # Not optional according to docs?

    # See models supporting tool calling: https://openrouter.ai/models?supported_parameters=tools
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None

    # See https://openrouter.ai/docs/transforms
    # Setting the default to [] disables the automatic middle-out compression
    transforms: list[str] | None = []

    # See https://openrouter.ai/docs/model-routing
    models: list[str] | None = None
    route: Literal["fallback"] | None = None

    # See https://openrouter.ai/docs/provider-routing
    provider: ProviderPreferences | None = None

    prediction: Prediction | None = None

    # Whether to return the model's reasoning. Default false.
    # Text will appear in the "reasoning" field on each message prior to those
    # containing "content".
    include_reasoning: bool | None = None

    plugins: list[Plugins] | None = None

    # https://openrouter.ai/docs/features/web-search#specifying-search-context-size
    web_search_options: WebSearchOptions | None = None


# --------------------------------------------------------
# Definitions from https://openrouter.ai/docs/responses
# --------------------------------------------------------
class Error(BaseModel):
    code: int
    message: str
    metadata: dict[str, Any] | None = None


class ResponseError(BaseModel):
    error: Error
    user_id: str | None = None


class PromptTokenDetails(BaseModel):
    cached_tokens: int


class CompletionTokenDetails(BaseModel):
    reasoning_tokens: int
    image_tokens: int | None = None


class ResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Undocumented as of 25/03/26
    prompt_tokens_details: PromptTokenDetails | None = None
    completion_tokens_details: CompletionTokenDetails | None = None


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    type: Literal["function"] = "function"
    id: str
    function: FunctionCall

    # Undocumented fields:
    index: int | None = None


class ResponseMessage(BaseModel):
    content: str | None = None
    role: str
    tool_calls: list[ToolCall] | None = None

    # Seems not fully documented
    # https://openrouter.ai/docs/use-cases/reasoning-tokens
    reasoning: str | None = None
    reasoning_details: Any | None = None

    # Undocumented fields:
    refusal: str | None = None
    annotations: list[Annotation] | None = None  # Results of plugins


FinishReason = Literal[
    "tool_calls",
    "stop",
    "length",
    "content_filer",
    "error",
]


class NonChatChoice(BaseModel):
    finish_reason: FinishReason | None
    text: str
    error: Error | None = None


class NonStreamingChoice(BaseModel):
    finish_reason: str | None
    message: ResponseMessage
    error: Error | None = None

    # Undocumented fields:
    index: int | None = None

    # Not implementing more details, because we currently don't need it.
    logprobs: dict[str, Any] | None = None
    native_finish_reason: str | None = None


class StreamingChoice(BaseModel):
    finish_reason: str | None
    delta: ResponseMessage
    error: Error | None = None


ChoicesList = list[NonStreamingChoice | StreamingChoice | NonChatChoice]


class ResponseDto(BaseModel):
    id: str
    choices: ChoicesList = []
    created: int
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    system_fingerprint: str | None = None
    usage: ResponseUsage | None = None

    # Undocumented fields:

    # Used by perplexity models to provide URLs.
    citations: list[str] | None = None

    # OpenRouter information who actually executed the LLM call
    provider: str | None = None


# --------------------------------------------------------
# https://openrouter.ai/docs/provider-routing
# --------------------------------------------------------

Providers = Literal[
    "OpenAI",
    "Anthropic",
    "Google",
    "Google AI Studio",
    "Groq",
    "SambaNova",
    "Cohere",
    "Mistral",
    "Together",
    "Together 2",
    "Fireworks",
    "DeepInfra",
    "Lepton",
    "Novita",
    "Avian",
    "Lambda",
    "Azure",
    "Modal",
    "AnyScale",
    "Replicate",
    "Perplexity",
    "Recursal",
    "OctoAI",
    "DeepSeek",
    "Infermatic",
    "AI21",
    "Featherless",
    "Inflection",
    "xAI",
    "01.AI",
    "HuggingFace",
    "Mancer",
    "Mancer 2",
    "Hyperbolic",
    "Hyperbolic 2",
    "Lynn 2",
    "Lynn",
    "Reflection",
]


class ProviderPreferences(BaseModel):
    allow_fallbacks: bool | None = None
    require_parameters: bool | None = None
    data_collection: Literal["deny", "allow"] | None = None
    order: list[Providers] | None = None
    ignore: list[Providers] | None = None
    quantizations: (
        list[Literal["int4", "int8", "fp8", "fp16", "bf16", "unknown"]] | None
    ) = None
