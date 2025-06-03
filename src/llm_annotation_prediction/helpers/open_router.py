"""
OpenRouter module.

This modules defines all data transfer objects for the OpenRouter API and contains
helper functions.

The OpenRouter Docs unfortunately do not always contain complete
information about responses, therefore the types here throw errors on any
unrecognized payload fields, so we do not miss them. If you encounter missing fields,
that are not documented on OpenRouter.ai, it's worth looking at the OpenAI defintions
to get a fast fix:
https://github.com/openai/openai-python/tree/main/src/openai/types/chat
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------
# Definitions from https://openrouter.ai/docs/limits
# --------------------------------------------------------
class KeyInfo(StrictBaseModel):
    label: str
    usage: float
    limit: Optional[float]
    limit_remaining: Optional[float]  # Undocumented
    is_free_tier: bool
    rate_limit: dict[str, Any] = {"requests": int, "interval": str}
    is_provisioning_key: bool  # Undocumented


class KeyInfoWrap(StrictBaseModel):
    """
    API key information from OpenRouter. See https://openrouter.ai/docs/limits
    """

    data: KeyInfo


# --------------------------------------------------------
# Plugins
# --------------------------------------------------------

# So far only https://openrouter.ai/docs/web-search


class WebPlugin(StrictBaseModel):
    id: str = "web"
    max_results: int
    search_prompt: Optional[str] = None


# Use union of all plugins for request type
Plugins = WebPlugin


# --------------------------------------------------------
# Definitions from https://openrouter.ai/docs/requests
# --------------------------------------------------------
class TextContent(StrictBaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(StrictBaseModel):
    url: str
    # URL or base64 encoded image data
    detail: Optional[str]  # Optional, defaults to 'auto'


class ImageContentPart(StrictBaseModel):
    type: Literal["image_url"]  # URL or base64 encoded image data
    image_url: ImageUrl  # Optional, defaults to 'auto'


ContentPart = Union[TextContent, ImageContentPart]


class UserMessage(StrictBaseModel):
    role: Literal["user", "assistant", "system"]
    # ContentParts are only for the 'user' role:
    content: Union[str, List[ContentPart]]

    # If "name" is included, it will be prepended like this
    # for non-OpenAI models: `{name}: {content}`
    name: Optional[str] = None

    # Undocumented fields
    # The assistant can send tool call requests, which would be in here.
    tool_calls: Optional[List["ToolCall"]] = None


class ToolMessage(StrictBaseModel):
    role: Literal["tool"]
    content: str
    tool_call_id: str
    name: Optional[str] = None


Message = Annotated[Union[UserMessage, ToolMessage], BaseModel]


class FunctionDescription(StrictBaseModel):
    description: Optional[str] = None
    name: str
    parameters: dict[str, Any]  # JSON Schema object


class Tool(StrictBaseModel):
    type: Literal["function"]
    function: FunctionDescription


class FunctionName(StrictBaseModel):
    name: str


class Function(StrictBaseModel):
    type: Literal["function"]
    function: FunctionName


ToolChoice = Union[Literal["none", "auto", "required"], Function]


class Prediction(StrictBaseModel):
    type: Literal["content"]
    content: str


class ResponseFormat(StrictBaseModel):
    type: Literal["json_schema"]
    json_schema: Dict[str, Any]


class RequestDto(StrictBaseModel):
    """
    Data transfer object for OpenRouter requests. For more comments on the
    properties, see https://openrouter.ai/docs/requests
    """

    # Either "messages" or "prompt" is required
    messages: Optional[List[Message]] = None
    prompt: Optional[str] = None

    # See https://openrouter.ai/docs/models
    model: Optional[str] = None

    # Can be used for structure outputs following a JSON schema
    # See https://openrouter.ai/docs/structured-outputs
    response_format: Optional[ResponseFormat] = None

    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = None

    # LLM Parameters (https://openrouter.ai/docs/parameters)
    max_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    temperature: Optional[Annotated[float, Field(ge=0, le=2)]] = None
    top_p: Optional[Annotated[float, Field(gt=0, le=1)]] = None
    top_k: Optional[Annotated[int, Field(gt=1)]] = None
    frequency_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    presence_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    repetition_penalty: Optional[Annotated[float, Field(gt=0, le=2)]] = None
    seed: Optional[int] = None
    logit_bias: Optional[Dict[int, float]] = None
    top_logprobs: Optional[int] = None  # Not optional according to docs?

    # See models supporting tool calling: https://openrouter.ai/models?supported_parameters=tools
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = None

    # See https://openrouter.ai/docs/transforms
    # Setting the default to [] disables the automatic middle-out compression
    transforms: Optional[List[str]] = []

    # See https://openrouter.ai/docs/model-routing
    models: Optional[List[str]] = None
    route: Optional[Literal["fallback"]] = None

    # See https://openrouter.ai/docs/provider-routing
    provider: Optional["ProviderPreferences"] = None

    prediction: Optional[Prediction] = None

    # Whether to return the model's reasoning. Default false.
    # Text will appear in the "reasoning" field on each message prior to those
    # containing "content".
    include_reasoning: Optional[bool] = None

    plugins: Optional[List[Plugins]] = None


# --------------------------------------------------------
# Definitions from https://openrouter.ai/docs/responses
# --------------------------------------------------------
class Error(StrictBaseModel):
    code: int
    message: str
    metadata: Optional[Dict[str, Any]] = None


class ResponseError(StrictBaseModel):
    error: Error
    user_id: Optional[str] = None


class PromptTokenDetails(StrictBaseModel):
    cached_tokens: int


class CompletionTokenDetails(StrictBaseModel):
    reasoning_tokens: int


class ResponseUsage(StrictBaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Undocumented as of 25/03/26
    prompt_tokens_details: Optional[PromptTokenDetails] = None
    completion_tokens_details: Optional[CompletionTokenDetails] = None


class FunctionCall(StrictBaseModel):
    name: str
    arguments: str


class ToolCall(StrictBaseModel):
    id: str
    type: Literal["function"]
    function: FunctionCall

    # Undocumented fields:
    index: Optional[int] = None


class ResponseMessage(StrictBaseModel):
    content: str
    role: Optional[str]
    tool_calls: Optional[List[ToolCall]] = None

    # Undocumented fields:
    refusal: Optional[str] = None
    reasoning: Optional[str] = None


FinishReason = Literal[
    "tool_calls",
    "stop",
    "length",
    "content_filer",
    "error",
]


class NonChatChoice(StrictBaseModel):
    finish_reason: Optional[FinishReason]
    text: str
    error: Optional[Error] = None


class NonStreamingChoice(StrictBaseModel):
    finish_reason: Optional[str]
    message: ResponseMessage
    error: Optional[Error] = None

    # Undocumented fields:
    index: Optional[int] = None

    # Not implementing more details, because we currently don't need it.
    logprobs: Optional[dict[str, Any]] = None
    native_finish_reason: Optional[str] = None


class StreamingChoice(StrictBaseModel):
    finish_reason: Optional[str]
    delta: ResponseMessage
    error: Optional[Error] = None


ChoicesList = List[Union[NonStreamingChoice, StreamingChoice, NonChatChoice]]


class ResponseDto(StrictBaseModel):
    id: str
    choices: ChoicesList = []
    created: int
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    system_fingerprint: Optional[str] = None
    usage: Optional[ResponseUsage] = None

    # Undocumented fields:

    # Used by perplexity models to provide URLs.
    citations: Optional[List[str]] = None

    # OpenRouter information who actually executed the LLM call
    provider: Optional[str] = None


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


class ProviderPreferences(StrictBaseModel):
    allow_fallbacks: Optional[bool] = None
    require_parameters: Optional[bool] = None
    data_collection: Optional[Literal["deny", "allow"]] = None
    order: Optional[List[Providers]] = None
    ignore: Optional[List[Providers]] = None
    quantizations: Optional[
        List[Literal["int4", "int8", "fp8", "fp16", "bf16", "unknown"]]
    ] = None

    class Config:
        extra = "forbid"
