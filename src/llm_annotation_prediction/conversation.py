from copy import deepcopy
from logging import getLogger
from typing import Any, Dict, List, Optional

import httpx
from pydantic import AnyUrl, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_annotation_prediction.handlers.handler import HandlerConfig
from llm_annotation_prediction.helpers.constants import Context
from llm_annotation_prediction.helpers.http import SimplifiedResponse, get_key_info
from llm_annotation_prediction.helpers.open_router import (
    ChoicesList,
    Message,
    NonStreamingChoice,
    ProviderPreferences,
    Providers,
    RequestDto,
    ResponseDto,
    ResponseError,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from llm_annotation_prediction.helpers.rate_limiter import RateLimitedQueue
from llm_annotation_prediction.helpers.save import (
    DictSerializable,
)
from llm_annotation_prediction.helpers.utils import get
from llm_annotation_prediction.publication import Publication
from llm_annotation_prediction.schema import Schema
from llm_annotation_prediction.turn import Turn, TurnConfig


class OpenRouterConversationConfig(BaseSettings):
    """
    Configuration base class for experiments.

    Derives from BaseSettings so secrets like API keys can be loaded from
    environment variables or .env files.
    """

    type: str = "OpenRouterConversationConfig"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Don't make HTTP calls to OpenRouter. Saves credits when testing.
    dry_run: bool = False

    # The OpenRouter API key. Will be read from the environment variable.
    api_key: Optional[str] = Field(alias="open_router_api_key", default=None)

    # The URL of the OpenRouter API. Defaults to the official API.
    api_url: AnyUrl = Field(
        default_factory=lambda: AnyUrl("https://openrouter.ai/api/v1/")
    )

    # Which model to use for the LLM conversation
    # https://openrouter.ai/docs/models
    model: str

    # The providers to use for this conversation in this order. If not set,
    # OpenRouter decides.
    # https://openrouter.ai/docs/features/provider-routing
    providers: Optional[List[Providers]] = None

    # The flow of the conversation is divided into turns. Each turn is responsible
    # for a request/response cycle.
    turns: List[TurnConfig[HandlerConfig]] = Field(default_factory=list)

    # Use custom validator to provide a clearer error message about .env
    @field_validator("api_key", mode="before")
    def check_api_key(cls, value: str) -> str:
        if not value:
            raise ValueError(
                "OpenRouter conversations require an API key as environment variable "
                "or through a .env file. See README.md."
            )
        return value


class OpenRouterConversation(DictSerializable):
    """
    Orchestrates a conversation with an LLM via OpenRouter. A conversation consists
    of turns, where the request and response are modified by handlers. They get the
    chance to read from or modify a context that persists over the whole conversation.

    Should other types of conversations become necessary, an interface should be
    extracted from this class.
    """

    def __init__(
        self,
        config: OpenRouterConversationConfig,
        rate_limiter: RateLimitedQueue[httpx.Response | None],
        publication: Publication,
        schema: Schema | None = None,
        trial: int | None = None,
    ) -> None:
        self._config = config
        self._rate_limiter = rate_limiter
        self._publication = publication
        self._schema = schema
        self._trial = trial

        # A conversation is identified by the publication UUID and the trial index
        self._logger = getLogger(f"Conv:{trial or 0}:{publication.uuid}")
        self._api_url = httpx.URL(str(config.api_url))

        # Retrieve information about the OpenRouter API key once
        if not self._config.dry_run:
            self._key_info = get_key_info(config.api_key, self._api_url)
            self._headers = {"Authorization": f"Bearer {config.api_key}"}

        self._failed: bool = False
        self._context: Context = self._create_context()
        self._turns = [Turn(turn_config, self._context) for turn_config in config.turns]

        # We separate the history of messages as the LLM will see it and the HTTP
        # communication history. Request handlers can add to
        # and modify the conversation history, because it is part of the request DTO.
        self._message_history: List[Message] = []
        self._http_history: List[SimplifiedResponse] = []

    async def converse(self) -> bool:
        """
        Organizes the conversation in turns and captures errors. Returns False if errors
        occured.
        """
        try:
            self._logger.info("Starting conversation")
            for turn in self._turns:
                await self._handle_turn(turn)
            self._logger.info("Conversation ended")
            self._context["succeeded"] = True
            return True
        except Exception as e:
            self._context["succeeded"] = False
            self._logger.error(f"Conversation failed with error: {e}", exc_info=True)

            # We attach the error to the message history, so it can be seen with
            # the show tool.
            self._message_history.append(
                UserMessage(role="system", content=f"[Error] {str(e)}")
            )
            return False

    async def _communicate(self, request_dto: RequestDto) -> ResponseDto | None:
        """
        Handles sending the request to OpenRouter and parsing the response. Logs time
        and communication even in failure cases.
        """
        self._logger.debug("Communicating with OpenRouter")

        response = None
        response_dto = None
        error_response = None

        try:
            response = await self._rate_limiter.enqueue(self._send_request, request_dto)
        except Exception as error:
            # In case of a status error, remember the response, so we can log it
            if isinstance(error, httpx.HTTPStatusError):
                if error.response:
                    error_response = error.response
            raise error
        finally:
            if not response:
                if not error_response:
                    raise ValueError(
                        f"No response received. Request DTO: {request_dto}"
                    )
                response = error_response

            # In case of an error response, we still log everything and try to parse
            # the DTO
            simplified_response = SimplifiedResponse.from_httpx_response(response)
            self._http_history.append(simplified_response)
            self._log_http_elapsed_time(simplified_response)

            response_dto = self._extract_response_dto(simplified_response)
            self._log_usage(response_dto)

        return response_dto

    async def _handle_turn(self, turn: Turn, is_tool_cycle: bool = False) -> None:
        """
        Handles one turn in a conversation. Dispatches (de)serialization and
        communication. If the LLM responds with tool calls, this turn is repeated
        with this additional information, so handlers can react accordingly.
        """
        request_dto = self._create_request_dto()
        request_dto = await turn.prepare_request(request_dto, is_tool_cycle)

        # The new history is what the request handlers decided, including potentially
        # new chat messages.
        self._message_history = deepcopy(request_dto.messages or [])

        response_dto = await self._communicate(request_dto)

        if response_dto:
            response_dto = await turn.parse_response(response_dto, is_tool_cycle)
            self._append_choices_to_history(response_dto.choices)

            # If there were tool calls in the response, a handler should have already
            # executed them. We need to send the results in an additional request to the
            # LLM, so it can respond with a text response.
            tool_calls = self._get_tool_calls(response_dto)
            if len(tool_calls) > 0:
                self._add_tool_messages_to_history(tool_calls)
                await self._handle_turn(turn, is_tool_cycle=True)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the message history and the context in a JSON-serializable dict.
        """
        self._logger.debug(
            f"Converting to dict. {len(self._message_history)} messages and "
            f"{len(self._http_history)} payloads."
        )
        return {
            "context": self._context,
            "conversation": [
                m.model_dump(exclude_none=True) for m in self._message_history
            ],
            "payloads": [m.model_dump(exclude_none=True) for m in self._http_history],
        }

    @property
    def uuid(self) -> str:
        """
        Conversations have 1:1 relationship to publications, so reuse their UUID.
        """
        return self._publication.uuid

    @property
    def failed(self) -> bool:
        """
        Indicates that the conversations failed for technical reasons.
        """
        return self._failed

    def _create_context(self) -> Context:
        """
        Creates the context dict and fills it with publication and schema information
        """
        context: Context = {"publication": self._publication.publication_text}

        if self._schema:
            context["schema"] = self._schema.collection

        return context

    def _create_request_dto(self) -> RequestDto:
        """
        Create a new request object with the chat history and other configurations set.
        """
        # "require_parameters" is currently in beta, but seems to work well.
        # Important to make sure that only providers are used that can handle
        # the parameters we require (structured output, tool calls, etc.)
        providers = ProviderPreferences(require_parameters=True)
        if self._config.providers:
            providers.allow_fallbacks = False
            providers.order = self._config.providers

        return RequestDto(
            model=self._config.model,
            messages=self._message_history,
            provider=providers,
        )

    async def _send_request(self, request_dto: RequestDto) -> httpx.Response | None:
        """
        Serialize payload and communicate with OpenRouter. Returns a simplified
        response to leave cleaner logs.
        """
        self._logger.debug("Sending request")

        if self._config.dry_run:
            self._logger.info("Dry run, not sending request")
            return None

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self._api_url.join("chat/completions"),
                headers=self._headers,
                json=request_dto.model_dump(exclude_none=True),
            )

        # Logging the complete request/response content will make the logs hard to read.
        # Only use when looking for specific errors.
        # self._logger.debug(f"Request headers: {response.request.headers}")
        # self._logger.debug(f"HTTPX request content: {response.request.content}")
        # self._logger.debug(f"Response headers: {response.headers}")
        # self._logger.debug(f"HTTPX response content: {response.content}")

        # Throw exception for technical errors, so the rate limiter can retry
        response.raise_for_status()

        return response

    def _extract_response_dto(self, response: SimplifiedResponse) -> ResponseDto:
        """
        Validate the payload. Assumes the HTTP request was successful, but handles
        contained errors.
        """
        try:
            dto = ResponseDto.model_validate(response.response_body)
        except ValidationError as validation_error:
            # When the content is not the expected answer, we'll try to fit the
            # error responses, so we can throw an appropriate exception
            self._failed = True
            wrapped_error = None
            try:
                error = ResponseError.model_validate(response.response_body)
                wrapped_error = RuntimeError(
                    f"{error.error.code}: {error.error.message}\n"
                    f"Metadata: {error.error.metadata}"
                )

            except Exception as error_parsing_exception:
                # This is a completely unexpected error, log as much as we can
                self._logger.error("Unexpected response from LLM")
                self._logger.debug(
                    f"Error response validation exception: {error_parsing_exception}"
                )
                self._logger.error(f"Original exception: {validation_error}")
                self._logger.error(f"Content: {response.response_body}")
                wrapped_error = RuntimeError(validation_error)

            # We should always have a wrapped error, but raising validation for safety
            if wrapped_error:
                raise wrapped_error from validation_error
            raise validation_error

        # Check for errors in the messages. These are not communication errors, but
        # errors in the LLM response.
        for choice in dto.choices:
            if not isinstance(choice, NonStreamingChoice):
                raise TypeError("Last choice in response is of unexpected type", choice)

            if choice.error is not None:
                raise ValueError("Choice error", choice)

            if choice.finish_reason not in ["stop", "tool_calls"]:
                raise ValueError(
                    f"Unexpected finish reason: {choice.finish_reason} "
                    f"(native: {choice.native_finish_reason})",
                )

        return dto

    def _append_choices_to_history(self, choices: ChoicesList) -> None:
        """
        Converts the LLM responses to the corresponding request models, so they
        can be sent as history in future turns.
        """

        for choice in choices:
            if not isinstance(choice, NonStreamingChoice):
                self._logger.warning(f"Found unhandled choice type: {choice}. Skipping")
                continue

            role = choice.message.role
            if role == "assistant":
                self._message_history.append(
                    UserMessage(
                        role="assistant",
                        content=choice.message.content,
                        tool_calls=choice.message.tool_calls,
                    )
                )
            else:
                raise NotImplementedError(
                    f"Unhandled LLM non-streaming choice: {choice}"
                )

    def _get_tool_calls(self, response_dto: ResponseDto) -> List[ToolCall]:
        """
        This method returns all tool calls in the response DTO.
        """
        all_tool_calls: List[ToolCall] = []
        for choice in response_dto.choices:
            tool_calls = get(choice, "message", "tool_calls")
            if tool_calls is not None and len(tool_calls) > 0:
                if len(all_tool_calls) > 0:
                    self._logger.warning(
                        "Found several choices with tool calls. This has not been tested."
                    )

                all_tool_calls.extend(tool_calls)
        return all_tool_calls

    def _add_tool_messages_to_history(self, tool_calls: List[ToolCall]) -> None:
        """
        Looks up the results of the tool calls in the context and adds it to the
        message history.
        """
        count: int = 0
        for call in tool_calls:
            # Check if result of a tool call actually exists
            if (
                "tool_calls" not in self._context
                or call.id not in self._context["tool_calls"]
            ):
                raise KeyError(
                    f"Tool call results not found (ID '{call.id}') "
                    "Did you forget to add the Tool handler to the response handlers?"
                )

            self._message_history.append(
                ToolMessage(
                    role="tool",
                    content=self._context["tool_calls"][call.id],
                    tool_call_id=call.id,
                    name=call.function.name,
                )
            )
            count += 1
        self._logger.info(f"Added {count} tool messages to history")

    def _log_http_elapsed_time(self, response: SimplifiedResponse) -> None:
        """
        Store the time it took to receive an LLM response in the context.
        """
        self._logger.info(
            f"Received response from OpenRouter ({response.status_code}) "
            f"after {response.elapsed_time:.2f} seconds"
        )

        if "http_elapsed_time" not in self._context:
            self._context["http_elapsed_time"] = []

        self._context["http_elapsed_time"].append(response.elapsed_time)

    def _log_usage(self, response_dto: ResponseDto) -> None:
        """
        If the response contains usage information, store it in the context.
        """
        if response_dto.usage:
            if "usage" not in self._context:
                self._context["usage"] = []

            self._context["usage"].append(
                response_dto.usage.model_dump(exclude_none=True)
            )
