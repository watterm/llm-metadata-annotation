from json import dumps
from logging import getLogger
from typing import Any, Dict

import json_repair
from jsonschema import validate

from llm_annotation_prediction.handlers.handler import Handler, HandlerConfig
from llm_annotation_prediction.helpers.open_router import (
    NonStreamingChoice,
    RequestDto,
    ResponseDto,
    ResponseFormat,
)

_logger = getLogger("Structured Output")


class StructuredOutputHandlerConfig(HandlerConfig):
    type: str = "StructuredOutput"

    # Decides when the LLM is forced to use the JSON output, because we cannot determine
    # if the LLM is going to make tool calls.
    apply_in_tool_cycle: bool = False

    # If set, the validated response is stored with this key in the context
    key_for_context_storage: str | None = None

    # The JSON schema the LLM's response has to follow.
    json_schema: dict[str, Any]


class StructuredOutputHandler(Handler):
    """
    Forces the LLM answer to be compliant with a JSON schema.
    """

    def __init__(self, config: StructuredOutputHandlerConfig, context: Dict[str, Any]):
        super().__init__(config, context)
        self._config: StructuredOutputHandlerConfig = config

        if "__ignore_types__" in self._config.json_schema:
            del self._config.json_schema["__ignore_types__"]

    async def handle_request(
        self, request_dto: RequestDto, is_tool_cycle: bool = False
    ) -> RequestDto:
        """
        Add the JSON schema to the requested response format.
        """

        if self._config.apply_in_tool_cycle == is_tool_cycle:
            _logger.debug("Enabling structured output")
            request_dto.response_format = ResponseFormat(
                type="json_schema",
                json_schema=self._config.json_schema,
            )
        else:
            _logger.debug("Not in targeted cycle: not requesting structured output")

        return request_dto

    async def handle_response(
        self, response_dto: ResponseDto, is_tool_cycle: bool = False
    ) -> ResponseDto:
        """
        Validate the response with the JSON schema.
        """
        _logger.debug("Handling response in structured output handler")

        if len(response_dto.choices) == 0:
            return response_dto

        if self._config.apply_in_tool_cycle == is_tool_cycle:
            last_choice = response_dto.choices[-1]
            if isinstance(last_choice, NonStreamingChoice):
                # This will throw a validation error if the LLM response does not follow
                # the schema. We let the exception bubble up, because the experiment failed
                # in this case.
                try:
                    _logger.debug("Validating JSON response from LLM")

                    # In case there are minor mistakes in the JSON string, we try to repair it
                    content_object = json_repair.loads(last_choice.message.content)

                    validate(content_object, self._config.json_schema["schema"])
                except Exception as e:
                    raise ValueError(
                        f"LLM did not emit valid JSON or follow output schema: {e}"
                    ) from e

                # Store in context for later analysis
                key = self._config.key_for_context_storage
                if key:
                    self._context[key] = content_object

                # Pretty print it back to the response as a markdown code block,
                # so it can be read by people
                last_choice.message.content = (
                    f"```json\n{dumps(content_object, indent=2)}\n```"
                )
        return response_dto
