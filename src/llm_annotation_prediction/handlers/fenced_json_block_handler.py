import json
from logging import getLogger
from typing import Any, Dict

from jsonschema import validate

from llm_annotation_prediction.handlers.handler import Handler, HandlerConfig
from llm_annotation_prediction.helpers.open_router import (
    NonStreamingChoice,
    RequestDto,
    ResponseDto,
)

_logger = getLogger("Structured Output")


class FencedJsonBlockHandlerConfig(HandlerConfig):
    type: str = "FencedJsonBlock"

    # If set, the validated response is stored with this key in the context
    key_for_context_storage: str | None = None

    # The JSON schema the LLM's response has to follow.
    json_schema: dict[str, Any]

    # If set, stops the conversation when no valid json response is found
    fail_on_parsing_error: bool = True


class FencedJsonBlockHandler(Handler):
    """
    Parses the LLM answer for a code block containing a JSON object and validates
    it with a JSON schema. This is an alternative to the structured output handler
    for models and providers that do not support structured output natively.
    """

    def __init__(self, config: FencedJsonBlockHandlerConfig, context: Dict[str, Any]):
        super().__init__(config, context)
        self._config: FencedJsonBlockHandlerConfig = config

        if "__ignore_types__" in self._config.json_schema:
            del self._config.json_schema["__ignore_types__"]

    async def handle_request(
        self, request_dto: RequestDto, is_tool_cycle: bool = False
    ) -> RequestDto:
        raise TypeError(f"{__name__} cannot be used for request handling")

    async def handle_response(
        self, response_dto: ResponseDto, is_tool_cycle: bool = False
    ) -> ResponseDto:
        """
        Parses the LLM answer for a code block containing a JSON object and validates
        it with a JSON schema.
        """
        if len(response_dto.choices) == 0:
            return response_dto

        _logger.debug("Parsing response for fenced JSON block")

        last_choice = response_dto.choices[-1]
        if not isinstance(last_choice, NonStreamingChoice):
            _logger.error("Last choice is not a NonStreamingChoice, skipping")
            return response_dto

        try:
            # Extract the JSON block from the response
            json_string = self._extract_json_block(last_choice.message.content)

            # Validate the JSON block against the schema
            json_object = self._validate_json(json_string)

            # Store the validated JSON block in the context if specified
            if self._config.key_for_context_storage:
                self._context[self._config.key_for_context_storage] = json_object
        except Exception as error:
            if self._config.fail_on_parsing_error:
                raise error
            else:
                _logger.warning(f"Continuing despite error: {error}")

        return response_dto

    def _extract_json_block(self, message: str) -> str:
        """
        Extracts the JSON block from the response string and makes sure it's unique.
        """
        # Find the first code block in the response
        start = message.find("```json")
        end = message.find("```", start + 7)

        if start == -1 or end == -1:
            raise ValueError("No JSON block found in the response")

        if message.find("```json", end) != -1:
            raise ValueError("Multiple JSON blocks found in the response")

        # Extract and return the JSON block
        return message[start + 7 : end].strip()

    def _validate_json(self, json_block: str) -> Any:
        """
        Validates the JSON block against the schema.
        """
        try:
            # Parse the JSON block
            json_object = json.loads(json_block)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON block: {e}") from e

        # Validate the JSON object against the schema
        try:
            validate(json_object, self._config.json_schema["schema"])
        except Exception as e:
            raise ValueError(f"JSON block does not match schema: {e}") from e

        return json_object
