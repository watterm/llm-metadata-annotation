from logging import getLogger
from typing import Any, Dict, Optional

from llm_annotation_prediction.handlers.handler import Handler, HandlerConfig
from llm_annotation_prediction.helpers.constants import Context
from llm_annotation_prediction.helpers.open_router import (
    RequestDto,
    ResponseDto,
    UserMessage,
)

_logger = getLogger("AddUserMessageHandler")


class AddUserMessageHandlerConfig(HandlerConfig):
    type: str = "AddUserMessageHandler"

    # The message to be added. Can contain placeholders in curly braces, that
    # will be filled from the context.
    # Example: The publication text is {publication}
    message: str = ""

    # Allows to add variables on top of the original context to be used in the message
    additional_context: Optional[Dict[str, Any]] = None


class AddUserMessageHandler(Handler):
    """
    Adds a message from the user to the current conversation. The message can
    contain placeholders in curly braces {}, which will be filled in from
    the context object of the conversation.
    """

    def __init__(self, config: AddUserMessageHandlerConfig, context: Context):
        super().__init__(config, context)
        self._config: AddUserMessageHandlerConfig = config

    async def handle_request(
        self, request_dto: RequestDto, is_tool_cycle: bool = False
    ) -> RequestDto:
        _logger.debug("Adding user message")
        if not is_tool_cycle:
            format_vars = dict(self._context)
            if self._config.additional_context:
                format_vars.update(self._config.additional_context)

            text = self._config.message.format(**format_vars)

            if request_dto.messages is None:
                request_dto.messages = []
            request_dto.messages.append(UserMessage(role="user", content=text))
        return request_dto

    async def handle_response(
        self, response_dto: ResponseDto, is_tool_cycle: bool = False
    ) -> ResponseDto:
        raise TypeError(f"{__name__} cannot be used for response handling")
