import logging
from typing import Generic, List, TypeVar

from pydantic import BaseModel

from llm_annotation_prediction.handlers.handler import Handler, HandlerConfig
from llm_annotation_prediction.helpers.constants import Context
from llm_annotation_prediction.helpers.open_router import RequestDto, ResponseDto
from llm_annotation_prediction.helpers.utils import load_class

_logger = logging.getLogger("Turn")

HandlerConfigType = TypeVar("HandlerConfigType", bound="HandlerConfig")


class TurnConfig(BaseModel, Generic[HandlerConfigType]):
    type: str = "TurnConfig"
    name: str

    request_handlers: list[HandlerConfigType] = []
    response_handlers: list[HandlerConfigType] = []


class Turn:
    def __init__(self, config: TurnConfig[HandlerConfig], context: Context):
        self._config = config
        self._context = context

        self._request_handlers: list[Handler] = self._create_handlers(
            config.request_handlers
        )
        self._response_handlers: list[Handler] = self._create_handlers(
            config.response_handlers
        )

    def _create_handlers(self, configs: list[HandlerConfig]) -> List[Handler]:
        """
        Instantiate the list of handlers by considering their type configuration,
        so the correct subclass can be picked.
        """
        return [load_class(c.type)(c, self._context) for c in configs]

    async def prepare_request(
        self, request_dto: RequestDto, is_tool_cycle: bool = False
    ) -> RequestDto:
        """
        Apply all request handlers to translate the configured features
        into a request object.
        """
        _logger.debug("Preparing request")

        for handler in self._request_handlers:
            request_dto = await handler.handle_request(request_dto, is_tool_cycle)
        return request_dto

    async def parse_response(
        self, response_dto: ResponseDto, is_tool_cycle: bool = False
    ) -> ResponseDto:
        """
        Apply all response handlers to extract or transform information from the LLM
        response.
        """
        _logger.debug("Parsing response")
        for handler in self._response_handlers:
            response_dto = await handler.handle_response(response_dto, is_tool_cycle)
        return response_dto
