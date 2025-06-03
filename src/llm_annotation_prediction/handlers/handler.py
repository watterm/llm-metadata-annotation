from abc import ABC, abstractmethod
from logging import getLogger

from pydantic import BaseModel

from llm_annotation_prediction.helpers.constants import Context
from llm_annotation_prediction.helpers.open_router import (
    RequestDto,
    ResponseDto,
)

_logger = getLogger("Handler")


class HandlerConfig(BaseModel):
    type: str = "Handler"


class Handler(ABC):
    """
    Subclasses of the Handler base class can modify requests and responses in
    conversations, so each turn of a conversation is composed of a list of request
    and response handlers in order to implement a flow.
    """

    def __init__(self, config: HandlerConfig, context: Context):
        self._config = config
        self._context = context

    @abstractmethod
    async def handle_request(
        self, request_dto: RequestDto, is_tool_cycle: bool = False
    ) -> RequestDto:
        pass

    @abstractmethod
    async def handle_response(
        self, response_dto: ResponseDto, is_tool_cycle: bool = False
    ) -> ResponseDto:
        pass
