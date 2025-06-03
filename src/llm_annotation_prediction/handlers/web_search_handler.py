from logging import getLogger
from typing import Any, Dict, Optional

from llm_annotation_prediction.handlers.handler import Handler, HandlerConfig
from llm_annotation_prediction.helpers.open_router import (
    RequestDto,
    ResponseDto,
    WebPlugin,
)

_logger = getLogger("WebSearchHandler")


class WebSearchHandlerConfig(HandlerConfig):
    type: str = "WebSearchHandler"

    # Maximum number of results shown to the LLM
    max_results: int = 3

    # Optional prompt to use for the web search
    search_prompt: Optional[str] = None


class WebSearchHandler(Handler):
    """
    Enables web search in OpenRouter by adding the plugin definition to the request.
    """

    def __init__(self, config: WebSearchHandlerConfig, context: Dict[str, Any]):
        super().__init__(config, context)
        self._config: WebSearchHandlerConfig = config

    async def handle_request(
        self, request_dto: RequestDto, is_tool_cycle: bool = False
    ) -> RequestDto:
        _logger.debug("Enabling web search")

        plugin = WebPlugin(
            max_results=self._config.max_results,
            search_prompt=self._config.search_prompt,
        )

        request_dto.plugins = (request_dto.plugins or []) + [plugin]
        return request_dto

    async def handle_response(
        self, response_dto: ResponseDto, is_tool_cycle: bool = False
    ) -> ResponseDto:
        raise TypeError(f"{__name__} cannot be used for response handling")
