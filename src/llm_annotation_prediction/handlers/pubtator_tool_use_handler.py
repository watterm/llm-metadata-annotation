from asyncio import TaskGroup
from logging import getLogger
from typing import Any, List

from llm_annotation_prediction.handlers.handler import Handler, HandlerConfig
from llm_annotation_prediction.helpers.constants import Context
from llm_annotation_prediction.helpers.open_router import (
    NonStreamingChoice,
    RequestDto,
    ResponseDto,
    ToolCall,
)
from llm_annotation_prediction.helpers.pubtator.common import PubtatorStrategy
from llm_annotation_prediction.helpers.pubtator.find_entity_by_pub_search import (
    FindEntityByPublicationSearchStrategy,
)
from llm_annotation_prediction.helpers.pubtator.find_entity_id import (
    FindEntityIdStrategy,
)
from llm_annotation_prediction.helpers.utils import get

_logger = getLogger("PubtatorToolUse")


class PubtatorToolUseHandlerConfig(HandlerConfig):
    type: str = "PubtatorToolUse"

    # Fallback to the FindEntityId endpoint. Otherwise the entity is extracted via
    # the publication search.
    use_find_entity_id_endpoint: bool = False

    # Forces the model to use the Pubtator tool.
    force_tool_use: bool = False


class PubtatorToolUseHandler(Handler):
    """
    Enables Pubtator searches in conversations. The request provides information
    about the pubtator tool and the response will include the actual calls the LLM
    wants to make.
    """

    _strategy: PubtatorStrategy[Any, Any]

    def __init__(self, config: PubtatorToolUseHandlerConfig, context: Context):
        super().__init__(config, context)
        self._config: PubtatorToolUseHandlerConfig = config

        if config.use_find_entity_id_endpoint:
            self._strategy = FindEntityIdStrategy()
        else:
            self._strategy = FindEntityByPublicationSearchStrategy()

    async def handle_request(
        self, request_dto: RequestDto, is_tool_cycle: bool = False
    ) -> RequestDto:
        if is_tool_cycle:
            return request_dto

        _logger.debug("Enabling PubTator tool use")
        if request_dto.tools is None:
            request_dto.tools = []

        request_dto.tools.append(self._strategy.tool)

        # Force this function to be used
        if self._config.force_tool_use:
            request_dto.tool_choice = "required"

        return request_dto

    async def handle_response(
        self, response_dto: ResponseDto, is_tool_cycle: bool = False
    ) -> ResponseDto:
        """
        Look for tool calls in the regular response. If they are Pubtator searches,
        we make the calls and write the answers into the context.
        """
        if is_tool_cycle:  # Our job happends before the tool cycle
            return response_dto

        _logger.debug("Looking for Pubtator tool calls")

        for choice in response_dto.choices:
            tool_calls = get(choice, "message", "tool_calls")
            if tool_calls is not None:
                await self._process_tool_calls(tool_calls)

            # This seems to be a Gemini specific error. Finish reason is "stop",
            # but there's no content and the undocumented native_finish_reason
            # indicates an LLM failure.
            if (
                isinstance(choice, NonStreamingChoice)
                and choice.native_finish_reason == "MALFORMED_FUNCTION_CALL"
            ):
                raise ValueError("Malformed function call detected. Aborting.")

        return response_dto

    async def _process_tool_calls(self, tool_calls: List[ToolCall]) -> None:
        """
        Handles tool calls for Pubtator searches. The results will be stored in the
        context.
        """

        # Initialize context dictionaries if they don't exist
        self._context.setdefault("tool_calls", {})
        self._context.setdefault("pubtator", {})

        # Process all tool calls concurrently. The Pubtator worker will
        # make sure they are rate-limited and retried on failure.
        count = 0
        async with TaskGroup() as tg:
            for tool_call in tool_calls:
                if tool_call.function.name == self._strategy.tool.function.name:
                    count += 1
                    tg.create_task(self._process_pubtator_call(tool_call))

            _logger.info(f"Queued {count} Pubtator of {len(tool_calls)} tool calls")

    async def _process_pubtator_call(self, tool_call: ToolCall) -> None:
        """
        Processes a single Pubtator tool call.
        """
        arguments = self._strategy.validate_json(tool_call)
        _logger.info(f"Searching pubtator with arguments {arguments}")

        search_results = await self._strategy.find_ids(arguments)
        tool_answer = self._strategy.format_results(
            arguments=arguments, results=search_results
        )

        self._context["pubtator"][tool_call.id] = {
            "arguments": arguments.model_dump(exclude_defaults=True, exclude_none=True),
            "search_results": search_results.model_dump(),
        }
        self._context["tool_calls"][tool_call.id] = tool_answer
