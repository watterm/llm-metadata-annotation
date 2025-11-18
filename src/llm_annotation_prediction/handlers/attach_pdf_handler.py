from logging import Logger, getLogger

from llm_annotation_prediction.handlers.handler import Handler, HandlerConfig
from llm_annotation_prediction.helpers.constants import Context
from llm_annotation_prediction.helpers.open_router import (
    FileContent,
    FileContentData,
    FileParserPlugin,
    PdfEngine,
    PdfPlugin,
    RequestDto,
    ResponseDto,
    UserMessage,
)

_logger: Logger = getLogger("AttachPdfHandler")


class AttachPdfHandlerConfig(HandlerConfig):
    type: str = "AttachPdfHandler"

    # The key of the base64-encoded PDF content in the context
    context_key: str

    # The filename to pass on to the LLM for the PDF attachment
    filename: str

    # Decides how OpenRouter will process or pass on the PDF file
    pdf_engine: PdfEngine = "pdf-text"


class AttachPdfHandler(Handler):
    """
    Adds a PDF file to the last message current conversation. This message
    must be a UserMessage.
    """

    def __init__(self, config: AttachPdfHandlerConfig, context: Context):
        super().__init__(config, context)
        self._config: AttachPdfHandlerConfig = config

    async def handle_request(
        self, request_dto: RequestDto, is_tool_cycle: bool = False
    ) -> RequestDto:
        _logger.debug(f"Adding PDF attachment as {self._config.filename}")
        if not is_tool_cycle:
            self._ensure_pdf_plugin(request_dto)
            last_user_message: UserMessage = self._get_last_user_message(request_dto)
            self._add_pdf_attachment(last_user_message)

        return request_dto

    async def handle_response(
        self, response_dto: ResponseDto, is_tool_cycle: bool = False
    ) -> ResponseDto:
        raise TypeError(
            f"{self.__class__.__name__} cannot be used for response handling"
        )

    def _ensure_pdf_plugin(self, request_dto: RequestDto) -> None:
        """
        Ensures that the PDF plugin is included in the request DTO.
        """
        if request_dto.plugins is None:
            request_dto.plugins = []

        if not any(plugin.id == "file-parser" for plugin in request_dto.plugins):
            request_dto.plugins.append(
                FileParserPlugin(
                    id="file-parser", pdf=PdfPlugin(engine=self._config.pdf_engine)
                )
            )

    def _get_last_user_message(self, request_dto: RequestDto) -> UserMessage:
        """
        Ensures that the last message is a UserMessage.
        """
        if request_dto.messages is None or len(request_dto.messages) == 0:
            raise ValueError("No messages found in request DTO.")

        if request_dto.messages[-1].role != "user":
            raise ValueError("Last message must be a UserMessage.")

        return request_dto.messages[-1]

    def _add_pdf_attachment(self, user_message: UserMessage) -> None:
        """
        Adds the PDF attachment to the user message, if its content is a list.
        """
        if isinstance(user_message.content, list):
            if self._config.context_key not in self._context:
                raise KeyError(
                    f"Context key '{self._config.context_key}' not found in context."
                )

            base64_data: str = (
                f"data:application/pdf;base64,{self._context[self._config.context_key]}"
            )

            user_message.content.append(
                FileContent(
                    file=FileContentData(
                        filename=self._config.filename,
                        file_data=base64_data,
                    )
                )
            )
        else:
            raise TypeError(
                "UserMessage content must be a list of ContentPart."
                " Try using AddUserMessageHandler first to create a proper UserMessage."
            )
