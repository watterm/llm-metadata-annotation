from __future__ import annotations

import base64
from binascii import Error as BinasciiError
from dataclasses import dataclass
import logging
from pathlib import Path

import httpx
from pydantic import AnyUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_annotation_prediction.helpers.open_router import (
    FileAnnotation,
    FileContent,
    FileContentData,
    FileParserPlugin,
    ImageContent,
    NonStreamingChoice,
    PdfPlugin,
    RequestDto,
    ResponseDto,
    ResponseError,
    ResponseMessage,
    TextContent,
    UserMessage,
)


@dataclass(frozen=True)
class ExtractionResult:
    markdown: str

    # Images in base64 format as they occur in the annotations
    image_attachments: list[str]


class _MistralOcrConfig(BaseSettings):
    """Settings for Mistral OCR via OpenRouter."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    api_key: str | None = Field(alias="open_router_api_key", default=None)
    api_url: AnyUrl = Field(
        default_factory=lambda: AnyUrl("https://openrouter.ai/api/v1/")
    )
    # We need a model that can actually handle the converted PDF, i.e. with sufficient
    # context length
    model: str = Field(
        default="openai/gpt-oss-20b:free",
        alias="pdf_conversion_text_model",
    )


def convert_pdf_to_markdown(
    *,
    pdf_filename: str,
    pdf_base64: str | None,
    markdown_path: Path,
    logger: logging.Logger,
    store_images: bool = False,
) -> None:
    """Convert a PDF to Markdown using Mistral OCR through OpenRouter."""
    ocr_config = _MistralOcrConfig()
    if not ocr_config.api_key:
        logger.error("OPEN_ROUTER_API_KEY not set. Skipping Mistral OCR.")
        return

    if not pdf_base64:
        logger.error("Missing base64 encoded PDF. Skipping Mistral OCR.")
        return

    request: RequestDto = _build_request(ocr_config.model, pdf_filename, pdf_base64)
    response_dto: ResponseDto | None = _send_request(ocr_config, request, logger)
    if not response_dto:
        return

    extraction: ExtractionResult = _extract_markdown_and_images(response_dto)
    markdown: str = extraction.markdown
    if not markdown:
        logger.error("Mistral OCR returned no markdown annotations.")
        return

    if store_images and extraction.image_attachments:
        replaced_markdown: str | None = _store_images(
            markdown_path=markdown_path,
            markdown=markdown,
            image_attachments=extraction.image_attachments,
            logger=logger,
        )
        if replaced_markdown:
            markdown = replaced_markdown

    markdown_path.write_text(markdown, encoding="utf-8")
    logger.info("Saved Markdown converted with Mistral OCR")


def _build_request(model: str, pdf_filename: str, pdf_base64: str) -> RequestDto:
    pdf_data = f"data:application/pdf;base64,{pdf_base64}"
    return RequestDto(
        model=model,
        messages=[
            UserMessage(
                role="user",
                content=[
                    FileContent(
                        file=FileContentData(
                            filename=pdf_filename,
                            file_data=pdf_data,
                        )
                    )
                ],
            )
        ],
        plugins=[FileParserPlugin(pdf=PdfPlugin(engine="mistral-ocr"))],
    )


def _send_request(
    config: _MistralOcrConfig, request: RequestDto, logger: logging.Logger
) -> ResponseDto | None:
    try:
        response: httpx.Response = httpx.post(
            httpx.URL(str(config.api_url)).join("chat/completions"),
            headers={"Authorization": f"Bearer {config.api_key}"},
            json=request.model_dump(exclude_none=True),
            timeout=300,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        logger.error("Failed to convert PDF with Mistral: %s", error)
        return None

    try:
        return ResponseDto.model_validate(response.json())
    except Exception as error:
        _log_validation_error(response, error, logger)
        return None


def _log_validation_error(
    response: httpx.Response, error: Exception, logger: logging.Logger
) -> None:
    try:
        error_response: ResponseError = ResponseError.model_validate(response.json())
        logger.error("OpenRouter returned an error: %s", error_response)
    except Exception:
        logger.error("Invalid OpenRouter error response: %s", error)
        logger.debug("Full response: %s", response.content)


def _extract_markdown_and_images(response_dto: ResponseDto) -> ExtractionResult:
    markdown_parts: list[str] = []
    image_attachments: list[str] = []

    # Only check for annotations in the first message
    if len(response_dto.choices) == 0:
        return ExtractionResult(markdown="", image_attachments=[])

    choice = response_dto.choices[0]
    if not isinstance(choice, NonStreamingChoice):
        return ExtractionResult(markdown="", image_attachments=[])

    # Put text annotations together and store image separately
    _collect_message_annotations(choice.message, markdown_parts, image_attachments)

    # Converted files start with <file...>. Remove it.
    _strip_file_wrappers(markdown_parts)

    markdown: str = "".join(markdown_parts)
    return ExtractionResult(markdown=markdown, image_attachments=image_attachments)


def _collect_message_annotations(
    message: ResponseMessage,
    markdown_parts: list[str],
    image_attachments: list[str],
) -> None:
    for annotation in message.annotations or []:
        file_annotation: FileAnnotation | None = annotation.file
        if file_annotation:
            _collect_file_annotation_parts(
                file_annotation, markdown_parts, image_attachments
            )


def _collect_file_annotation_parts(
    file_annotation: FileAnnotation,
    markdown_parts: list[str],
    image_attachments: list[str],
) -> None:
    for part in file_annotation.content or []:
        # Text content is directly appended to the markdown file
        if isinstance(part, TextContent):
            markdown_parts.append(part.text)
            continue

        # Images are collected as they appear
        if isinstance(part, ImageContent):
            image_url: str | None = getattr(
                getattr(part, "image_url", None), "url", None
            )
            if image_url:
                image_attachments.append(image_url)
            continue


def _strip_file_wrappers(markdown_parts: list[str]) -> None:
    if not markdown_parts:
        return

    literal_start: str = markdown_parts[0].strip()
    if literal_start.lower().startswith("<file") and literal_start.endswith(">"):
        markdown_parts.pop(0)

    if not markdown_parts:
        return

    literal_end: str = markdown_parts[-1].strip()
    if literal_end == "</file>":
        markdown_parts.pop()


def _store_images(
    *,
    markdown_path: Path,
    markdown: str,
    image_attachments: list[str],
    logger: logging.Logger,
) -> str | None:
    updated_markdown: str = markdown
    images_dir: Path = markdown_path.parent / "images"
    try:
        images_dir.mkdir(exist_ok=True)
    except OSError as error:
        logger.error("Failed to create images directory '%s': %s", images_dir, error)
        return updated_markdown

    saved_files: int = 0
    for i, image_attachment in enumerate(image_attachments):
        extension: str | None = _determine_extension(image_attachment)
        target_path = images_dir / f"img-{i}{extension or ''}"

        try:
            file_bytes: bytes = _decode_payload(image_attachment)
        except ValueError as error:
            logger.error("Failed to decode attachment '%s': %s", i, error)
            continue

        try:
            target_path.write_bytes(file_bytes)
        except OSError as error:
            logger.error("Failed to write image '%s': %s", target_path, error)
            continue

        saved_files += 1

        # Update markdown to point to saved image
        old = f"img-{i}{extension or ''}"
        old_markdown_image: str = f"![{old}]({old})"
        new_markdown_image: str = f"![{old}](images/{old})"
        updated_markdown = updated_markdown.replace(
            old_markdown_image, new_markdown_image
        )

    if saved_files:
        logger.info("Saved %s file(s) to %s", saved_files, images_dir)
    return updated_markdown


def _determine_extension(payload: str) -> str | None:
    extension: str | None = None
    if payload.startswith("data:"):
        header: str = payload.split(",", 1)[0]
        mime_part: str = header[5:]
        mime_type: str = mime_part.split(";", 1)[0]
        if "/" in mime_type:
            subtype: str = mime_type.split("/", 1)[1]
            sanitized_subtype: str = subtype.replace("+", "-")
            if sanitized_subtype:
                extension = f".{sanitized_subtype}"

    return extension


def _decode_payload(payload: str) -> bytes:
    if payload.startswith("data:"):
        _, _, encoded = payload.partition(",")
        if not encoded:
            raise ValueError("Invalid data URI payload")
        payload = encoded

    try:
        return base64.b64decode(payload)
    except (BinasciiError, ValueError) as error:
        raise ValueError("Invalid base64 payload") from error
