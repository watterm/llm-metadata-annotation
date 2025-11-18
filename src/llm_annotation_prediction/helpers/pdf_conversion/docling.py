from __future__ import annotations

import logging
from pathlib import Path


def convert_pdf_to_markdown(
    pdf_path: Path, markdown_path: Path, html_path: Path, logger: logging.Logger
) -> None:
    """Convert a PDF to Markdown (and HTML) using Docling."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        logger.error(
            "docling is not installed. Install with 'uv sync --extra docling'."
        )
        return

    logger.info("Converting PDF to Markdown with Docling")
    try:
        easy_ocr_options = EasyOcrOptions()

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = False
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options = easy_ocr_options

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(pdf_path)
    except Exception as error:  # noqa: BLE001 - maintain behaviour
        logger.error("Error while converting PDF with Docling: %s", error)
        return

    if result.errors:
        logger.error("Non-fatal errors while converting PDF with Docling:")
        logger.error(result.errors)

    result.document.save_as_markdown(markdown_path)
    result.document.save_as_html(html_path)
