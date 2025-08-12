import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from llm_annotation_prediction.helpers.constants import (
    METADATA_FILENAME,
    PAPER_MD_FILENAME,
    PAPER_PDF_FILENAME,
)
from llm_annotation_prediction.helpers.format import format_doi
from llm_annotation_prediction.helpers.utils import get

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

document_converter: "DocumentConverter | None"
try:
    from docling.document_converter import DocumentConverter

    document_converter = DocumentConverter()
except ImportError:
    document_converter = None


class PublicationConfig(BaseModel):
    type: str = "Publication"


class Publication:
    """
    A publication instance represents a paper and possibly its supplementary material.
    """

    def __init__(self, config: PublicationConfig, publication_folder: Path):
        self._config: PublicationConfig = config
        self.publication_folder: Path = publication_folder
        self.uuid: str = self.publication_folder.name

        self._metadata_path: Path = self.publication_folder / METADATA_FILENAME
        self._paper_pdf_path: Path = self.publication_folder / PAPER_PDF_FILENAME  # noqa: F821
        self._paper_md_path: Path = self.publication_folder / PAPER_MD_FILENAME

        self._loaded: bool = False
        self._logger: logging.Logger = logging.getLogger(f"Pub:{self.uuid}")
        self._logger.debug("Created")

    def __repr__(self) -> str:
        return f"Publication(uuid={self.uuid}, folder={self.publication_folder})"

    def load(self, verify: bool = True) -> None:
        self._logger.debug("Loading")
        loading_error = None
        try:
            self.metadata = json.load(open(self._metadata_path, "r", encoding="utf8"))
            self._paper_md = self._paper_md_path.read_text(encoding="utf-8")
        except Exception as error:
            loading_error = error

        if verify:
            if loading_error:
                self._logger.error(f"Error while loading: {loading_error}")
                raise loading_error
            if not self.verify():
                raise ValueError("Publication is not valid")

        self._loaded = True

    def verify(self, warn_on_missing_supplementary: bool = False) -> bool:
        """
        Verifies the existence of the publication files.

        Returns:
            bool: True if this publication is ready to be used in experiments.
        """
        self._logger.debug("Verifying publication")
        valid = True

        # Metadata is only loaded for verification; does not initialize publication
        metadata = None
        if not self._metadata_path.exists():
            self._logger.error("Metadata file does not exist")
            valid = False
        else:
            metadata = json.load(open(self._metadata_path, "r", encoding="utf8"))

        if not self._paper_md_path.exists():
            valid = False
            self._logger.error("Markdown file does not exist")

        # Check if PDF exits. If not, try to provide DOI link.
        if not self._paper_pdf_path.exists():
            self._logger.error(
                "Paper PDF does not exist. Please download and save it as 'paper.pdf'."
            )
            valid = False
            if metadata:
                title = get(metadata, "publication", "publicationTitle")
                doi = get(metadata, "publication", "doi") or ""
                self._logger.info(f"  Title: {title}")
                if doi:
                    self._logger.info(f"  DOI: {format_doi(doi)}")

        return valid

    @property
    def publication_text(self) -> str:
        return self._paper_md

    def convert_pdf_to_markdown(self, force: bool = False) -> None:
        """
        Uses docling to convert the PDF to a Markdown file
        """
        if self._paper_md_path.exists() and not force:
            self._logger.info(
                "PDF Conversion: Markdown file already exists. Ignoring publication."
            )
            return

        if not document_converter:
            self._logger.error(
                "docling is not installed. Install with 'uv sync --extra docling'."
            )
            return

        self._logger.info("Converting PDF to Markdown")
        try:
            result = document_converter.convert(self._paper_pdf_path)
        except Exception as e:
            self._logger.error(f"Error while converting PDF: {e}")
            return

        if len(result.errors) > 0:
            self._logger.error("Non-fatal errors while converting PDF:")
            self._logger.error(result.errors)

        result.document.save_as_markdown(self._paper_md_path)
        result.document.save_as_html(self._paper_md_path.with_suffix(".html"))
