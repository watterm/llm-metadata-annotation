import base64
import json
import logging
from pathlib import Path
import re

from pydantic import BaseModel

from llm_annotation_prediction.helpers.constants import (
    METADATA_FILENAME,
    PAPER_MD_FILENAME,
    PAPER_PDF_FILENAME,
    Context,
)
from llm_annotation_prediction.helpers.format import format_doi
from llm_annotation_prediction.helpers.pdf_conversion.docling import (
    convert_pdf_to_markdown as convert_pdf_with_docling,
)
from llm_annotation_prediction.helpers.pdf_conversion.mistral import (
    convert_pdf_to_markdown as convert_pdf_with_mistral,
)
from llm_annotation_prediction.helpers.utils import get


class PublicationConfig(BaseModel):
    type: str = "Publication"

    # Whether to load the base64-encoded PDF into memory
    load_base64_pdf: bool = False

    # Whether the markdown files need to exist when verifying the publication
    require_markdown: bool = True


class Publication:
    """
    A publication instance represents a paper and possibly its supplementary material.
    """

    def __init__(self, config: PublicationConfig, publication_folder: Path):
        self._config: PublicationConfig = config
        self.publication_folder: Path = publication_folder
        self.uuid: str = self.publication_folder.name

        self._metadata_path: Path = self.publication_folder / METADATA_FILENAME
        self._paper_pdf_path: Path = self.publication_folder / PAPER_PDF_FILENAME
        self._paper_md_path: Path = self.publication_folder / PAPER_MD_FILENAME
        self._paper_md: str | None = None

        self._loaded: bool = False
        self._logger: logging.Logger = logging.getLogger(f"Pub:{self.uuid}")
        self._logger.debug("Created")

    def __repr__(self) -> str:
        return f"Publication(uuid={self.uuid}, folder={self.publication_folder})"

    def load(self, verify: bool = True) -> None:
        self._logger.debug("Loading")
        loading_error = None
        try:
            self.metadata = json.load(open(self._metadata_path, encoding="utf8"))

            if self._config.require_markdown or self._paper_md_path.exists():
                self._paper_md = self._paper_md_path.read_text(encoding="utf-8")
            else:
                self._paper_md = None
        except Exception as error:
            loading_error = error

        if verify:
            if loading_error:
                self._logger.error(f"Error while loading: {loading_error}")
                raise loading_error
            if not self.verify():
                raise ValueError("Publication is not valid")

        self._loaded = True

    def verify(self) -> bool:
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
            metadata = json.load(open(self._metadata_path, encoding="utf8"))

        if self._config.require_markdown and not self._paper_md_path.exists():
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
    def publication_text(self, normalize_whitespace: bool = True) -> str | None:
        text: str = self._paper_md or ""
        if normalize_whitespace:
            text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 consecutive newlines
        return text

    @property
    def base64_encoded_pdf(self) -> str | None:
        """
        Loads and returns the base64-encoded PDF content. On-the-fly for now, because
        it takes little time for publication PDFs.
        """
        try:
            with open(self._paper_pdf_path, "rb") as pdf_file:
                return base64.b64encode(pdf_file.read()).decode("utf-8")
        except (FileNotFoundError, PermissionError, OSError) as e:
            self._logger.error(f"Failed to load PDF file for base64 encoding: {e}")
            return None

    def get_context(self) -> Context:
        """
        Returns a context dict with publication information for prompting.
        """

        context: Context = {}

        if self._paper_md:
            context["publication"] = self._paper_md

        if self._config.load_base64_pdf:
            context["publication_base64_pdf"] = self.base64_encoded_pdf

        return context

    def convert_pdf_to_markdown(
        self, use_mistral: bool = False, force: bool = False
    ) -> None:
        if self._paper_md_path.exists() and not force:
            self._logger.info(
                "PDF Conversion: Markdown file already exists. Ignoring publication."
            )
            return

        if use_mistral:
            self._logger.info("Converting PDF to Markdown with Mistral")
            pdf_base64: str | None = self.base64_encoded_pdf
            convert_pdf_with_mistral(
                pdf_filename=f"{self.uuid}.pdf",
                pdf_base64=pdf_base64,
                markdown_path=self._paper_md_path,
                logger=self._logger,
                store_images=True,
            )
        else:
            self._logger.info("Converting PDF to Markdown with Docling")
            convert_pdf_with_docling(
                pdf_path=self._paper_pdf_path,
                markdown_path=self._paper_md_path,
                html_path=self._paper_md_path.with_suffix(".html"),
                logger=self._logger,
            )
