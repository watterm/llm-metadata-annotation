import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, TypeVar

from pydantic import BaseModel

from llm_annotation_prediction.helpers.constants import METADATA_FILENAME
from llm_annotation_prediction.helpers.utils import load_class
from llm_annotation_prediction.publication import (
    Publication,
    PublicationConfig,
)
from llm_annotation_prediction.schema import Schema, SchemaConfig

_logger = logging.getLogger("Dataset")

DatasetType = TypeVar("DatasetType", bound="Dataset")
DatasetConfigType = TypeVar("DatasetConfigType", bound="DatasetConfig")


class DatasetConfig(BaseModel):
    type: str = "Dataset"

    # The folder from which publications will be loaded
    dataset_folder: str

    # The identifiers for the subset of publications to be loaded. If not specified,
    # all publications will be loaded.
    uuids: Optional[List[str]] = None

    # A schema is optional. Future experiments might not need one.
    metadata_schema: Optional[SchemaConfig] = None


class Dataset:
    """
    Handles either all publications in a folder or a subset thereof. Can contain a
    schema to provide more information in conversations.
    """

    def __init__(
        self, dataset_config: DatasetConfig, publication_config: PublicationConfig
    ):
        self._publication_config = publication_config
        self._dataset_config = dataset_config

        self._dataset_folder: Path = Path(dataset_config.dataset_folder)

        self.publications: Dict[str, Publication] = {}
        self._publication_class = load_class(publication_config.type)
        self.schema: Schema | None = None

        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, verify: bool = True) -> None:
        """
        Load the dataset based on the configuration.

        If the dataset configuration contains specific UUIDs, load the publications
        corresponding to those UUIDs. Otherwise, load all available publications.
        """
        if self._dataset_config.uuids:
            self._load_publications(self._dataset_config.uuids, verify=verify)
        else:
            self._load_all_publications(verify=verify)

        if self._dataset_config.metadata_schema:
            self.schema = Schema(self._dataset_config.metadata_schema)

        self._loaded = True

    def _load_publications(self, uuids: List[str], verify: bool = True) -> None:
        """
        Load publications based on a list of UUIDs.
        """
        _logger.info(
            f"Loading configured UUIDs from dataset folder {self._dataset_folder}"
        )
        for uuid in uuids:
            folder = self._dataset_folder / uuid
            if folder.is_dir():
                publication = self._publication_class(self._publication_config, folder)
                publication.load(verify=verify)
                self.publications[publication.uuid] = publication

    def _load_all_publications(self, verify: bool = True) -> None:
        """
        Loads all publications from the dataset folder.

        This method iterates through all directories in the dataset folder,
        creates a publication instance for each directory, and loads the
        publication data.

        Args:
            verify (bool): If True, the publication data will be verified
                           during loading.
        """
        _logger.info(
            f"Loading all publications from dataset folder {self._dataset_folder}"
        )
        for folder in self._dataset_folder.iterdir():
            if folder.is_dir():
                publication = self._publication_class(self._publication_config, folder)
                publication.load(verify=verify)
                self.publications[publication.uuid] = publication

    def convert(self, force: bool = False) -> None:
        """
        Convert the PDF of each publication to a markdown file
        """
        if not self._loaded:
            _logger.error("Dataset not loaded")
            return

        for publication in self.publications.values():
            publication.convert_pdf_to_markdown(force=force)

    @classmethod
    def create_from_metadata_folder(
        cls, metadata_folder: Path, dataset_folder: Path
    ) -> None:
        """
        Create a dataset from a folder containing metadata JSON files.

        This method scans the specified metadata folder for JSON files and copies them
        into a corresponding folder structure within the dataset folder. If no JSON files
        are found in the root of the metadata folder, it checks in the '.fredato/metadata/entries'
        subfolder. Each JSON file is copied to a subfolder named after the file's UUID.

        Args:
            metadata_folder (Path): The path to the folder containing metadata JSON files.
            dataset_folder (Path): The path to the folder where the dataset will be created.

        Returns:
            None
        """

        _logger.info(
            f"Creating dataset from metadata folder '{metadata_folder}' in '{dataset_folder}'"
        )
        dataset_folder.mkdir(parents=True, exist_ok=True)

        json_files = list(metadata_folder.glob("*.json"))
        if not json_files:
            _logger.info("No JSON files found. Checking in '.fredato/metadata/entries'")
            subfolder = metadata_folder / ".fredato/metadata/entries"
            json_files = list(subfolder.glob("*.json"))

        if not json_files:
            _logger.error(f"Could not find JSON metadata files in {metadata_folder}")
            return

        for file in json_files:
            if file.suffix == ".json":
                uuid = file.stem
                uuid_folder = dataset_folder / uuid

                if uuid_folder.exists():
                    _logger.info(f"{uuid}: Target folder already exists")
                else:
                    _logger.info(f"{uuid}: Creating target folder")
                    uuid_folder.mkdir(exist_ok=True)

                metadata_target_path = uuid_folder / METADATA_FILENAME
                if metadata_target_path.exists():
                    _logger.info(
                        f"{uuid}: Metadata file already exists (not overwriting)"
                    )
                else:
                    _logger.info(f"{uuid}: Copying metadata file")
                    shutil.copy2(file, metadata_target_path)

    def verify(self, warn_on_missing_supplementary: bool = False) -> bool:
        """
        Verifies all publications in this dataset.

        Returns:
            bool: True if all publications are valid
        """
        _logger.info(f"Verifying dataset in {self._dataset_folder}")
        if not self._dataset_folder.exists():
            _logger.error(f'Dataset folder "{self._dataset_folder}" does not exist')
            return False

        valid = True
        for publication in self.publications.values():
            valid = valid and publication.verify(
                warn_on_missing_supplementary=warn_on_missing_supplementary
            )

        return valid
