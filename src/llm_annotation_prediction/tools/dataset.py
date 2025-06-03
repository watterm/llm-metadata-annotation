import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from llm_annotation_prediction.dataset import Dataset, DatasetConfig
from llm_annotation_prediction.helpers.logging import LOG_FORMAT
from llm_annotation_prediction.publication import PublicationConfig


def parse_create_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "metadata_folder",
        type=str,
        help="Path to the fredato repository folder",
    )
    parser.add_argument(
        "target_folder",
        type=str,
        help="Path to the folder where the dataset will be created",
    )
    parser.add_argument(
        "--publication_type",
        "-p",
        type=str,
        default="publication.Publication",
        help="Publication class to use",
    )


def parse_convert_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "dataset_folder",
        type=str,
        help="Path to the dataset folder containing publication folders",
    )
    parser.add_argument(
        "--publication_type",
        "-p",
        type=str,
        default="publication.Publication",
        help="Publication class to use",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing markdown files"
    )


def parse_args() -> Namespace:
    """
    Collects the subparsers.
    """
    parser = ArgumentParser(
        description="Dataset management tool for creating and converting datasets"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command to execute"
    )

    create_parser = subparsers.add_parser(
        "create", help="Create a dataset from a fredato repository folder with metadata"
    )
    parse_create_args(create_parser)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert PDF files to Markdown for all publications in a dataset",
    )
    parse_convert_args(convert_parser)

    return parser.parse_args()


def create_dataset(args: Namespace) -> None:
    """
    Creates a dataset from a fredato metadata folder. Verifies dataset after
    creation to show DOI links for the publications.
    """
    Dataset.create_from_metadata_folder(
        Path(args.metadata_folder), Path(args.target_folder)
    )
    publication_config = PublicationConfig(type=args.publication_type)
    dataset: Dataset = Dataset(
        DatasetConfig(dataset_folder=args.target_folder), publication_config
    )

    # Load without verification, so we can enable the warnings on missing supp. PDF
    dataset.load(verify=False)
    dataset.verify(warn_on_missing_supplementary=True)


def convert_dataset(args: Namespace) -> None:
    """Converts PDFs to Markdown in a dataset."""
    publication_config = PublicationConfig(type=args.publication_type)
    dataset: Dataset = Dataset(
        DatasetConfig(dataset_folder=args.dataset_folder), publication_config
    )

    # Load without verification since we only need the PDF files
    dataset.load(verify=False)
    dataset.convert(force=args.force)


def main() -> None:
    """Main entry point for the dataset management tool."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args()

    if args.command == "create":
        create_dataset(args)
    elif args.command == "convert":
        convert_dataset(args)
