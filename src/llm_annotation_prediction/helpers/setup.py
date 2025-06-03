from datetime import datetime
from logging import getLogger
from pathlib import Path

from llm_annotation_prediction.helpers.config import Config
from llm_annotation_prediction.helpers.format import sanitize_folder_name

EXPERIMENT_FOLDER = "experiments"

logger = getLogger("Setup")


def get_experiment_folder(folder_name: str) -> Path:
    """
    Generates a path for an experiment folder with a timestamp and sanitized folder name.

    Args:
        folder_name (str): The name of the folder to be sanitized and used in the path.
    Returns:
        Path: A Path object representing the experiment folder path with the current
            timestamp and sanitized folder name.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = sanitize_folder_name(folder_name)

    return Path(f"{EXPERIMENT_FOLDER}/{timestamp}-{name}")


def setup_experiment_folder(config: Config) -> Path:
    """
    Sets up the experiment folder based on the provided configuration.
    This function creates a directory for the experiment if saving is enabled
    in the configuration. The directory name is derived from the experiment
    name specified in the configuration or defaults to "untitled" if no name
    is provided.
    Args:
        config (Config): The configuration object containing experiment settings.
    Returns:
        Path: The path to the created or existing experiment folder.
    """

    logger.debug("Setting up experiment folder")
    folder = get_experiment_folder(config.name or "untitled")
    if not config.no_save:
        logger.info(f"Creating experiment folder {folder}")
        folder.mkdir(parents=True, exist_ok=True)

    return folder
