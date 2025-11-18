from abc import ABC, abstractmethod
import json
from logging import getLogger
from pathlib import Path
from typing import Any

import yaml

logger = getLogger("Results")


class Saveable(ABC):
    """
    Interface to identify classes that save results to the experiment folder.
    """

    @abstractmethod
    def save(self, folder: Path) -> None:
        pass


class DictSerializable(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass


def save_results(folder: Path, config_path: str, classes: list[Any]) -> None:
    """
    Saves all results from classes implementing the corresponding method
    """
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
        yaml.safe_dump(
            config,
            open(folder / "config.yaml", "w"),
            default_flow_style=False,
        )

    for c in classes:
        if not isinstance(c, Saveable):
            continue
        class_name = c.__class__.__name__
        logger.info(f"Saving results for {class_name}")
        c.save(folder)


def dump_to_json(path: Path, content: dict[str, Any]) -> None:
    """
    Helper to shorten and consolidate saving
    """
    logger.debug(f"Saving dict to {path}")
    with open(path, "w") as f:
        json.dump(content, f, indent=2)
