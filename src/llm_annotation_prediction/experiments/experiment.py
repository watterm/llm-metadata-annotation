from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

from llm_annotation_prediction.conversation import (
    OpenRouterConversationConfig,
)
from llm_annotation_prediction.dataset import Dataset
from llm_annotation_prediction.helpers.save import Saveable

ExperimentConfigType = TypeVar("ExperimentConfigType", bound="ExperimentConfig")


class ExperimentConfig(BaseModel):
    type: str = "Experiment"


class Experiment(Generic[ExperimentConfigType], Saveable):
    """
    Base class for all experiments.
    """

    def __init__(
        self,
        config: ExperimentConfigType,
        conversation_config: OpenRouterConversationConfig,
        dataset: Dataset,
    ):
        self._config: ExperimentConfigType = config
        self._conversation_config: OpenRouterConversationConfig = conversation_config
        self._dataset: Dataset = dataset

    def run(self) -> None:
        raise NotImplementedError(
            f"run method not implemented in {self.__class__.__name__}"
        )

    def save(self, folder: Path) -> None:
        raise NotImplementedError(
            f"save method not implemented in {self.__class__.__name__}"
        )
