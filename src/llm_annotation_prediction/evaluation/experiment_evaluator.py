import os
import sys
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, PrivateAttr

from llm_annotation_prediction.evaluation.conversation_evaluator import (
    EntityLists,
    GeneralStatistics,
)
from llm_annotation_prediction.helpers.constants import (
    CONTEXT_FILENAME,
    Data,
    DataAdapter,
)
from llm_annotation_prediction.helpers.utils import get

from .conversation_evaluator import ConversationEvaluator, ConversationEvaluatorConfig

_logger = getLogger("ExperimentEvaluator")

# Each publication can have multiple trials, each with its own evaluator.
PublicationEvaluator = Dict[str, List[ConversationEvaluator]]

# The publication metrics are aggregrated for each trial
PublicationMetrics = Dict[str, "EntityListMetrics"]


class EntityListMetrics(BaseModel):
    """
    Holds statistical metrics for a entity list in conversations of an experiment.
    """

    min: float = Field(default=sys.maxsize)
    max: float = Field(default=-sys.maxsize - 1)
    mean: float = 0.0
    variance: float = 0.0
    total_sum: float = 0

    # data points (e.g., conversations) used for these stats
    observations: List[float] = Field(default_factory=list)


class EntityStats(BaseModel):
    """
    Holds aggregated statistics for entity lists across all conversations in an experiment.
    """

    # Keys are publication UUIDs
    intra_publication: Dict[str, PublicationMetrics] = Field(default_factory=dict)

    # Keys are entity list names
    inter_publication: Dict[str, EntityListMetrics] = Field(default_factory=dict)


class ExperimentEvaluatorConfig(BaseModel):
    """
    Configuration for the experiment evaluator.
    """

    # Config for the conversation evaluators
    conversation_config: ConversationEvaluatorConfig = ConversationEvaluatorConfig()

    # Path to the experiment directory containing the context file.
    experiment_path: str = Field(...)

    # Name of the model used in the experiment. Will otherwise be read from config.yaml.
    model_name: Optional[str] = None

    # List of publication UUIDs to evaluate. If None, all publications in the context
    # file will be evaluated."
    publications: Optional[List[str]] = None


class ExperimentEvaluator(BaseModel):
    """Evaluates statistics for a single experiment (one model, multiple publications)."""

    _config: ExperimentEvaluatorConfig = PrivateAttr()
    _data: Data = PrivateAttr()
    _evaluated: bool = PrivateAttr(False)

    model_name: Optional[str] = None
    stats: EntityStats = Field(default_factory=EntityStats)
    publication_evaluators: PublicationEvaluator = Field(default_factory=dict)

    def __init__(self, config: ExperimentEvaluatorConfig):
        super().__init__()
        self._config = config

        # Read model name from config.yaml if not provided
        self.model_name = config.model_name or self._get_model_name_from_config()

        self._data = self._load_data()

    def _get_model_name_from_config(self) -> str:
        """Read the model name from config.yaml in the experiment folder."""
        config_path = os.path.join(self._config.experiment_path, "config.yaml")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    # Use the get function from utils to safely navigate the nested structure
                    model_name: str = get(config, "conversation", "model")
                    if model_name:
                        _logger.debug(f"Found model name in config.yaml: {model_name}")
                        return model_name
            except Exception as e:
                _logger.error(f"Error reading config.yaml: {e}")

        raise ValueError(
            f"Model name not found in config.yaml at {config_path}. Please provide a model name."
        )

    def _load_data(self) -> Data:
        """
        Loads the context data from the respective JSON in the experiment.
        """
        print(f"Loading experiment in {self._config.experiment_path}")
        try:
            context_path = Path(self._config.experiment_path) / CONTEXT_FILENAME
            with open(context_path, encoding="utf-8") as f:
                context = DataAdapter.validate_json(f.read())

            self.publication_evaluators: PublicationEvaluator = {}
            for uuid, trials in context.items():
                if self._config.publications and uuid not in self._config.publications:
                    continue

                self.publication_evaluators[uuid] = []
                for trial in trials:
                    evaluator = ConversationEvaluator(
                        config=self._config.conversation_config, context=trial
                    )
                    self.publication_evaluators[uuid].append(evaluator)
        except FileNotFoundError:
            print(
                (
                    f"Error: Context file not found in '{self._config.experiment_path}'. "
                    "Is the experiment still running?"
                )
            )
            exit(1)
        return context

    def _evaluate_intra_publication(self) -> None:
        """
        Compute inter-publication statistics for all entity lists.
        """

        for uuid, pub_evaluator in self.publication_evaluators.items():
            fields = list(EntityLists.model_fields.keys()) + list(
                GeneralStatistics.model_fields.keys()
            )

            # Create metrics for each entity list
            for field in fields:
                metrics = EntityListMetrics()

                # Iterate through all trials to get counts
                for conv_evaluator in pub_evaluator:
                    var = None
                    try:
                        var = len(get(conv_evaluator.lists, field))
                    except Exception:
                        pass

                    if var is None:
                        var = get(conv_evaluator.general, field)

                    if var is None:
                        _logger.warning(
                            f"Field '{field}' not found in conversation evaluator for "
                            f"publication {uuid} and model {self.model_name}."
                        )
                        continue

                    metrics.min = min(metrics.min, var)
                    metrics.max = max(metrics.max, var)
                    metrics.total_sum += var
                    metrics.observations.append(var)

                # Calculate and store the metrics for this entity list
                n_observations = len(metrics.observations)
                if n_observations > 0:
                    metrics.mean = metrics.total_sum / n_observations

                    variance_sum = sum(
                        (length - metrics.mean) ** 2 for length in metrics.observations
                    )
                    metrics.variance = variance_sum / n_observations

                    self.stats.intra_publication.setdefault(uuid, {})[field] = metrics

    def _evaluate_inter_publication(self) -> None:
        """
        Compute inter-publication statistics for all entity lists.
        """
        for field_name in EntityLists.model_fields.keys():
            metrics = EntityListMetrics()

            for uuid, pub_metrics in self.stats.intra_publication.items():
                if field_name not in pub_metrics:
                    _logger.warning(
                        f"Entity list '{field_name}' not found in publication {uuid}."
                    )
                    continue

                entity_list_metrics = pub_metrics[field_name]
                metrics.min = min(metrics.min, entity_list_metrics.min)
                metrics.max = max(metrics.max, entity_list_metrics.max)

                # Only look at the mean values for inter-publication stats
                # More elaborate statistics should be done outside this program.
                metrics.total_sum += entity_list_metrics.mean
                metrics.observations.append(entity_list_metrics.mean)

            n_observations = len(metrics.observations)
            if n_observations > 0:
                metrics.mean = metrics.total_sum / n_observations

                variance_sum = sum(
                    (length - metrics.mean) ** 2 for length in metrics.observations
                )
                metrics.variance = variance_sum / n_observations

                self.stats.inter_publication[field_name] = metrics

    async def evaluate(self) -> None:
        """Compute aggregate statistics for this experiment."""
        if not self.publication_evaluators:
            raise ValueError(
                f"No conversations found in {self._config.experiment_path}"
            )

        # Evaluate all conversations
        for pub_evaluator in self.publication_evaluators.values():
            for conv_evaluator in pub_evaluator:
                await conv_evaluator.evaluate()

        # Compute intra-publication statistics
        self._evaluate_intra_publication()

        # Compute inter-publication statistics
        self._evaluate_inter_publication()

        self._evaluated = True
