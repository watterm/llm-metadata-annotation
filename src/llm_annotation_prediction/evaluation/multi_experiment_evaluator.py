from importlib.util import find_spec
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel, Field, PrivateAttr

from llm_annotation_prediction.evaluation.experiment_evaluator import (
    ExperimentEvaluator,
    ExperimentEvaluatorConfig,
)
from llm_annotation_prediction.evaluation.model_info import model_info
from llm_annotation_prediction.evaluation.plots import (
    plot_all_features,
    plot_pubtator_evaluation,
)

# Plots are done with pandas and seaborn, but installed optionally, so we
# load them dynamically and only import pandas here for type checking.
if TYPE_CHECKING:
    import pandas as pd

_logger = getLogger("MultiExperimentEvaluator")


class MultiExperimentEvaluatorConfig(BaseModel):
    """Configuration for the multi-experiment evaluator."""

    # Config for the individual experiment evaluators
    experiment_evaluator_config: ExperimentEvaluatorConfig = ExperimentEvaluatorConfig(
        experiment_path=""
    )

    # Root directory containing all experiment subdirectories
    experiments_root_dir: str = "experiments"


class MultiExperimentEvaluator(BaseModel):
    """Evaluates and compares statistics across multiple experiments."""

    _config: MultiExperimentEvaluatorConfig = PrivateAttr()
    _evaluated: bool = PrivateAttr(False)

    experiment_evaluators: List[ExperimentEvaluator] = Field(default_factory=list)

    def __init__(self, config: MultiExperimentEvaluatorConfig):
        super().__init__()
        self._config = config

        self._load_experiments()

    def _load_experiments(self) -> None:
        """Load all experiment directories from the root directory."""
        root_path = Path(self._config.experiments_root_dir)
        if not root_path.exists() or not root_path.is_dir():
            print(
                f"Experiments root directory does not exist: {self._config.experiments_root_dir}"
            )
            return

        for entry in root_path.iterdir():
            if entry.is_dir():
                entry_path = str(entry.resolve())
                try:
                    # Use a copy of the evaluator config with the specific experiment path
                    evaluator_config = (
                        self._config.experiment_evaluator_config.model_copy(
                            update={"experiment_path": entry_path}
                        )
                    )
                    evaluator = ExperimentEvaluator(evaluator_config)
                    self.experiment_evaluators.append(evaluator)
                except Exception as e:
                    print(f"Failed to load experiment {entry}: {e}")

    async def evaluate(self) -> None:
        """Evaluate all loaded experiments."""
        for evaluator in self.experiment_evaluators:
            await evaluator.evaluate()
        self._evaluated = True

    def has_plotting_support(self) -> bool:
        """Check if plotting dependencies are available."""

        for lib in ["matplotlib", "pandas", "seaborn"]:
            if find_spec(lib) is None:
                return False
        return True

    def _to_dataframe(self) -> "pd.DataFrame | None":
        """Convert the evaluation results into a Pandas DataFrame for plotting."""
        if not self.has_plotting_support():
            raise ImportError("Plotting support is not available.")
        import pandas as pd

        all_trials_data = []

        # Top level is the experiments for each model
        for exp_evaluator in self.experiment_evaluators:
            model_name = exp_evaluator.model_name

            if model_name is None:
                raise KeyError(
                    "Experiment evaluator for a model without a name found. Skipping."
                )

            # Then we move to the experiments over all publications
            for (
                publication,
                conv_evaluators,
            ) in exp_evaluator.publication_evaluators.items():
                # Innermost are the repeated trials for each publication
                for conv_evaluator in conv_evaluators:
                    trial_data: Dict[str, Any] = {
                        "model_name": model_name,
                        "publication_uuid": publication,
                    }

                    trial_data.update(model_info[model_name])

                    trial_data.update(
                        {
                            k: len(v)
                            for k, v in conv_evaluator.lists.model_dump().items()
                        }
                    )
                    trial_data.update(conv_evaluator.general.model_dump())

                    all_trials_data.append(trial_data)

        if not all_trials_data:
            print("No trial data found to create a DataFrame.")
            return None

        df = pd.DataFrame(all_trials_data)

        # Simplify model names by removing companies
        df["model_name"] = df["model_name"].apply(
            lambda x: x.split("/")[-1] if "/" in x else x
        )

        return df

    def _prepare_plotting(self) -> "pd.DataFrame":
        if not self.has_plotting_support():
            raise ImportError("Plotting support is not available.")

        if not self._evaluated:
            raise ValueError("Experiments have not been evaluated yet.")

        df = self._to_dataframe()
        if df is None:
            raise ValueError("No data available for plotting.")

        return df

    def generate_pubtator_plot(self, out_folder: Path) -> None:
        """Generate a plot for the PubTator evaluations of the experiments."""

        df = self._prepare_plotting()
        plot_pubtator_evaluation(df, out_folder)

    def generate_features_plot(self, out_folder: Path) -> None:
        """Generate a plot for the feature evaluations of the experiments."""

        df = self._prepare_plotting()
        plot_all_features(df, out_folder)
