import argparse

from llm_annotation_prediction.helpers.config import load_config
from llm_annotation_prediction.helpers.logging import (
    setup_logging,
    setup_memory_logging,
)
from llm_annotation_prediction.helpers.save import save_results
from llm_annotation_prediction.helpers.setup import (
    setup_experiment_folder,
)
from llm_annotation_prediction.helpers.utils import load_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental framework for few-shot ontology prediction with LLMs"
    )
    parser.add_argument(
        "--silent", type=bool, help="Prevent console output", required=False
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the configuration file for the experiment",
    )
    return parser.parse_args()


def main() -> None:
    setup_memory_logging()  # Save messages in a buffer until config and logging is established
    args = parse_args()

    config = load_config(args.config_path)
    folder = setup_experiment_folder(config)

    setup_logging(
        folder, config.no_save, config.silent or args.silent, config.log_level
    )

    dataset = load_class(config.dataset.type)(config.dataset, config.publication)
    experiment = load_class(config.experiment.type)(
        config.experiment, config.conversation, dataset
    )

    dataset.load()
    experiment.run()

    if not config.no_save:
        save_results(folder, config, [experiment, dataset])


if __name__ == "__main__":
    main()
