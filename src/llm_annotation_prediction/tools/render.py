"""
Helper tool to render all conversations and evaluations of all experiments in a folder
as HTML and PDFs. Chrome is used to convert HTML to PDF, if available, so no additional
dependencies are required.

Warning: Some HTML pages could not be converted by Chrome. It loaded Tensorflow and
then timed out.
"""

import argparse
import asyncio
import io
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from llm_annotation_prediction.evaluation.conversation_evaluator import (
    ConversationEvaluatorConfig,
)
from llm_annotation_prediction.evaluation.experiment_evaluator import (
    ExperimentEvaluatorConfig,
)
from llm_annotation_prediction.evaluation.multi_experiment_evaluator import (
    MultiExperimentEvaluator,
    MultiExperimentEvaluatorConfig,
)
from llm_annotation_prediction.helpers.constants import Context, Conversation
from llm_annotation_prediction.tools.show import (
    console,
    evaluate_context,
    load_conversations,
    load_data,
    print_message,
)

chrome = (
    shutil.which("google-chrome") or shutil.which("chrome") or shutil.which("chromium")
)


def html_to_pdf(input_html: str, output_pdf: str) -> None:
    """
    Convert an HTML file to a PDF using Google Chrome in headless mode.
    """
    if not chrome:
        raise FileNotFoundError("Could not find Chrome/Chromium on your PATH")

    args = [
        chrome,
        "--headless",
        "--disable-gpu",  # (obsolete on newer Chrome but harmless)
        f"--print-to-pdf={output_pdf}",
        Path(input_html).absolute().as_uri(),
    ]

    try:
        subprocess.run(args, check=True, timeout=60)
    except subprocess.TimeoutExpired:
        print(f"Timeout converting {input_html} to PDF.", file=sys.stderr)


def get_experiment_folders(containing_folder: Path) -> List[Path]:
    """
    Get all experiment folders in the given containing folder.
    """
    return [f for f in containing_folder.iterdir() if f.is_dir()]


async def _render_single_evaluation(
    context: Context,
    title: str,
    out_folder: Path,
    detailed: bool,
    overwrite: bool = False,
) -> None:
    """
    Renders a single evaluation to HTML and PDF.
    """
    suffix = "detailed" if detailed else "short"
    html_file = out_folder / f"eval_{suffix}.html"
    pdf_file = out_folder / f"eval_{suffix}.pdf"

    if overwrite or not (html_file.exists() and pdf_file.exists()):
        await evaluate_context(
            context=context,
            title=title,
            verify_pubtator_ids=True,
            disable_description=not detailed,
            disable_elements=not detailed,
        )
        console.save_html(str(html_file))
    else:
        print(f"Skipping {suffix} evaluation for '{title}' (HTML already exists).")

    if overwrite or not pdf_file.exists():
        if chrome:
            try:
                html_to_pdf(str(html_file), str(pdf_file))
                print(f"Generated PDF: {pdf_file}")
                html_file.unlink()
            except Exception as e:
                print(f"Error generating PDF: {e}")
    else:
        print(
            f"Skipping PDF conversion for {suffix} evaluation '{title}' (PDF already exists)."
        )


async def render_evaluation(
    context: Context, title: str, out_folder: Path, overwrite: bool = False
) -> None:
    """Renders both short and detailed evaluations of a single trial"""
    print(f"Rendering evaluation for '{title}'")
    await _render_single_evaluation(
        context=context,
        title=title,
        out_folder=out_folder,
        detailed=False,
        overwrite=overwrite,
    )
    await _render_single_evaluation(
        context=context,
        title=title,
        out_folder=out_folder,
        detailed=True,
        overwrite=overwrite,
    )


def render_conversation(
    conversation: Conversation, title: str, out_folder: Path, overwrite: bool = False
) -> None:
    """Renders a conversation to HTML and PDF"""
    html_file = out_folder / "conversation.html"
    pdf_file = out_folder / "conversation.pdf"

    if overwrite or not (html_file.exists() and pdf_file.exists()):
        for message in conversation:
            print_message(message)
        console.save_html(str(html_file))
    else:
        print(f"Skipping conversation for '{title}' (HTML already exists).")

    if overwrite or not pdf_file.exists():
        if chrome:
            html_to_pdf(str(html_file), str(pdf_file))
            try:
                html_file.unlink()
            except Exception as e:
                print(f"Warning: Could not remove {html_file}: {e}")
    else:
        print(
            f"Skipping PDF conversion for conversation '{title}' (PDF already exists)."
        )


async def render_experiment(
    experiment: Path, out_folder: Path, overwrite: bool = False
) -> None:
    """Renders all trials in an experiment"""
    print(f"Rendering experiment '{experiment}'")
    data = load_data(str(experiment))
    conversations = load_conversations(str(experiment))

    # Conversations and Data should have the same keys
    for uuid, trials in data.items():
        print(f"Rendering {uuid}: {len(trials)} trials")

        trial_folder = out_folder / uuid
        for trial_index, context in enumerate(trials):
            if len(trials) > 1:
                trial_folder = trial_folder / f"trial_{str(trial_index)}"
            trial_folder.mkdir(exist_ok=True)
            title = f"'{uuid}' (Trial {trial_index})"

            # We have to run all rendering sequentially, because we use a single console
            # from "show.py". Could be improved by using a different console for each trial.
            await render_evaluation(
                context=context,
                title=title,
                out_folder=trial_folder,
                overwrite=overwrite,
            )

            trial_conversation = conversations[uuid][trial_index]
            render_conversation(
                conversation=trial_conversation,
                title=title,
                out_folder=trial_folder,
                overwrite=overwrite,
            )


async def render_all_experiments(
    in_folder: Path, out_folder: Path, overwrite: bool = False
) -> None:
    """Renders all experiments in a folder"""
    print(f"Rendering all experiments in '{in_folder}' to '{out_folder}'")
    experiments = get_experiment_folders(in_folder)

    for experiment in experiments:
        out_experiment = out_folder / experiment.name
        out_experiment.mkdir(exist_ok=True)

        await render_experiment(experiment, out_experiment, overwrite=overwrite)


async def plot(in_folder: Path, out_folder: Path) -> None:
    config = MultiExperimentEvaluatorConfig(
        experiments_root_dir=str(in_folder),
        experiment_evaluator_config=ExperimentEvaluatorConfig(
            conversation_config=ConversationEvaluatorConfig(verify_pubtator_ids=True),
            experiment_path="",
            # Currently publications are hardcoded, but could be made configurable
            publications=[
                "patil",
                "schlosser",
                "monteiro-martins",
                "scherer",
                "steinbrenner",
                "rodriguez-hernandez",
            ],
        ),
    )

    multi_evaluator = MultiExperimentEvaluator(config)
    if not multi_evaluator.has_plotting_support():
        print("Cannot create plots. Analysis dependencies not installed.")
        print("Install with: uv sync --group analysis")
        return

    print("Evaluating all PubTator IDs and generating plots...")
    await multi_evaluator.evaluate()
    multi_evaluator.generate_pubtator_plot(out_folder)
    multi_evaluator.generate_features_plot(out_folder)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render all conversations and evaluations of all experiments "
            "in a folder as PDFs."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input folder containing experiments",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output folder for rendered files",
    )
    parser.add_argument(
        "--overwrite",
        "-f",
        action="store_true",
        help="Overwrite existing HTML/PDF files.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating summary plots across all experiments",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Only generate plots, skip individual experiment rendering",
    )
    args = parser.parse_args()

    if not chrome:
        print(
            "Could not find Chrome/Chromium on your PATH. "
            "HTML output will not be converted to PDF."
        )

    console.file = io.StringIO()
    in_folder = Path(args.input)

    out_folder = Path(args.output)
    out_folder.mkdir(exist_ok=True)

    if args.plots_only and args.no_plots:
        print("Very funny")
        sys.exit(1)

    if not args.plots_only:
        asyncio.run(
            render_all_experiments(in_folder, out_folder, overwrite=args.overwrite)
        )

    if not args.no_plots:
        asyncio.run(plot(in_folder, out_folder))


if __name__ == "__main__":
    main()
