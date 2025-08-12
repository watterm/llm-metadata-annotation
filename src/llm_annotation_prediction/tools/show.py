"""
Helper tool to show conversations and evaluations from experiments.
"""

import argparse
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generic, List, Optional, Tuple, cast, overload

from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table

from llm_annotation_prediction.evaluation.conversation_evaluator import (
    ConversationEvaluator,
    ConversationEvaluatorConfig,
)
from llm_annotation_prediction.helpers.constants import (
    CONTEXT_FILENAME,
    CONVERSATIONS_FILENAME,
    Context,
    Conversation,
    Conversations,
    ConversationsAdapter,
    Data,
    DataAdapter,
    ExperimentData,
    T,
)
from llm_annotation_prediction.helpers.logging import LOG_FILENAME
from llm_annotation_prediction.helpers.open_router import Message, UserMessage
from llm_annotation_prediction.helpers.setup import EXPERIMENT_FOLDER

console = Console(width=100, record=True)


@dataclass
class TrialEntry(Generic[T]):
    """
    Helper class to represent a trial with its indices for titles.
    """

    trial: T
    uuid: str
    trial_index: int


def get_latest_experiment_folder() -> Optional[str]:
    """
    Retrieves the folder with the latest experiment.
    """
    folders = [
        f
        for f in os.listdir(EXPERIMENT_FOLDER)
        if os.path.isdir(os.path.join(EXPERIMENT_FOLDER, f))
    ]
    if not folders:
        print("Error: No experiments found")
        exit(1)
    return os.path.join(EXPERIMENT_FOLDER, sorted(folders)[-1])


def load_conversations(experiment_folder: str) -> Conversations:
    """
    Loads the conversations from the respective JSON in the experiment.
    """
    print(f"Loading experiment in {experiment_folder}")
    try:
        conversations_path = Path(experiment_folder) / CONVERSATIONS_FILENAME
        conversations = ConversationsAdapter.validate_json(
            open(conversations_path, encoding="utf-8").read()
        )
    except FileNotFoundError:
        print(
            (
                f"Error: Conversation file not found in '{experiment_folder}'. "
                "Is the experiment still running?"
            )
        )
        exit(1)
    return conversations


def _index_trials(data: ExperimentData[T]) -> List[Tuple[str, int]]:
    """
    Returns a flattened representation of UUIDs with trial indices for easier indexing by int.
    """
    return [
        (uuid, trial_index)
        for uuid, trials in data.items()
        for trial_index in range(len(trials))
    ]


def _show_trial_keys(
    data: ExperimentData[T], title: str = "Available conversations"
) -> None:
    """
    Show all available trials in an experiment with their index.
    """
    index = _index_trials(data)

    table = Table(title=title)
    table.add_column("Index", justify="right")
    table.add_column("UUID")
    table.add_column("Trial")

    for i, (uuid, trial) in enumerate(index):
        table.add_row(str(i), uuid, str(trial))

    console.print()
    console.print(table)


def _get_trial_by_index(data: ExperimentData[T], trial_id: str) -> TrialEntry[T]:
    """
    Returns a trial by integer index from data (Conversations or Context).
    """
    trials_index = _index_trials(data)
    n_indices = len(trials_index)
    index = int(trial_id)
    if -n_indices - 1 < index < n_indices:
        uuid, trial_index = trials_index[index]
        return TrialEntry(
            trial=data[uuid][trial_index], uuid=uuid, trial_index=trial_index
        )

    print(f"Index {index} is out of bound for {n_indices} trials")
    exit(1)


def _get_trial_by_uuid(data: ExperimentData[T], trial_id: str) -> TrialEntry[T]:
    """
    Returns a trial when a UUID is provided. Uses "_i" suffix for trial index.
    """
    n_trials = len(list(data.values())[0])
    id_parts = trial_id.split("_")

    try:
        uuid, trial_index_str = id_parts
        trial_index = int(trial_index_str)
    except ValueError:
        print(
            (
                "Error: Invalid trial index. If this experiment had repeated trials per "
                "publication, a trial index number is required after the UUID. Example: "
            )
        )
        print("\tf4326b84-beb3-40f4-98fa-539789a22e28_1")
        exit(1)

    if -n_trials - 1 < trial_index < n_trials:
        return TrialEntry(
            trial=data[uuid][trial_index], uuid=uuid, trial_index=trial_index
        )
    else:
        print(
            f"Error: Trial index {trial_index} is out of bounds for {n_trials} trials"
        )
        exit(1)


@overload
def get_trial(
    data: Conversations,
    trial_id: str | None = None,
    title: str | None = None,
) -> TrialEntry[Conversation]: ...
@overload
def get_trial(
    data: Data,
    trial_id: str | None = None,
    title: str | None = None,
) -> TrialEntry[Context]: ...


def get_trial(
    data: Dict[str, List[T]],
    trial_id: str | None = None,
    title: str | None = None,
) -> TrialEntry[T]:
    """
    Returns the trial which is either specified by index or by UUID (with potential
    suffix) in `trial_id`.
    If trial_id is None, shows all available trials and exits.
    Overloaded to work with both Conversations and Context.
    """
    if trial_id is None:
        _show_trial_keys(data, title=title or "Available trials")
        exit(1)

    try:
        _ = int(trial_id)
        return cast(TrialEntry[T], _get_trial_by_index(data, trial_id))
    except ValueError:
        pass

    return cast(TrialEntry[T], _get_trial_by_uuid(data, trial_id))


def get_conversation(
    experiment_folder: str, conversation_id: str | None = None
) -> TrialEntry[Conversation] | None:
    """
    Returns the conversation which is either specified by index or by UUID
    (with potential suffix) in `conversation_id`.
    """
    conversations = load_conversations(experiment_folder)
    return get_trial(conversations, conversation_id, title="Available conversations")


def _get_number_of_tool_calls(message: Message) -> int:
    """
    Detects if the LLM made tool calls instead of answering
    """
    if isinstance(message, UserMessage):
        if isinstance(message.tool_calls, list):
            return len(message.tool_calls)

    return 0


def print_message(message: Message) -> None:
    """
    Pretty prints a provided message.
    """
    console.print(Panel(f"[bold]{message.role.capitalize()}[/bold]:"))

    text = str(message.content)
    n_tool_calls = _get_number_of_tool_calls(message)
    if not text and n_tool_calls > 0:
        text = f"[{message.role.capitalize()} answered with {n_tool_calls} tool calls]"

    formatted: RenderableType = Markdown(text)
    formatted = Panel(formatted)
    formatted = Padding(formatted, (0, 0, 0, 4))

    console.print(formatted)


def show_conversation(
    experiment_folder: str, conversation_id: str | None = None
) -> None:
    """
    Print all messages in a conversation.
    """
    trial_entry = get_conversation(experiment_folder, conversation_id)

    if trial_entry is None:
        print("Conversation not found")
        exit(1)

    conversation = trial_entry.trial
    console.print(
        f"Showing conversation '{trial_entry.uuid} (Trial {trial_entry.trial_index})':"
    )
    for message in conversation:
        print_message(message)


def show_message(
    experiment_folder: str, role: str, conversation_id: str, message_index: int
) -> None:
    """
    Print a specific message for the given role in a conversation.
    """
    trial_entry = get_conversation(experiment_folder, conversation_id)

    if trial_entry is None:
        return

    conversation = trial_entry.trial
    messages = [m for m in conversation if m.role == role]
    n_messages = len(messages)

    if -n_messages - 1 < message_index < n_messages:
        print_message(messages[message_index])
    else:
        print(
            f"Error: Prompt index {message_index} is out of bounds for {n_messages} {role} messages"
        )


def show_errors(experiment_folder: str) -> None:
    """Shows all ERROR level log messages from the experiment log."""

    log_path = Path(experiment_folder) / LOG_FILENAME
    if not log_path.exists():
        print(f"Error: Log file not found in {experiment_folder}")
        return

    error_entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "| ERROR" in line:
                error_entries.append(line.strip())

    if not error_entries:
        console.print("[green]No errors found in the log.[/green]")
        return

    table = Table(title="Error Messages")
    table.add_column("Timestamp")
    table.add_column("Logger")
    table.add_column("Message")

    for entry in error_entries:
        # Split by '|' and strip whitespace
        parts = [p.strip() for p in entry.split("|")]
        if len(parts) >= 4:
            timestamp = parts[0]

            if len(parts) == 5:  # DEBUG log level has 5 cols
                logger = f"{parts[2]}({parts[3]})"
                message = parts[4]
            else:
                logger = parts[2]
                message = parts[3]

            table.add_row(timestamp, logger, message)

    error_console = Console(width=160)
    error_console.print()
    error_console.print(table)


def load_data(experiment_folder: str) -> Data:
    """
    Loads the context data from the respective JSON in the experiment.
    """
    print(f"Loading experiment in {experiment_folder}")
    try:
        context_path = Path(experiment_folder) / CONTEXT_FILENAME
        with open(context_path, encoding="utf-8") as f:
            context = DataAdapter.validate_json(f.read())
    except FileNotFoundError:
        print(
            (
                f"Error: Context file not found in '{experiment_folder}'. "
                "Is the experiment still running?"
            )
        )
        exit(1)
    return context


async def evaluate_context(
    context: Context,
    title: str,
    verify_pubtator_ids: bool = True,
    disable_elements: bool = False,
    disable_description: bool = False,
) -> None:
    """
    Prints the evaluation statistics of the context.
    """
    config = ConversationEvaluatorConfig(verify_pubtator_ids=verify_pubtator_ids)
    evaluator = ConversationEvaluator(config, context)
    await evaluator.evaluate()
    table = evaluator.print_to_table(
        title=title,
        show_elements=not disable_elements,
        show_description=not disable_description,
    )
    console.print(table)


async def evaluate_experiment(
    experiment_folder: str,
    verify_pubtator_ids: bool = True,
    show_all: bool = False,
    disable_elements: bool = False,
    disable_description: bool = False,
    trial_id: str | None = None,
) -> None:
    """
    Prints the evaluation statistics of the experiment.
    """
    data = load_data(experiment_folder)

    settings = {
        "verify_pubtator_ids": verify_pubtator_ids,
        "disable_elements": disable_elements,
        "disable_description": disable_description,
    }

    if show_all:
        for uuid, trials in data.items():
            for index, trial in enumerate(trials):
                title = f"'{uuid}' (Trial {index})"
                await evaluate_context(trial, title, **settings)

    if not show_all:
        if trial_id is None:
            _show_trial_keys(data, title="Available evaluations")
            exit(1)
        else:
            trial_entry = get_trial(data, trial_id)
            title = f"'{trial_entry.uuid}' (Trial {trial_entry.trial_index})"
            await evaluate_context(trial_entry.trial, title, **settings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="""
            Show conversations in experiment objects. If no experiment folder is specified,
            will use the latest.
        """
    )
    parser.add_argument(
        "-e", "--experiment_folder", help="Optional path to experiment folder"
    )
    parser.add_argument(
        "-s",
        "--save-html",
        type=str,
        help="Save the output as HTML to the given path",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'conversation' command
    conv_parser = subparsers.add_parser("conversation", aliases=["conv", "c"])
    conv_parser.add_argument(
        "conversation_id", nargs="?", help="UUID or index of conversation"
    )
    conv_parser.set_defaults(
        func=lambda args: show_conversation(
            args.experiment_folder, args.conversation_id
        )
    )

    # 'prompt' command
    prompt_parser = subparsers.add_parser("prompt", aliases=["p"])
    prompt_parser.add_argument("conversation_id", help="UUID or index of conversation")
    prompt_parser.add_argument("prompt_index", type=int, nargs=1, help="Prompt index")
    prompt_parser.set_defaults(
        func=lambda args: show_message(
            args.experiment_folder, "user", args.conversation_id, args.prompt_index[0]
        )
    )

    # 'answer' command
    answer_parser = subparsers.add_parser("answer", aliases=["a"])
    answer_parser.add_argument("conversation_id", help="UUID or index of conversation")
    answer_parser.add_argument("answer_index", type=int, nargs=1, help="Answer index")
    answer_parser.set_defaults(
        func=lambda args: show_message(
            args.experiment_folder,
            "assistant",
            args.conversation_id,
            args.answer_index[0],
        )
    )

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", aliases=["eval"])
    eval_parser.add_argument("trial", nargs="?", help="UUID or index of the context")
    eval_parser.add_argument(
        "--no-verify-pubtator-ids",
        dest="verify_pubtator_ids",
        action="store_false",
        help="Disables the verification of the LLM's Pubtator IDs if they are not"
        "results from tool use.",
    )
    eval_parser.add_argument(
        "-e",
        "--disable-elements",
        action="store_true",
        help="Disables the printing of elements in the evaluation table.",
    )
    eval_parser.add_argument(
        "-d",
        "--disable-description",
        action="store_true",
        help="Disables the printing of description in the evaluation table.",
    )
    eval_parser.add_argument(
        "-a",
        "--show-all",
        action="store_true",
        help="Show all conversations in the experiment.",
    )

    async def eval_func(args: argparse.Namespace) -> None:
        await evaluate_experiment(
            experiment_folder=args.experiment_folder,
            verify_pubtator_ids=args.verify_pubtator_ids,
            show_all=args.show_all,
            disable_elements=args.disable_elements,
            disable_description=args.disable_description,
            trial_id=args.trial,
        )

    eval_parser.set_defaults(func=eval_func)

    # errors command
    error_parser = subparsers.add_parser("errors", aliases=["e"])
    error_parser.set_defaults(func=lambda args: show_errors(args.experiment_folder))

    args = parser.parse_args()
    if not args.experiment_folder:
        args.experiment_folder = get_latest_experiment_folder()

    if asyncio.iscoroutinefunction(args.func):
        asyncio.run(args.func(args))
    else:
        args.func(args)

    if args.save_html:
        console.save_html(args.save_html, inline_styles=True)
        print(f"Saved output to {args.save_html}")
