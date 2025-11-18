from typing import Any, TypeVar

from pydantic import TypeAdapter

from llm_annotation_prediction.helpers.open_router import Message

# Filenames in experiment folders
CONVERSATIONS_FILENAME = "conversations.json"
CONTEXT_FILENAME = "data.json"
PAYLOADS_FOLDER = "payloads"

# Constants for expected filenames in publication folder
METADATA_FILENAME = "metadata.json"
PAPER_PDF_FILENAME = "paper.pdf"
PAPER_MD_FILENAME = "paper.md"

T = TypeVar("T")  # T will be Conversation or Context

ExperimentData = dict[str, list[T]]

# Helper type, so we don't have to import Dict and Any everywhere
Context = dict[str, Any]
Data = dict[str, list[Context]]
DataAdapter = TypeAdapter(Data)

Conversation = list[Message]
Conversations = dict[str, list[Conversation]]
ConversationsAdapter = TypeAdapter(Conversations)
