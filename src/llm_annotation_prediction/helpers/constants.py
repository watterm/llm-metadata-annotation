from typing import Any, Dict, List, TypeVar

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

ExperimentData = Dict[str, List[T]]

# Helper type, so we don't have to import Dict and Any everywhere
Context = Dict[str, Any]
Data = Dict[str, List[Context]]
DataAdapter = TypeAdapter(Data)

Conversation = List[Message]
Conversations = Dict[str, List[Conversation]]
ConversationsAdapter = TypeAdapter(Conversations)
