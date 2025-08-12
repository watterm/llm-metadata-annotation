import asyncio
from logging import getLogger
from pathlib import Path
from typing import Annotated, Dict, List

import httpx
from pydantic import Field

from llm_annotation_prediction.conversation import (
    OpenRouterConversation,
    OpenRouterConversationConfig,
)
from llm_annotation_prediction.dataset import Dataset
from llm_annotation_prediction.experiments.experiment import (
    Experiment,
    ExperimentConfig,
)
from llm_annotation_prediction.helpers.constants import (
    CONTEXT_FILENAME,
    CONVERSATIONS_FILENAME,
    PAYLOADS_FOLDER,
    Context,
)
from llm_annotation_prediction.helpers.rate_limiter import RateLimitedQueue
from llm_annotation_prediction.helpers.save import dump_to_json

_logger = getLogger("AsyncExperiment")


class AsyncExperimentConfig(ExperimentConfig):
    type: str = "AsyncExperiment"

    # Rate limit: See https://openrouter.ai/docs/limits
    # Note that this depends on the number of remaining credits.
    max_requests_per_second: Annotated[float, Field(gt=0, le=100)] = 10

    # Repeat the same conversation for a publication to estimate
    # LLM randomness
    num_trials_per_publication: Annotated[int, Field(gt=0)] = 1


class AsyncExperiment(Experiment[AsyncExperimentConfig]):
    """
    Asynchronous experiment class for handling multiple conversations concurrently.
    """

    def __init__(
        self,
        config: AsyncExperimentConfig,
        conversation_config: OpenRouterConversationConfig,
        dataset: Dataset,
    ):
        super().__init__(config, conversation_config, dataset)
        self._config: AsyncExperimentConfig = config
        self._conversations: Dict[str, List[OpenRouterConversation]] = {}
        self._rate_limit: float = config.max_requests_per_second

        # This limits the number of concurrent requests to OpenRouter for all conversations
        self._rate_limiter: RateLimitedQueue[httpx.Response | None] = RateLimitedQueue(
            name="OpenRouterLimiter",
            min_rps=1,
            max_rps=self._rate_limit,
            initial_rps=self._rate_limit,
            max_retries=2,
        )

    def _create_conversations(self) -> Dict[str, List[OpenRouterConversation]]:
        """
        Creates a conversation for each trial of a publication.
        """
        if not self._dataset.is_loaded:
            raise RuntimeError("Dataset has not been loaded")

        conversations: Dict[str, List[OpenRouterConversation]] = {}
        for uuid, publication in self._dataset.publications.items():
            trials = []
            for trial in range(self._config.num_trials_per_publication):
                trials.append(
                    OpenRouterConversation(
                        config=self._conversation_config,
                        rate_limiter=self._rate_limiter,
                        publication=publication,
                        schema=self._dataset.schema,
                        trial=trial,
                    )
                )
            conversations[uuid] = trials
        return conversations

    async def _run_async(self) -> None:
        """
        Run rate-limited experiment. All conversations will be run concurrently
        """
        converse_methods = []
        for pub_conversations in self._conversations.values():
            for trial in pub_conversations:
                converse_methods.append(trial.converse())

        results = await asyncio.gather(*converse_methods)

        failed_count = results.count(False)
        _logger.info(f"{failed_count} of {len(results)} conversations failed.")

    def run(self) -> None:
        """
        Start the experiment. Responsible for handling the async process.
        """
        _logger.debug("Setting up conversations")
        self._conversations = self._create_conversations()

        _logger.info("Starting experiment")
        asyncio.run(self._run_async())

    def save(self, folder: Path) -> None:
        """
        Saves messages and context for every conversation
        """

        conversations: Dict[str, List[OpenRouterConversation]] = {}
        contexts: Dict[str, List[Context]] = {}

        for uuid, pub_conversations in self._conversations.items():
            conversations[uuid] = []
            contexts[uuid] = []
            for trial in pub_conversations:
                conversation_data = trial.to_dict()
                conversations[uuid].append(conversation_data["conversation"])
                contexts[uuid].append(conversation_data["context"])

        dump_to_json(folder / CONVERSATIONS_FILENAME, conversations)
        dump_to_json(folder / CONTEXT_FILENAME, contexts)

        payloads_folder = folder / PAYLOADS_FOLDER
        payloads_folder.mkdir(exist_ok=True)

        for uuid, pub_conversations in self._conversations.items():
            for i, conv in enumerate(pub_conversations):
                payload_file = payloads_folder / f"{uuid}_{i}.json"
                conv_dict = conv.to_dict()
                dump_to_json(payload_file, conv_dict["payloads"])
