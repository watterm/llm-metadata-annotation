from abc import ABC, abstractmethod
from enum import StrEnum
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from llm_annotation_prediction.helpers.open_router import Tool, ToolCall
from llm_annotation_prediction.helpers.rate_limiter import (
    RateLimiterConfig,
    RedisRateLimiterConfig,
    create_rate_limiter,
)
from llm_annotation_prediction.helpers.rate_limiter.base import AsyncRateLimiter


class EntityType(StrEnum):
    """
    Describes the prefixes in PubTator IDs as well as possible values for the concept
    filter in the FindEntityId endpoint. Note that species and cell line do not work
    there.
    """

    GENE = "GENE"
    DISEASE = "DISEASE"
    CHEMICAL = "CHEMICAL"
    VARIANT = "VARIANT"
    SPECIES = "SPECIES"  # Does not work in FindEntityId
    CELLLINE = "CELLLINE"  # Does not work in FindEntityId


# I'm not sure if some LLMs have issues with non-programming like function names,
# so I'll go with the snake_case. We define it with the same name for the FindEntityId
# tool and the FindIdByPublication Tool to be able to compare them without inadvertently
# changing the LLM performance through other names/descriptions. They should not be used
# together.
PUBTATOR_TOOL_NAME = "pubtator_id_search"
# PUBTATOR_TOOL_NAME="FindEntityID", # Original name in PubTator GPT config

PUBTATOR_TOOL_DESCRIPTION = (
    "Given an entity, return its associated entity IDs. Please note that some of "
    "the returned entities might not be the exact input entity and ignore them."
)

QUERY_ARGUMENT_DESCRIPTION = (
    "The free text query representing the bioconcept entity name to search for"
)


def id_has_valid_prefix(id: str) -> bool:
    """
    Check if the the provided ID has a prefix that is a PubTator entity type.
    """
    parts = id.split("_", 1)

    prefix = parts[0]
    prefix = prefix.removeprefix("@")

    return prefix.upper() in EntityType


PubtatorArguments = TypeVar("PubtatorArguments", bound=BaseModel)
PubtatorResults = TypeVar("PubtatorResults")


class PubtatorStrategy(Generic[PubtatorArguments, PubtatorResults], ABC):
    """
    Base class for the different ways of querying for Pubtator IDs. See the
    find_entity_py_pub_search module for a detailled explanation.
    """

    # The definition of the tool use, that will be inserted into the LLM request
    tool: Tool

    # The arguments class, which the handler needs to know about
    Arguments: type[PubtatorArguments]

    # Class-wide (shared) rate limiter for all Pubtator API calls.
    # Now lazily created to allow switching between local and redis implementations.
    _rate_limiter: ClassVar[AsyncRateLimiter | None] = None
    _rate_limiter_config: ClassVar[RateLimiterConfig | RedisRateLimiterConfig] = (
        RateLimiterConfig(
            name="PubtatorRateLimiter",
            min_rps=1.0,
            max_rps=2.0,
            initial_rps=2.0,
        )
    )

    @classmethod
    def configure_rate_limiter(
        cls,
        config: RateLimiterConfig | RedisRateLimiterConfig,
    ) -> None:
        """Set the shared limiter config (lazy creation on first use)."""
        cls._rate_limiter_config = config
        cls._rate_limiter = None

    @classmethod
    def _get_rate_limiter(
        cls,
    ) -> AsyncRateLimiter:
        """
        Lazy creation of the rate limiter on first use.
        """
        if cls._rate_limiter is None:
            cls._rate_limiter = create_rate_limiter(cls._rate_limiter_config)
        return cls._rate_limiter

    @classmethod
    async def close_rate_limiter(cls) -> None:
        """Close the shared rate limiter, if it has been created."""
        if cls._rate_limiter is not None:
            await cls._rate_limiter.close()
        cls._rate_limiter = None

    def validate_json(self, tool_call: ToolCall) -> PubtatorArguments:
        """
        Validates that the LLM's tool call arguments are in the expected format.
        """
        return self.Arguments.model_validate_json(tool_call.function.arguments)

    @classmethod
    @abstractmethod
    async def find_ids(cls, arguments: PubtatorArguments) -> PubtatorResults:
        """
        Make the necessary calls to Pubtator endpoints and return the search results.
        """
        pass

    @classmethod
    @abstractmethod
    def format_results(
        cls, arguments: PubtatorArguments, results: PubtatorResults
    ) -> str:
        """
        Format the search arguments and results into a string, that can be presented
        to an LLM.
        """
        pass
