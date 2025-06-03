"""
This module implements the Pubtator FindEntityId functionality, which is the
official endpoint for searching for bioconcept entities by name. However, it seems
not to return entities for species and cell lines. We can work around this
by using the FindIdByPublication tool, which is a more generic search tool.
"""

# This describes the functionality and parameters of the tool to the LLM. The parameters
# are defined as JSON schema.
from logging import getLogger
from typing import List, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from llm_annotation_prediction.helpers.open_router import FunctionDescription, Tool
from llm_annotation_prediction.helpers.pubtator.common import (
    PUBTATOR_TOOL_DESCRIPTION,
    PUBTATOR_TOOL_NAME,
    EntityType,
    PubtatorStrategy,
)

_logger = getLogger("Pubtator")

find_entity_id_description = FunctionDescription(
    name=PUBTATOR_TOOL_NAME,
    description=PUBTATOR_TOOL_DESCRIPTION,
    parameters={
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The free text query representing the bioconcept entity name to "
                    "search for."
                ),
            },
            "concept": {
                "type": "string",
                "enum": [e.name for e in EntityType],  # allowed values from EntityType
            },
            "limit": {
                "type": "integer",
                "description": "Limit for the number of suggestions returned",
            },
        },
    },
)

# The complete tool description we append to the request
find_entity_id_tool = Tool(type="function", function=find_entity_id_description)

_FIND_ENTITY_ID_URL = (
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/entity/autocomplete/"
)


class FindEntityIdArguments(BaseModel):
    """
    The arguments for the Pubtator function call subject to the JSON schema sent
    to the LLM.
    """

    query: str
    concept: EntityType | None = None
    limit: int = 10

    @field_validator("concept")
    @classmethod
    def validate_concept(cls, v: str) -> str:
        if v.upper() not in EntityType:
            raise ValidationError(f"Invalid concept: {v}")
        return v.upper()


class FindEntityIdResult(BaseModel):
    """
    Single entity result from Pubtator. There's no formal description online, so the
    types are guessed.
    """

    id: str = Field(..., alias="_id")  # JSON contains _id, but that would get stripped
    biotype: str  # Presumable an enum, but can't find a list
    name: str
    description: Optional[str] = None
    match: str


class FindEntityIdResults(BaseModel):
    results: List[FindEntityIdResult]


async def _find_entity_id_task(
    search_arguments: FindEntityIdArguments,
) -> FindEntityIdResults:
    """
    Makes the HTTP request to the pubtator FindEntityId endpoint and parses the
    response. Intended to be used in the Pubtator worker.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            _FIND_ENTITY_ID_URL,
            params=search_arguments.model_dump(exclude_none=True),
        )
        payload = response.json()

        _logger.debug(f"Pubtator response payload {payload}")
        _logger.debug(f"Pubtator response headers {response.headers}")
    search_results = FindEntityIdResults(results=payload)
    return search_results


class FindEntityIdStrategy(
    PubtatorStrategy[FindEntityIdArguments, FindEntityIdResults]
):
    tool = find_entity_id_tool
    Arguments = FindEntityIdArguments

    @classmethod
    async def find_ids(cls, arguments: FindEntityIdArguments) -> FindEntityIdResults:
        results = await cls._rate_limiter.enqueue(_find_entity_id_task, arguments)
        if not isinstance(results, FindEntityIdResults):
            raise TypeError("Unexpected results from FindEntityId")
        return results

    @classmethod
    def format_results(
        cls, arguments: FindEntityIdArguments, results: FindEntityIdResults
    ) -> str:
        """
        Wrap the JSON answer in a markdown text block and include the query.
        """
        return (
            f"PubTator entity search results for query: '{arguments.query}'\n"
            "```json \n"
            f"{results.model_dump_json(indent=2)}\n"
            "```"
        )
