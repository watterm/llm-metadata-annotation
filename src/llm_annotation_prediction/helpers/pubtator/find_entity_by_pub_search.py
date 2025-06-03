"""
This modules offers an LLM tool to find Pubtator IDs via the publication search
endpoint. This is a workaround to the incomplete endpoint in `find_entity_id.py`.

The publication search results contain a highlighted text in <m></m> tags, which
marks the found query text. If this is a Pubtator ID, it will be prefixed with "@".
It it is a normalized name, it will be surrounded with "@@@". Synonyms pointing to the
same bioentity will always be preceding the normalized name. Consequently, we can
check neighbouring words of a search result to find all Pubtator IDs and the normalized
name.

Example:
    Search for "HeLa": https://www.ncbi.nlm.nih.gov/research/pubtator3-api/search/?text=HeLa
    Response contains:
        "text_hl": "... selectivity to @CELLLINE_CVCL:0030 @@@<m>HeLa</m>@@@ cells, and
            its IC50 value to @CELLLINE_CVCL:0030 @@@<m>HeLa</m>@@@ cells ...",

    Searching for the cell line "HeLa" highlights this query text in the normalized
    name (@@@HeLa@@@). Preceding this is the cell line id (@CELLLINE_CVCL:0030), which
    would otherwise not be found.

Assumptions:
    - Each search result from the database has a normalized name with @@@ markers
    - Each normalized name can have preceding PubTator IDs starting with @ and ending
      in with space
    - A normalized name and the IDs form a group pointing to the same entity. The first
      preceding non-ID word stops the group.
    - Search results can contain highlighted text (<m>) outside of PubTator groups. We
      ignore that.
"""

from logging import getLogger
from typing import List, Optional, Set

from httpx import URL, AsyncClient
from pydantic import BaseModel

from llm_annotation_prediction.helpers.open_router import FunctionDescription, Tool
from llm_annotation_prediction.helpers.pubtator.common import (
    PUBTATOR_TOOL_DESCRIPTION,
    PUBTATOR_TOOL_NAME,
    QUERY_ARGUMENT_DESCRIPTION,
    PubtatorStrategy,
    id_has_valid_prefix,
)

_logger = getLogger("Pubtator")


search_endpoint = URL("https://www.ncbi.nlm.nih.gov/research/pubtator3-api/search/")


find_entity_by_publication_description = FunctionDescription(
    name=PUBTATOR_TOOL_NAME,
    description=PUBTATOR_TOOL_DESCRIPTION,
    parameters={
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": QUERY_ARGUMENT_DESCRIPTION},
            # "concept": {
            #     "type": "string",
            #     "enum": [e.name for e in EntityType],  # allowed values from EntityType
            # },
        },
    },
)

# The complete tool description we append to the request
find_entity_by_publication_tool = Tool(
    type="function", function=find_entity_by_publication_description
)


class PubtatorPublicationSearchResult(BaseModel):
    """
    Represents a result in the publication search. It contains a lot more fields,
    but we're only extracting the relevant ones here.
    """

    text_hl: Optional[str]


class PubtatorPublicationSearchResults(BaseModel):
    """
    Represents a page of results in the publication search. We're not interested
    in the other fields (facets, page info).
    """

    results: list[PubtatorPublicationSearchResult]


class ExtractedEntity(BaseModel):
    """
    Describes a group of IDs with a normalized name that refer to the same entity.
    """

    normalized_name: Optional[str]
    pubtator_ids: List[str]


class NormalizedEntry(BaseModel):
    """
    Describes a normalized name (e.g. @@@HeLa@@@) with the text preceding and following
    it.
    """

    name: str  # Text without @@@
    leading_text: str
    trailing_text: str


async def _search_publications(query: str) -> List[PubtatorPublicationSearchResult]:
    """
    Makes the HTTP call to the publication search endpoint and returns the validated
    results. We will get up to 10 publications in one call and it seems there's no
    query parameter to change that limit.
    """
    query_url = search_endpoint.copy_with(params={"text": query})

    async with AsyncClient(timeout=60) as client:
        response = await client.get(query_url)

        # We sometimes get redirected to another server. Not sure if that one works.
        if response.status_code == 302:
            location = response.headers["Location"]
            _logger.warning(f"Putator search got redirected to: {location}")
            redirect_url = URL(location, params={"text": query})
            response = await client.get(redirect_url)

        response.raise_for_status()

    _logger.debug(response.json())

    return PubtatorPublicationSearchResults(**response.json()).results


def _find_normalized_name(text: str) -> NormalizedEntry | None:
    """
    Finds one occurence of a normalized name and splits the text. Returns a normalized
    entry or None it no tag was found.
    """
    try:
        start = text.index("@@@")
        end = text.index("@@@", start + 3)

        return NormalizedEntry(
            name=text[start + 3 : end],
            leading_text=text[:start],
            trailing_text=text[end + 3 :],
        )
    except ValueError:
        _logger.debug("Could not find more normalized names.")
    return None


def _extract_id_text(text: str) -> str | None:
    """
    Returns the substring of text up to a space, if it has a pubtator prefix. Ignores
    markers.
    """
    partition = text.partition(" ")
    marker_free_id = _remove_marker(partition[0])
    if partition[1] == " " and id_has_valid_prefix(marker_free_id):
        return partition[0]
    return None


def _extract_putator_ids(text: str) -> List[str]:
    """
    Scan the text *from the end* for strings starting with @ and ending with space.
    These are PubTator IDs.
    """

    ids = []
    scan_text = text
    while scan_text:
        partition = scan_text.rpartition("@")

        # Without @ there can't be a Pubtator ID
        if partition[1] != "@":
            break

        id_text = _extract_id_text(partition[2])

        # On the first invalid ID, the group ends and we stop looking
        if not id_text:
            break

        ids.append(f"@{id_text}")
        scan_text = partition[0]

    return ids


def _remove_marker(text: str) -> str:
    """
    Remove the highlight tag from text.
    """
    return text.replace("<m>", "").replace("</m>", "")


def _create_entity_if_marked(
    normalized_name: str, pubtator_ids: List[str]
) -> ExtractedEntity | None:
    """
    Create and entity if the highlight marker is found in the normalized name or one
    of the detected IDs. This verifies that it's a search result and not accidental
    context in the found text.
    """

    # Check the normalized name for the marker first, then all pubtator IDs.
    marked = "<m>" in normalized_name
    if not marked:
        marked = next((True for id in pubtator_ids if "<m>" in id), False)

    if marked:
        return ExtractedEntity(
            normalized_name=_remove_marker(normalized_name),
            pubtator_ids=[_remove_marker(id) for id in pubtator_ids],
        )
    else:
        return None


def _extract_entities(text: str) -> List[ExtractedEntity]:
    """
    Extract all PubTator entities in the highlighted text.
    """
    _logger.debug(f"Extracting IDs from text: '{text}'")

    entities: List[ExtractedEntity] = []
    scan_text: str | None = text
    while scan_text:
        normalized_entry = _find_normalized_name(scan_text)

        # No more normalized names in the text
        if not normalized_entry:
            break

        # We only add entities if they have highlight tags <m> in them
        ids = _extract_putator_ids(normalized_entry.leading_text)
        entity = _create_entity_if_marked(
            normalized_name=normalized_entry.name, pubtator_ids=ids
        )
        if entity:
            _logger.debug(f"Extracted marked entity {entity.model_dump()}")
            entities.append(entity)
        else:
            _logger.debug(
                f"Ignoring unmarked entity with normalized name {normalized_entry.name}"
            )

        scan_text = normalized_entry.trailing_text

    return entities


class FindEntityByPublicationArguments(BaseModel):
    """
    The arguments for the Pubtator function call subject to the JSON schema sent
    to the LLM.
    """

    text: str


class FindEntityByPublicationResults(BaseModel):
    # Mimicks FindEntityIdResults
    results: List[ExtractedEntity]


class FindEntityByPublicationSearchStrategy(
    PubtatorStrategy[FindEntityByPublicationArguments, FindEntityByPublicationResults]
):
    tool = find_entity_by_publication_tool
    Arguments = FindEntityByPublicationArguments

    @classmethod
    async def find_ids(
        cls, arguments: FindEntityByPublicationArguments
    ) -> FindEntityByPublicationResults:
        """
        Finds all occurences in the PubTator publication search results that are highlighted
        to match the query and that are PubTator entities. Returns a list of all extracted
        entities, which likely contains duplicate IDs and possibly several variants of
        normalized names (e.g. "Human", "Patient" have the same species ID).
        """
        _logger.debug(f"Searching Pubtator for query '{arguments.text}'")
        all_entities: List[ExtractedEntity] = []
        publications = await cls._rate_limiter.enqueue(
            _search_publications, arguments.text
        )

        for publication in publications:
            if publication.text_hl is not None:
                all_entities.extend(_extract_entities(publication.text_hl))
        return FindEntityByPublicationResults(results=all_entities)

    @classmethod
    def format_results(
        cls,
        arguments: FindEntityByPublicationArguments,
        results: FindEntityByPublicationResults,
    ) -> str:
        """
        Convert extracted entities to a markdown list. Exactly identical results are
        only listed once. Spelling differences in the normalized name are not considered.
        """
        _logger.debug(f"Converting to text: {results.results}")
        list_elements: Set[str] = set()
        for group in results.results:
            id_list = " ".join(group.pubtator_ids)
            list_elements.add(f"- {group.normalized_name}: {id_list}")

        if len(list_elements) > 0:
            list_string = "\n".join(list_elements)
        else:
            list_string = "No IDs found."

        return f"Pubtator entity search results for query '{arguments.text}':\n{list_string}"
