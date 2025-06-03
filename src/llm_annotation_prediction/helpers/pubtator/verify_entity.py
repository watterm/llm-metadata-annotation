from logging import getLogger

from llm_annotation_prediction.helpers.pubtator.common import (
    id_has_valid_prefix,
)
from llm_annotation_prediction.helpers.pubtator.find_entity_by_pub_search import (
    FindEntityByPublicationArguments,
    FindEntityByPublicationSearchStrategy,
)

_logger = getLogger("Pubtator")


async def verify_entity(id: str | None) -> bool:
    """
    Verifies that the ID of a Pubtator entry exists by conducting a search and
    checking the results for the entity ID. The ID needs to start with @.
    """
    _logger.debug(f"Verifying Pubtator entity: {id}")
    if id is None:
        return False

    if not id.startswith("@"):
        _logger.debug(f"Pubtator entity without '@' prefix: {id}")
        return False

    if not id_has_valid_prefix(id):
        _logger.debug(f"Pubtator entity with invalid prefix: {id}")
        return False

    if " " in id:
        _logger.debug(f"Pubtator entity with space: {id}")
        return False

    results = await FindEntityByPublicationSearchStrategy.find_ids(
        FindEntityByPublicationArguments(text=id)
    )

    for found_entity in results.results:
        if id in found_entity.pubtator_ids:
            return True

    return False
