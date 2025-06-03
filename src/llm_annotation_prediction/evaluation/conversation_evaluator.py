import asyncio
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field, PrivateAttr
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llm_annotation_prediction.helpers.constants import Context
from llm_annotation_prediction.helpers.pubtator.verify_entity import verify_entity
from llm_annotation_prediction.helpers.utils import get

_logger = getLogger("ContextEvaluator")


class ConversationEvaluatorConfig(BaseModel):
    type: str = "ConversationEvaluator"

    publication_list_key: str = "publication_list"
    pubtator_list_key: str = "pubtator_list"
    schema_list_key: str = "schema_list"
    consolidated_list_key: str = "consolidated_list"

    # Verifies that Pubtator IDs actually exist by searching in Pubtator.
    # This slows down the evaluation process significantly.
    verify_pubtator_ids: bool = True


class Entity(BaseModel):
    """
    Represents a biomedical entity as expected from an experiment.
    """

    entity_name: str
    in_pubtator: Optional[bool] = None
    pubtator_id: Optional[str] = None
    schema_category: Optional[str] = None
    from_provided_schema: Optional[bool] = None


class EntityLists(BaseModel):
    """
    Collects all lists of entities from the conversation and the derived evaluations.

    Makes the following assumptions about the context of an experiment:
     - Field "schema" contains the collection of markdown lists created from the schema
     - Field "pubtator" contains all Pubtator tool calls and results made by the Pubtator handler
     - It contains 4 lists with results for each step:
        "publication_list", "pubtator_list", "schema_list", "consolidated_list"
    """

    publication_list: List[Entity] = Field(
        default_factory=list,
        title="Step 1: Publication Entities",
        description="LLM-extracted entities from the publication text",
    )
    pubtator_list: List[Entity] = Field(
        default_factory=list,
        title="Step 2: Pubtator Entities",
        description="LLM-extracted entities after using the Pubtator tool",
    )
    schema_list: List[Entity] = Field(
        default_factory=list,
        title="Step 3: Schema Entities",
        description="LLM-extracted entities from the provided schema",
    )
    consolidated_list: List[Entity] = Field(
        default_factory=list,
        title="Step 4: Consolidated List",
        description="Final list as composed by the LLM",
    )

    # Pubtator Evaluation
    pubtator_ids: Set[str] = Field(
        default_factory=set,
        title="Pubtator IDs",
        description="Pubtator IDs extracted from the search results",
    )
    pubtator_names: Set[str] = Field(
        default_factory=set,
        title="Pubtator Names",
        description="Normalized Pubtator names extracted from the search results",
    )
    pubtator_queries: Set[str] = Field(default_factory=set, title="Pubtator Queries")
    pubtator_hits: Set[str] = Field(
        default_factory=set,
        title="Pubtator hits",
        description="Pubtator searches with results",
    )
    pubtator_misses: Set[str] = Field(
        default_factory=set,
        title="Pubtator Misses",
        description="Pubtator searches without results",
    )

    # Distinguish if Pubtator ID in or not in search results
    true_positive_pubtator_entities: List[Entity] = Field(
        default_factory=list,
        title="True Positive Pubtator Entities",
        description="Entities in Pubtator list that were found in searches",
    )
    false_positive_pubtator_entities: List[Entity] = Field(
        default_factory=list,
        title="False Positives Pubtator Entities",
        description="Entities with Pubtator ID in Pubtator list that do not exist in Pubtator. "
        "Does not depend on tool calls.",
    )
    entities_with_unmatched_pubtator_ids: List[Entity] = Field(
        default_factory=list,
        title="Entities With Unmatched Pubtator IDs",
        description="Entities in Pubtator list that were not found in searches",
    )

    # Entities in step 2 that the LLM claimed to be sourced from the schema or not
    true_positives_from_schema: List[Entity] = Field(
        default_factory=list,
        title="True Positives From Schema (Unreliable)",
        description="Entities in schema list that LLM correctly claimed to be from schema. "
        "Schema evaluation is based on strict text matching and disregards categories. ",
    )
    true_negatives_from_schema: List[Entity] = Field(
        default_factory=list,
        title="True Negatives From Schema (Unreliable)",
        description="Entities in schema list that LLM correctly claimed not to be from schema. "
        "Schema evaluation is based on strict text matching and disregards categories. ",
    )
    false_positives_from_schema: List[Entity] = Field(
        default_factory=list,
        title="False Positives From Schema (Unreliable)",
        description="Entities in schema list that LLM incorrectly claimed to be from schema. "
        "Schema evaluation is based on strict text matching and disregards categories. ",
    )
    false_negatives_from_schema: List[Entity] = Field(
        default_factory=list,
        title="False Negatives From Schema (Unreliable)",
        description=(
            "Entities in schema list that LLM incorrectly claimed not to be from schema. "
            "Schema evaluation is based on strict text matching and disregards categories. "
        ),
    )

    # Entities from step 4
    # Entities that have been in steps 2 or 3, either identically or manipulated
    copied_schema_entities: List[Entity] = Field(
        default_factory=list,
        title="Copied Schema Entities",
        description="Entities in consolidated list that were copied from schema list",
    )
    changed_schema_entities: List[Entity] = Field(
        default_factory=list,
        title="Changed Schema Entities",
        description="Entities in consolidated list that were changed from schema list",
    )
    copied_pubtator_entities: List[Entity] = Field(
        default_factory=list,
        title="Copied Pubtator Entities",
        description="Entities in consolidated list that were copied from Pubtator list",
    )
    changed_pubtator_entities: List[Entity] = Field(
        default_factory=list,
        title="Changed Pubtator Entities",
        description="Entities in consolidated list that were changed from Pubtator list",
    )

    new_entities_in_consolidated_list: List[Entity] = Field(
        default_factory=list,
        title="New Entities In Consolidated List",
        description="Entities in consolidated list that were not in schema or Pubtator lists",
    )

    def __init__(self, config: ConversationEvaluatorConfig, context: Context):
        super().__init__()
        self._config = config
        self._context = context
        self._evaluated: bool = False

    async def evaluate(self) -> None:
        """
        Evaluates the conversation and all entity lists.
        """
        self._parse_entity_lists()
        self._parse_pubtator_calls()
        await self._evaluate_pubtator_entities()
        self._evaluate_schema_entities()
        self._compare_consolidated_list()
        self._evaluated = True

    def print_to_table(
        self, show_elements: bool = True, show_description: bool = False
    ) -> Table:
        """
        Generates a rich table with all entity lists.
        """
        if not self._evaluated:
            raise ValueError("Entity lists have not been evaluated yet")

        list_table = Table(title="Entity lists", show_lines=True, title_justify="left")
        list_table.add_column("List", width=30)
        list_table.add_column("Count", style="bold")
        if show_elements:
            list_table.add_column("Elements")
        if show_description:
            list_table.add_column("Description", max_width=50)

        for field_name, field in EntityLists.model_fields.items():
            container = getattr(self, field_name)

            # Check if the container has entities
            if len(container) > 0 and isinstance(next(iter(container)), Entity):
                container = [entity.entity_name for entity in container]

            row = [field.title, str(len(container))]

            if show_elements:
                row.append(", ".join(container))
            if show_description:
                row.append(field.description)
            list_table.add_row(*row)

        return list_table

    def _get_entities(self, key: str) -> List[Entity]:
        """
        Extracts entities from a result list in the context.
        """
        container = self._context.get(key, {})
        if "entity_list" not in container:
            _logger.debug(f"No entity list found for key {key}")
            return []

        return [Entity.model_validate(item) for item in container["entity_list"]]

    def _parse_entity_lists(self) -> None:
        """
        Creates entity lists for all steps of a conversation.
        """
        _logger.info("Parsing entity lists")

        self.publication_list = self._get_entities(self._config.publication_list_key)
        self.pubtator_list = self._get_entities(self._config.pubtator_list_key)
        self.schema_list = self._get_entities(self._config.schema_list_key)
        self.consolidated_list = self._get_entities(self._config.consolidated_list_key)

    def _parse_pubtator_calls(self) -> None:
        """
        Converts the stored Pubtator calls to a cleaner format.
        """
        if "pubtator" not in self._context:
            _logger.info("No Pubtator calls found")
            return

        calls: Dict[str, Any] = self._context.get("pubtator", {})
        for id, call in calls.items():
            search_term = get(call, "arguments", "text")
            if not search_term:
                raise ValueError(f"Pubtator call '{id}' has no search term")

            self.pubtator_queries.add(search_term)

            results = get(call, "search_results", "results")
            if not results or len(results) == 0:
                self.pubtator_misses.add(search_term)
            else:
                self.pubtator_hits.add(search_term)
                for hit in results:
                    self.pubtator_names.add(hit["normalized_name"])
                    self.pubtator_ids.update(hit["pubtator_ids"])

    def _find_unqueried_pubtator_ids(self, entity_list: List[Entity]) -> List[Entity]:
        """
        Finds entities with Pubtator IDs which the LLM did not query for with the
        Pubtator tool.
        """
        return [
            entity
            for entity in entity_list
            if entity.pubtator_id not in self.pubtator_ids
        ]

    async def _evaluate_pubtator_entities(self) -> None:
        """
        Check if the pubtator entries have been queried or hallucinated.
        """
        entities_to_verify: List[Entity] = []
        for entity in self.pubtator_list:
            if entity.pubtator_id in self.pubtator_ids:
                self.true_positive_pubtator_entities.append(entity)
            else:
                if self._config.verify_pubtator_ids:
                    entities_to_verify.append(entity)
                self.entities_with_unmatched_pubtator_ids.append(entity)

        if len(entities_to_verify) > 0:
            self.false_positive_pubtator_entities.extend(
                await self._find_invalid_pubtator_ids(entities_to_verify)
            )

    async def _find_invalid_pubtator_ids(self, entities: List[Entity]) -> List[Entity]:
        """
        Verifies if the Pubtator IDs actually exist in Pubtator and returns a list of
        entities that do not exist.
        """
        to_verify: List[Entity] = [entity for entity in entities if entity.in_pubtator]
        verification_tasks = [verify_entity(entity.pubtator_id) for entity in to_verify]

        results = await asyncio.gather(*verification_tasks)

        invalid: List[Entity] = []
        for entity, is_valid in zip(to_verify, results, strict=True):
            if not is_valid:
                invalid.append(entity)
        return invalid

    def _evaluate_schema_entities(self) -> None:
        """
        Determines the outcomes of the binary classification task in step 3
        """
        for entity in self.schema_list:
            found: bool = False

            for list_string in self._context.get("schema", {}).values():
                if entity.entity_name in list_string:
                    found = True
                    break

            if found:
                if entity.from_provided_schema:
                    self.true_positives_from_schema.append(entity)
                else:
                    self.false_negatives_from_schema.append(entity)
            else:
                if entity.from_provided_schema:
                    self.false_positives_from_schema.append(entity)
                else:
                    self.true_negatives_from_schema.append(entity)

    def _compare_consolidated_list(self) -> None:
        """
        Compares entities to the schema and Pubtator lists from previous steps.
        """
        for entity in self.consolidated_list:
            in_schema_list: bool = False
            for schema_entity in self.schema_list:
                if entity.entity_name == schema_entity.entity_name:
                    in_schema_list = True
                    if self._equal_schema_entity(entity, schema_entity):
                        self.copied_schema_entities.append(entity)
                    else:
                        self.changed_schema_entities.append(entity)

            in_pubtator_list: bool = False
            for pubtator_entity in self.pubtator_list:
                if entity.entity_name == pubtator_entity.entity_name:
                    in_pubtator_list = True
                    if self._equal_pubtator_entity(entity, pubtator_entity):
                        self.copied_pubtator_entities.append(entity)
                    else:
                        self.changed_pubtator_entities.append(entity)

            if not in_schema_list and not in_pubtator_list:
                self.new_entities_in_consolidated_list.append(entity)

    def _equal_schema_entity(self, entity_a: Entity, entity_b: Entity) -> bool:
        return (
            entity_a.entity_name == entity_b.entity_name
            and entity_a.from_provided_schema == entity_b.from_provided_schema
            and entity_a.schema_category == entity_b.schema_category
        )

    def _equal_pubtator_entity(self, entity_a: Entity, entity_b: Entity) -> bool:
        return (
            entity_a.entity_name == entity_b.entity_name
            and entity_a.pubtator_id == entity_b.pubtator_id
            and entity_a.in_pubtator == entity_b.in_pubtator
        )


class GeneralStatistics(BaseModel):
    """
    Evaluates statistics of the conversation besides the entities.
    """

    conversation_successful: bool = Field(
        default=False,
        title="Conversation succeeded",
        description="A conversation succeeded, if no errors occured.",
    )
    http_elapsed_time: float | None = Field(
        default=None,
        title="HTTP Elapsed Time",
        description="Time in seconds for all HTTP requests made in the conversation, "
        "excluding Pubtator calls. Note that concurrent conversations can increase this.",
    )
    total_prompt_tokens: int | None = Field(
        default=None,
        title="Total Prompt Tokens",
        description="Total number of tokens in all prompts.",
    )
    total_completion_tokens: int | None = Field(
        default=None,
        title="Total Completion Tokens",
        description="Total number of tokens in all completions.",
    )

    def __init__(self, config: ConversationEvaluatorConfig, context: Context):
        super().__init__()
        self._config = config
        self._context = context

        self.conversation_successful = context.get("succeeded", False)

        # Sum up all HTTP elapsed times
        times = context.get("http_elapsed_time", None)
        self.http_elapsed_time = sum(times) if times else None

        # Sum up all tokens
        usage = context.get("usage", [])
        self.total_prompt_tokens = sum(
            item["prompt_tokens"] for item in usage if "prompt_tokens" in item
        )
        self.total_completion_tokens = sum(
            item["completion_tokens"] for item in usage if "completion_tokens" in item
        )

    def print_to_table(self, show_description: bool = False) -> Table:
        """
        Prints all general statistics to a table.
        """
        stringifiers: dict[str, Callable[[Any], str]] = {
            "conversation_successful": lambda x: "[green]Yes" if x else "[red]No",
            "http_elapsed_time": lambda x: f"{x:.1f}s",
            "total_prompt_tokens": lambda x: f"{x:,}",
            "total_completion_tokens": lambda x: f"{x:,}",
        }

        table = Table(
            title="General statistics",
            show_lines=True,
            show_header=False,
            title_justify="left",
        )

        table.add_column("Field", width=30)
        table.add_column("Value", style="bold")

        if show_description:
            table.add_column("Description", max_width=50)

        for field_name, field in GeneralStatistics.model_fields.items():
            field_value = getattr(self, field_name)
            stringifier = stringifiers.get(field_name, None)

            if stringifier and field_value is not None:
                field_value = stringifier(field_value)

            if field_value is None:
                field_value = "-"

            if show_description:
                table.add_row(field.title, str(field_value), field.description)
            else:
                table.add_row(field.title, str(field_value))

        return table


class ConversationEvaluator(BaseModel):
    """
    Evaluates the performance of the LLM for a single conversation.
    """

    _config: ConversationEvaluatorConfig = PrivateAttr()
    _context: Context = PrivateAttr()

    lists: EntityLists
    general: GeneralStatistics

    def __init__(self, config: ConversationEvaluatorConfig, context: Context):
        super().__init__(
            lists=EntityLists(config, context),
            general=GeneralStatistics(config, context),
        )
        self._config = config
        self._context = context
        self._evaluated: bool = False

    async def evaluate(self) -> None:
        await self.lists.evaluate()
        self._evaluated = True

    def print_to_table(
        self, title: str, show_elements: bool = True, show_description: bool = False
    ) -> Panel:
        """
        Prints the fields of the evaluator
        """
        if not self._evaluated:
            raise ValueError("Conversation has not been evaluated yet")

        general_table = self.general.print_to_table(show_description=show_description)
        list_table = self.lists.print_to_table(
            show_elements=show_elements, show_description=show_description
        )

        titleText = Text(title, style="bold")
        group = Group(general_table, list_table)
        panel = Panel(group, title=titleText)

        return panel
