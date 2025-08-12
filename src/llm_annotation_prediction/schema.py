import json
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, TypeAdapter
from referencing import Registry, Resource

from llm_annotation_prediction.helpers.schema import match_gitlab_regex
from llm_annotation_prediction.helpers.utils import get

_logger = getLogger("Schema")


class SchemaConfig(BaseModel):
    type: str = "Schema"

    # Path to the folder with the `schema.json`.
    schema_folder: Optional[Path] = Field(None)

    # Instead of collecting the entities from the schema files, we use a previously
    # saved list.
    load_collection_from_file: Optional[Path] = Field(None)

    # If entities have to be extracted from schema, this will determine if and to what
    # file they are saved, so they can be loaded directly next time.
    save_collection_to_file: Optional[Path] = Field(None)

    # Prefix that will be used for all lookups in this schema.
    # Intended to be used for the schema folder, file or a root object.
    base_path: str = ""

    # Extracts these elements from the schema definitions.
    # Each of the elements in this array should point to a composition (anyOf, oneOf,
    # enum) in the schema.
    entity_collection: List["EntityListReference"] = []


class EntityListReference(BaseModel):
    """
    Describes the list references to extract from a schema file
    """

    name: str
    reference: str
    depth: Optional[int] = None


class OntologyLinkedEntity(BaseModel):
    """
    Representation of an extracted entry in a fredato JSON schema.
    """

    key: Optional[str] = None
    display: str
    uri: Optional[str] = None
    children: Optional[List["OntologyLinkedEntity"]] = None

    def __str__(self) -> str:
        return self.to_markdown_string()

    def to_markdown_string(self, indent: int = 0) -> str:
        """
        Converts the entity to a markdown list entry with a given indentation.
        Appends a markdown list of its children with increased indentation.
        """
        text = " " * indent + f"- {self.display}"
        if self.children:
            for child in self.children:
                text += f"\n{child.to_markdown_string(indent + 2)}"
        return text

    @staticmethod
    def list_to_markdown_string(entity_list: list["OntologyLinkedEntity"]) -> str:
        """
        Converts each entity in a list to a markdown list string and combines them
        to form a printable string.
        """
        return "\n".join(entity.to_markdown_string() for entity in entity_list)


@dataclass
class Reference:
    """
    Contains a referenced JSON object and the file it belongs to.
    """

    resource: Resource  # Prevents this from being a pydantic data class
    json_object: Any


# Helper type to describe the extracted lists from the schema
SchemaEntityCollection = dict[str, list[OntologyLinkedEntity]]


class Schema:
    """
    Repesents a collection of ontology-linked entity lists extracted from a fredato
    metadata schema to be used in LLM conversations. The collection can be stored and
    loaded to avoid parsing the schema every time.

    Does not support object lists or GitLab project references currently.
    """

    # Helps to store dicts of schema entity lists in JSON files
    _entity_collection_adapter = TypeAdapter(SchemaEntityCollection)

    def __init__(self, config: SchemaConfig):
        self._config: SchemaConfig = config

        if config.load_collection_from_file is not None:
            self._entity_collection = self._load_collection(
                config.load_collection_from_file
            )

        else:
            self._validate_schema_folder(config.schema_folder)
            self._entity_collection = self._build_collection(config.entity_collection)

            if config.save_collection_to_file:
                self._save_collection(
                    Path(config.save_collection_to_file), self._entity_collection
                )

    @property
    def collection(self) -> Dict[str, str]:
        """
        Returns the collection as a dict of stringified markdown lists.
        """
        return {
            k: OntologyLinkedEntity.list_to_markdown_string(v)
            for k, v in self._entity_collection.items()
        }

    def _load_collection(self, file_path: Path) -> SchemaEntityCollection:
        """
        Loads and validates the entities from a JSON file.
        """
        _logger.info(f"Loading schema from {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        collection = self._entity_collection_adapter.validate_json(content)
        return collection

    def _save_collection(
        self, file_path: Path, entity_collection: SchemaEntityCollection
    ) -> None:
        """
        Save the entities extracted from the schema to a JSON file.
        """
        _logger.info(f"Saving schema to {file_path}")

        content_bytes = Schema._entity_collection_adapter.dump_json(
            entity_collection, indent=4, exclude_none=True
        )
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content_bytes.decode("utf-8"))

    def _load_schema_file(self, file_path: str) -> Resource:
        """
        Loads a JSON schema file relative to the schema folder.
        """
        if self._config.schema_folder is None:
            raise ValueError(
                "If entities are not loaded from a file, a schema folder needs to be configured."
            )

        full_path = self._config.schema_folder / Path(file_path)
        with open(full_path, "r", encoding="utf-8") as file:
            content = json.load(file)
        return Resource.from_contents(content)

    def _validate_schema_folder(self, folder: Optional[Path]) -> None:
        """
        Make sure that the specified folder contains a fredato metadata schema.
        """
        if folder is None:
            raise ValueError(
                "If entities are not loaded from a file, a schema folder needs to be configured."
            )

        if not folder.exists():
            raise FileNotFoundError(f"Schema folder not found: {folder}")

        schema_path = folder / "schema.json"

        if not schema_path.exists():
            raise FileNotFoundError(f"File 'schema.json' not found in {folder}")

    def _build_collection(
        self, entity_lists: List[EntityListReference]
    ) -> SchemaEntityCollection:
        """
        Builds the configured collection of entity lists from the schema.
        """
        _logger.info("Building all entity lists from schema")
        return {e.name: self._build_entity_list(e) for e in entity_lists}

    def _build_entity_list(
        self, list_ref: EntityListReference
    ) -> List[OntologyLinkedEntity]:
        """
        Builds one entity list from the schema.
        """
        if "#/" not in list_ref.reference:
            raise ValueError(
                f"Schema pointer missing object reference ('#/'): {list_ref.reference}"
            )

        reference = self._resolve_reference_string(None, list_ref.reference)
        entity_list = self._extract_entities(
            reference.resource, reference.json_object, depth=list_ref.depth
        )

        if not isinstance(entity_list, list):
            raise TypeError(
                f"Expected a list of entities, but got {entity_list} instead."
            )

        return entity_list

    def _resolve_reference_string(
        self, resource: Resource | None, target: str
    ) -> Reference:
        """
        Resolves a reference and loads the new JSON file, if specified.
        """
        match = match_gitlab_regex(target)

        file_path = match.group("file")
        if file_path:
            resource = self._load_schema_file(file_path)

        if resource is None:
            raise TypeError(f"No resource found or provided for reference {target}")

        resolved = resource.pointer(match.group("object"), Registry().resolver())
        return Reference(resource, resolved.contents)

    def _resolve_reference_object(
        self, resource: Resource, obj: Any
    ) -> Reference | None:
        """
        Resolves JSON reference objects: { "$ref": "target" }.
        """
        _logger.debug(f"Trying to resolve reference in {obj}")

        reference = get(obj, "$ref")
        if reference is None:
            _logger.debug("No reference found")
            return None

        return self._resolve_reference_string(resource, reference)

    def _extract_entities(
        self, resource: Resource, obj: Any, depth: Optional[int] = None
    ) -> OntologyLinkedEntity | List[OntologyLinkedEntity]:
        """
        Recursively extracts ontology-linked entitites while resolving references.
        If depth is specified, the recursive lookup level for entities can be limited.
        This limit concerns the submenus of entities, not the JSON structure.
        """
        _logger.debug(f"Extracting entities in {obj} with depth: {depth}")

        # For lists, we look up every entry
        if isinstance(obj, list):
            return self._extract_list_items(resource, obj, depth)

        # If we encounter a dictionary, it should be an ontology entity or reference one
        if isinstance(obj, dict):
            return self._extract_from_dict(resource, obj, depth)

        # Allow regular string entries (from enums) as well for now.
        if isinstance(obj, str):
            return OntologyLinkedEntity(display=obj)

        raise ValueError(f"Unexpected content: {obj}")

    def _extract_list_items(
        self, resource: Resource, obj: List[Any], depth: Optional[int] = None
    ) -> List[OntologyLinkedEntity]:
        """
        Extract all entities from a list.
        """
        _logger.debug("Extracting list")

        entities = []
        for item in obj:
            # Some enums contain null values, which we need to ignore
            if item is not None:
                item_entities = self._extract_entities(resource, item, depth=depth)
                if isinstance(item_entities, list):
                    raise TypeError(f"List of lists not supported in schemas: {item}")
                entities.append(item_entities)

        return entities

    def _extract_from_dict(
        self, resource: Resource, obj: Any, depth: Optional[int] = None
    ) -> OntologyLinkedEntity | List[OntologyLinkedEntity]:
        """
        Extract all entities from a dictionary or follow compositions and references to
        find more.
        """
        _logger.debug("Extracting from dictionary")

        # Check if it's a reference object with { "$ref": "..."}
        reference = self._resolve_reference_object(resource, obj)
        if reference is not None:
            return self._extract_entities(
                reference.resource, reference.json_object, depth=depth
            )

        # Try to extract an ontology entity
        entity = self._extract_ontology_entity(obj)
        if entity is not None:
            entity.children = self._extract_submenu(resource, obj, depth)
            return entity

        # If this is neither a reference nor a an entity, try if we have a composition
        for property in ["anyOf", "oneOf", "enum"]:
            composition = get(obj, property)
            if composition is not None:
                return self._extract_entities(resource, composition, depth)

        # Last possibility is the now deprecated select-or-other schem with allOf
        allOf = get(obj, "allOf")
        if allOf is not None:
            select = get(allOf[0], "properties", "selected")
            if select is not None:
                return self._extract_entities(resource, select, depth)

        raise ValueError(f"Expected ontology entity. Found {obj}")

    def _extract_ontology_entity(self, obj: Any) -> OntologyLinkedEntity | None:
        """
        Extract an ontology entry, if the passed object is one. Otherwise return None.

        Does not handle its children.
        """
        _logger.debug(f"Extracting entity in {obj}")

        # If we can't find the key or display, it's not an ontology entry and we can quit
        key = get(obj, "properties", "key", "const")
        display = get(obj, "properties", "display", "const")

        if key is None or display is None:
            _logger.debug("No entity found")
            return None

        # Unfortunately there seems to be an inconsistency with the spelling of classUri
        if not (uri := get(obj, "properties", "classURI", "const")):
            uri = get(obj, "properties", "classUri", "const")

        return OntologyLinkedEntity(key=key, display=display, uri=uri)

    def _extract_submenu(
        self, resource: Resource, obj: Any, depth: Optional[int] = None
    ) -> List[OntologyLinkedEntity] | None:
        """
        Extracts the submenu of an entity, if the depth level allows it.
        """
        # Consider the depth-limit here before checking for the submenu
        if depth is None or depth > 0:
            submenu = self._get_submenu(obj)
            if submenu is not None:
                entities = self._extract_entities(
                    resource,
                    submenu,
                    depth=self._decrease_depth(depth),
                )

                if not isinstance(entities, list):
                    raise TypeError(f"Expected a list of entities, but got {entities}")
                return entities
        return None

    def _get_submenu(self, obj: Any) -> Any | None:
        """
        If an ontology entity has a submenu, its JSON representation will be returned.
        None otherwise.
        """
        _logger.debug(f"Looking for submenu in {obj}")

        key = get(obj, "properties", "key", "const")
        if key is None:
            _logger.error(f"Object is not an ontology entity: {obj}")
            return None

        return get(obj, "properties", key)

    def _decrease_depth(self, depth: Optional[int]) -> int | None:
        """
        Helper to handle decreasing the depth level in the tree.
        """
        if depth is None:
            return None
        return depth - 1
