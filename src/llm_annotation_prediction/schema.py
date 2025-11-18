from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from logging import getLogger
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter
from referencing import Registry, Resource
from referencing._core import Resolved, Resolver

from llm_annotation_prediction.helpers.schema import (
    contains,
    match_gitlab_regex,
    mnemonic,
    normalize,
)
from llm_annotation_prediction.helpers.utils import get

_logger = getLogger("Schema")


class SchemaConfig(BaseModel):
    type: str = "Schema"

    # Path to the folder with the `schema.json`.
    schema_folder: Path | None = Field(default=None)

    # Instead of collecting the entities from the schema files, we use a previously
    # saved list.
    load_collection_from_file: Path | None = Field(default=None)

    # If entities have to be extracted from schema, this will determine if and to what
    # file they are saved, so they can be loaded directly next time.
    save_collection_to_file: Path | None = Field(default=None)

    # Prefix that will be used for all lookups in this schema.
    # Intended to be used for the schema folder, file or a root object.
    base_path: str = ""

    # Extracts these elements from the schema definitions.
    # Each of the elements in this array should point to a composition (anyOf, oneOf,
    # enum) in the schema.
    entity_collection: list[EntityListReference] = Field(
        default_factory=list["EntityListReference"]
    )

    # Whether to print IDs when converting entities to markdown strings.
    print_ids: bool = True


class EntityListReference(BaseModel):
    """
    Describes the list references to extract from a schema file
    """

    name: str
    reference: str
    id_prefix: str = ""  # Optional prefix for generated IDs
    depth: int | None = None


class OntologyLinkedEntity(BaseModel):
    """
    Representation of an extracted entry in a fredato JSON schema.
    """

    key: str | None = None
    display: str
    uri: str | None = None
    children: list[OntologyLinkedEntity] | None = None

    # Optional generated stable mnemonic + hash ID based on hierarchy
    gen_id: str | None = None

    def __str__(self) -> str:
        return self.to_markdown_string()

    def to_markdown_string(self, indent: int = 0, print_ids: bool = True) -> str:
        """
        Converts the entity to a markdown list entry with a given indentation.
        Appends a markdown list of its children with increased indentation.
        """
        text: str = ""
        if print_ids and self.gen_id:
            text = " " * indent + f"- {self.gen_id} | {self.display}"
        elif not print_ids:
            text = " " * indent + f"- {self.display}"
        if self.children:
            for child in self.children:
                text += f"\n{child.to_markdown_string(indent + 2, print_ids=print_ids)}"
        return text

    @staticmethod
    def list_to_markdown_string(
        entity_list: list[OntologyLinkedEntity], print_ids: bool = True
    ) -> str:
        """
        Converts each entity in a list to a markdown list string and combines them
        to form a printable string.
        """
        return "\n".join(
            entity.to_markdown_string(print_ids=print_ids) for entity in entity_list
        )

    def generate_id(
        self,
        parent_mnemonic: str | None = None,
        parent_paths: list[str] | None = None,
        generate_for_children: bool = True,
    ) -> None:
        """
        Generates a stable ID for the entity based on its position in the hierarchy.

        Sets the `gen_id` attribute to a string formatted as:
        - `@{mnemonic}-{hash}` for root entities
        - `@{parent_mnemonic}-{mnemonic}-{hash}` for child entities

        The hash is a 5-character SHA-256 digest of the normalized hierarchical path.

        Parameters
        ----------
        parent_mnemonic: Mnemonic of the parent entity (None for root entities)
        parent_paths: List of display names from root to parent (for path construction)
        generate_for_children: Whether to recursively generate IDs for children
        """
        path: str = ""
        if parent_paths is not None and len(parent_paths) > 0:
            path = " > ".join(normalize(p) for p in parent_paths) + " > "
        path += normalize(self.display)

        # Generate 5-character hash from the path
        hash: str = hashlib.sha256(path.encode("utf-8")).hexdigest()[:5].upper()

        mm: str = mnemonic(self.display)

        self.gen_id = (
            f"@{parent_mnemonic}-{mm}-{hash}" if parent_mnemonic else f"@{mm}-{hash}"
        )

        if generate_for_children and self.children:
            for child in self.children:
                child.generate_id(
                    parent_mnemonic=mm,
                    parent_paths=(parent_paths or []) + [self.display],
                    generate_for_children=True,
                )


class SchemaSerialized(BaseModel):
    """Pydantic representation of a fully built Schema.

    This enables round‑trip JSON (de)serialization. Paths in ``SchemaConfig`` are
    converted to strings when dumping in *json* mode.
    """

    config: SchemaConfig
    entities: dict[str, list[OntologyLinkedEntity]]


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

    def __init__(
        self,
        config: SchemaConfig,
        entity_collection: SchemaEntityCollection | None = None,
    ):
        """Create a Schema.

        Parameters
        ----------
        config:
            Configuration object.
        entity_collection:
            Pre-built entity collection (used when restoring from serialized JSON).
            When provided, no building/loading from files is performed.
        """

        self._config: SchemaConfig = config

        if entity_collection is not None:
            # Direct injection (restored instance)
            self._entity_collection = entity_collection
            return

        if config.load_collection_from_file is not None:
            self._entity_collection = self._load_collection(
                config.load_collection_from_file
            )
            return

        # Build from schema folder
        self._validate_schema_folder(config.schema_folder)
        self._entity_collection = self._build_collection(config.entity_collection)

        if config.save_collection_to_file:
            self._save_collection(
                Path(config.save_collection_to_file), self._entity_collection
            )

    @property
    def collection(self) -> dict[str, str]:
        """
        Returns the collection as a dict of stringified markdown lists.
        """
        return {
            k: OntologyLinkedEntity.list_to_markdown_string(
                v, print_ids=self._config.print_ids
            )
            for k, v in self._entity_collection.items()
        }

    def contains_name(
        self,
        name: str,
        search_keys: bool = True,
        start_from: OntologyLinkedEntity | list[OntologyLinkedEntity] | None = None,
    ) -> bool:
        """Return True if any entity (or optionally its key) fuzzy‑matches ``name``.

        Parameters
        ----------
        name: String to look for (substring matching via ``helpers.schema.contains``).
        search_keys: Include entity.key values and list names when ``start_from`` is None.
        start_from: Limit the search to a subtree (single entity or list of entities).
        """

        def _match_entity(ent: OntologyLinkedEntity) -> bool:
            if contains(name, ent.display):
                return True
            if search_keys and ent.key and contains(name, ent.key):
                return True
            if ent.children and self.contains_name(
                name, search_keys=search_keys, start_from=ent.children
            ):
                return True
            return False

        if start_from is None:
            if search_keys and any(
                contains(name, list_name) for list_name in self._entity_collection
            ):
                return True
            roots = [e for lst in self._entity_collection.values() for e in lst]
        elif isinstance(start_from, list):
            roots = start_from
        else:
            roots = [start_from]

        return any(_match_entity(ent) for ent in roots)

    # ---------------------------------------------------------------------
    # Serialization helpers
    # ---------------------------------------------------------------------
    def _serialized(self) -> SchemaSerialized:
        """Return a Pydantic model representing this schema.

        Paths are preserved in the nested ``SchemaConfig``; converting to JSON uses
        Pydantic's ``mode='json'`` which stringifies Path objects.
        """

        return SchemaSerialized(config=self._config, entities=self._entity_collection)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON‑serializable dict representation."""
        return self._serialized().model_dump(mode="json", exclude_none=True)

    def to_json(self, **json_kwargs: Any) -> str:
        """Return a JSON string representation.

        Parameters
        ----------
        json_kwargs: forwarded to ``json.dumps`` after obtaining the dict.
        """
        import json as _json

        if "indent" not in json_kwargs:
            json_kwargs["indent"] = 2
        return _json.dumps(self.to_dict(), **json_kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Schema:
        """Reconstruct a ``Schema`` from a dict produced by :meth:`to_dict`."""
        serialized = SchemaSerialized.model_validate(data)
        return cls(config=serialized.config, entity_collection=serialized.entities)

    @classmethod
    def from_json(cls, json_str: str) -> Schema:
        """Reconstruct a ``Schema`` from a JSON string."""
        serialized = SchemaSerialized.model_validate_json(json_str)
        return cls(config=serialized.config, entity_collection=serialized.entities)

    @classmethod
    def from_file(cls, path: Path | str) -> Schema:
        """Load a serialized schema from a JSON file produced by :meth:`to_json`."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return cls.from_json(content)

    def _load_collection(self, file_path: Path) -> SchemaEntityCollection:
        """
        Loads and validates the entities from a JSON file.
        """
        _logger.info(f"Loading schema from {file_path}")
        with open(file_path, encoding="utf-8") as file:
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
        with open(full_path, encoding="utf-8") as file:
            content = json.load(file)
        return Resource.from_contents(content)

    def _validate_schema_folder(self, folder: Path | None) -> None:
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
        self, entity_lists: list[EntityListReference]
    ) -> SchemaEntityCollection:
        """
        Builds the configured collection of entity lists from the schema.
        """
        _logger.info("Building all entity lists from schema")
        return {e.name: self._build_entity_list(e) for e in entity_lists}

    def _build_entity_list(
        self, list_ref: EntityListReference
    ) -> list[OntologyLinkedEntity]:
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

        _logger.info("Generating IDs for collection")
        for entity in entity_list:
            entity.generate_id(
                parent_mnemonic=list_ref.id_prefix, generate_for_children=True
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

        registry: Registry[Any] = Registry()
        resolver: Resolver[Any] = registry.resolver()
        resolved: Resolved[Any] = resource.pointer(match.group("object"), resolver)
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
        self, resource: Resource, obj: Any, depth: int | None = None
    ) -> OntologyLinkedEntity | list[OntologyLinkedEntity] | None:
        """
        Recursively extracts ontology-linked entitites while resolving references.
        If depth is specified, the recursive lookup level for entities can be limited.
        This limit concerns the submenus of entities, not the JSON structure.
        """
        _logger.debug(f"Extracting entities in {obj} with depth: {depth}")

        # For lists, we look up every entry
        if isinstance(obj, list):
            return self._extract_list_items(resource, obj, depth)  # pyright: ignore[reportUnknownArgumentType]

        # If we encounter a dictionary, it should be an ontology entity or reference one
        if isinstance(obj, dict):
            return self._extract_from_dict(resource, obj, depth)

        # Allow regular string entries (from enums) as well for now.
        if isinstance(obj, str):
            return OntologyLinkedEntity(display=obj)

        raise ValueError(f"Unexpected content: {obj}")

    def _extract_list_items(
        self, resource: Resource, obj: list[Any], depth: int | None = None
    ) -> list[OntologyLinkedEntity]:
        """
        Extract all entities from a list.
        """
        _logger.debug("Extracting list")

        entities: list[OntologyLinkedEntity] = []
        for item in obj:
            # Some enums contain null values, which we need to ignore
            if item is not None:
                item_entities: (
                    OntologyLinkedEntity | list[OntologyLinkedEntity] | None
                ) = self._extract_entities(resource, item, depth=depth)

                # Some lists might contain null entries or custom entries that return None
                if item_entities is None:
                    continue

                if isinstance(item_entities, list):
                    raise TypeError(f"List of lists not supported in schemas: {item}")
                entities.append(item_entities)

        return entities

    def _extract_from_dict(
        self, resource: Resource, obj: Any, depth: int | None = None
    ) -> OntologyLinkedEntity | list[OntologyLinkedEntity] | None:
        """
        Extract all entities from a dictionary or follow compositions and references to
        find more.
        """
        _logger.debug("Extracting from dictionary")

        # Check if it's a reference object with { "$ref": "..."}
        reference: Reference | None = self._resolve_reference_object(resource, obj)
        if reference is not None:
            return self._extract_entities(
                reference.resource, reference.json_object, depth=depth
            )

        # Multi-Select: If x-enum exists, use it as a list of entities
        x_enum: list[str] | None = get(obj, "x-enum")
        if x_enum is not None:
            return self._extract_entities(resource, x_enum, depth)

        # Try to extract an ontology entity
        entity: OntologyLinkedEntity | None = self._extract_ontology_entity(obj)
        if entity is not None:
            entity.children = self._extract_submenu(resource, obj, depth)
            return entity

        # If this is neither a reference nor a an entity, try if we have a composition
        for property in ["anyOf", "oneOf", "enum"]:
            composition = get(obj, property)
            if composition is not None:
                return self._extract_entities(resource, composition, depth)

        # Check for custom entries in select components
        if self._is_custom_entry(obj):
            return None

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
        self, resource: Resource, obj: Any, depth: int | None = None
    ) -> list[OntologyLinkedEntity] | None:
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

    def _is_custom_entry(self, obj: Any) -> bool:
        """
        The select components have a special entry for custom values. These key-value
        items have no constant display value.
        """
        display: Any | None = get(obj, "properties", "display")
        if display:
            return display.get("const") is None
        return False

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

        submenu: Any | None = get(obj, "properties", key)
        if submenu is not None:
            _logger.debug(f"Found submenu for key {key}")
        else:
            _logger.debug(f"No submenu found for key {key}")
        return submenu

    def _decrease_depth(self, depth: int | None) -> int | None:
        """
        Helper to handle decreasing the depth level in the tree.
        """
        if depth is None:
            return None
        return depth - 1
