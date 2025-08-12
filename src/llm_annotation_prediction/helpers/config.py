import logging
import os
from typing import Any

import yaml
from pydantic import BaseModel

from llm_annotation_prediction.helpers.utils import load_class

CONFIG_CLASS_SUFFIX = "Config"
logger = logging.getLogger("Config")


class Config(BaseModel):
    """
    Typed global configuration definition for yaml files. Includes options for
    derived classes, which are dynamically loaded. Each class can have a corresponding
    config class with the same name and a "Config" suffix, that should be in the same
    file as the class.
    """

    class Config:
        extra = "forbid"  # Prohibit unknown config flags

    # The experiment name. Will be appended to folder name.
    name: str = "Default"

    # Don't save any output (logs, results) to disk
    no_save: bool = False

    # Don't write logs to console streams
    silent: bool = False

    log_level: str = "INFO"

    # Top-level configuration elements
    # Due to the dynamic loading, we cannot further define the types here
    experiment: Any
    dataset: Any
    conversation: Any
    publication: Any


def instantiate_dynamic_subconfigs(data: Any) -> Any:
    """
    Recursively walk 'data'. Whenever we find a dict with a 'type' key,
    import the matching Config class and instantiate it (recursively), unless this is
    disabled with the special __ignore_types__ key.
    """
    if isinstance(data, dict):
        # We ignore all types in this subtree. This is needed for JSON schema definitions.
        if "__ignore_types__" in data:
            return data

        if "type" in data:
            # e.g. 'some.module.MyClass' => 'some.module.MyClassConfig'
            config_cls = load_class(data["type"] + CONFIG_CLASS_SUFFIX)
            # Recursively transform all sub-dicts or lists in this dict
            transformed = {
                k: instantiate_dynamic_subconfigs(v) for k, v in data.items()
            }
            # Instantiate the Pydantic config class
            return config_cls(**transformed)
        else:
            # Normal dict => just process its values
            return {k: instantiate_dynamic_subconfigs(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Process each item in the list
        return [instantiate_dynamic_subconfigs(item) for item in data]
    else:
        # Primitive => return as is
        return data


_active_inclusions = set()
_include_cache = {}


def include_external_yaml_object_constructor(
    loader: yaml.SafeLoader, node: yaml.ScalarNode
) -> Any:
    """
    Custom constructor to include objects from other YAML files.

    Files are cached and checked for cyclic dependencies.

    Args:
        loader (SafeLoader): The YAML loader instance.
        node (ScalarNode): The scalar node representing the !include directive.

    Returns:
        Any: The extracted object from the included YAML file.

    Raises:
        ValueError: If a cyclic inclusion is detected.
        KeyError: If the specified key path is not found in the included file.
    """
    # Extract the filename and optional object path from the node value
    value: str = loader.construct_scalar(node)
    parts: list[str] = value.split(":")
    file_name: str = parts[0].strip()
    object_path: list[str] = [part.strip() for part in parts[1:]]

    # Determine the absolute path of the file to be included
    current_file: str = loader.name
    base_dir: str = os.path.dirname(current_file)
    file_path: str = os.path.abspath(os.path.join(base_dir, file_name))

    # Check for cyclic inclusion
    if file_path in _active_inclusions:
        raise ValueError(
            f"Cyclic inclusion detected: '{file_path}' is already being processed."
        )

    _active_inclusions.add(file_path)

    try:
        # Load the included YAML file with caching
        if file_path not in _include_cache:
            with open(file_path, "r") as f:
                included_content: Any = yaml.load(f, Loader=type(loader))
            _include_cache[file_path] = included_content
        else:
            included_content = _include_cache[file_path]

        # Extract the specific object if an object path is provided
        for key in object_path:
            if not isinstance(included_content, dict):
                raise KeyError(
                    f"Cannot extract key '{key}' from non-dictionary content in '{file_path}'."
                )
            included_content = included_content.get(key)
            if included_content is None:
                raise KeyError(f"Key '{key}' not found in '{file_path}'.")

    finally:
        # Remove the file from the set after processing
        _active_inclusions.remove(file_path)

    return included_content


def load_config(filename: str) -> Config:
    """
    Load a yaml configuration file

    Automatically instantiates the typed configuration classes defined in the file.
    """
    logger.info(f"Loading config from {filename}")

    yaml.SafeLoader.add_constructor(
        "!include", include_external_yaml_object_constructor
    )

    with open(filename, "r") as file:
        config_file = yaml.safe_load(file)

    try:
        # Replace all dicts with a `type` property with their configuration
        # counterpart classes
        transformed_config = instantiate_dynamic_subconfigs(config_file)
    except AttributeError as e:
        logger.error(e)
        logger.error(
            'Configuration error: Cannot find configuration class with the "Config" '
            "suffix. Aborting."
        )
        exit(1)

    return Config(**transformed_config)
