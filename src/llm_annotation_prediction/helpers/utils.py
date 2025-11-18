import importlib
from typing import Any, cast


def load_class(class_path: str) -> type[Any]:
    """
    Dynamically loads a class from a given class path string.

    This function takes a fully qualified class path and returns the corresponding class object.
    The class path should be relative to the top-level package.

    Args:
        class_path (str): A string representing the full path to the class,
                          using dot notation (e.g., "subpackage.module.ClassName")

    Returns:
        Type[Any]: The loaded class object

    Raises:
        ValueError: If the package name is not available
        ImportError: If the module or class cannot be imported
        AttributeError: If the specified class does not exist in the module
    """
    if not __package__:
        raise ValueError("Package name not available. Cannot load classes dynamically")
    module_name, class_name = class_path.rsplit(".", 1)
    top_level_package = __package__.split(".")[0]

    module = importlib.import_module(f"{top_level_package}.{module_name}")
    loadedClass = getattr(module, class_name)
    if not isinstance(loadedClass, type):
        raise TypeError(f"{class_name} is not a class")
    return loadedClass


# An optional chaining operator for objects/dicts. Ridiculous that Python does not have this
def get(obj: dict[str, Any] | object, *keys: str) -> Any:
    """
    Safely retrieve nested attributes from objects or nested keys from dictionaries.

    This function allows for safe access to deeply nested attributes in an object or
    keys in a dictionary without raising `AttributeError` or `KeyError`. If any of
    the intermediate attributes or keys is `None` or missing, the function will return `None`.

    Parameters:
    -----------
    obj : object or dict
        The object or dictionary to retrieve values from.
    *keys : str
        A sequence of keys (for dictionaries) or attribute names (for objects) to be
        accessed in order.

    Returns:
    --------
    Any
        The value associated with the final key or attribute if all keys or attributes are found;
        otherwise, `None` if any key or attribute is missing or evaluates to `None` at any level.
    """
    val: Any = obj
    for key in keys:
        if isinstance(val, dict):
            # Handle dictionary access
            val = cast(dict[str, Any], val).get(key, None)
        else:
            # Handle object attribute access
            val = getattr(val, key, None)

        if val is None:
            break
    return val


def set_if_none(obj: dict[str, Any] | object, key: str, value: Any) -> None:
    """
    Sets a given attribute or dictionary key to the specified value if it is currently None.

    Parameters:
    obj: Union[dict, object]
        The object or dictionary in which to set the attribute or key.
    key: str
        The attribute name or dictionary key to check.
    value: Any
        The value to set if the attribute or key is currently None.

    Returns:
    None
    """
    if isinstance(obj, dict):
        if cast(dict[str, Any], obj).get(key, None) is None:
            obj[key] = value
    elif getattr(obj, key, None) is None:
        setattr(obj, key, value)


def extract_values(
    obj: dict[str, Any] | object, path: str, flatten: bool = True
) -> list[Any]:
    """
    Extract values from nested structures using dot notation paths.

    Supports:
    - Simple paths: "key1.key2.key3"
    - Array traversal: automatically extracts from all array elements
    - Multiple values: returns all matching values as a list

    Parameters:
    -----------
    obj : dict or object
        The object or dictionary to extract values from.
    path : str
        Dot-separated path to the desired values (e.g., "data.items.url").
    flatten : bool
        If True, flattens nested lists into a single list. Default: True.

    Returns:
    --------
    list[Any]
        List of all values found at the specified path. Empty list if path not found.

    Examples:
    ---------
    >>> data = {"items": [{"url": "a.com"}, {"url": "b.com"}]}
    >>> extract_values(data, "items.url")
    ["a.com", "b.com"]

    >>> data = {"response": {"data": {"link": "test.com"}}}
    >>> extract_values(data, "response.data.link")
    ["test.com"]
    """
    path_parts = path.split(".")
    results = _extract_from_path(obj, path_parts)

    if flatten:
        return _flatten_list(results)
    return [results] if results else []


def _extract_from_path(obj: Any, path_parts: list[str]) -> Any:
    """
    Recursively extract values from nested structure following path parts.
    Handles arrays by extracting from all elements.
    """
    if not path_parts:
        return obj

    current_key = path_parts[0]
    remaining_parts = path_parts[1:]

    # Use the existing get() helper for a single level
    value = get(obj, current_key)

    if value is None:
        return None

    # If value is a list, extract from all elements
    if isinstance(value, list):
        results: list[Any] = []
        for item in value:
            extracted = _extract_from_path(item, remaining_parts)
            if extracted is not None:
                results.append(extracted)
        return results if results else None

    # Continue with remaining path
    if remaining_parts:
        return _extract_from_path(value, remaining_parts)

    return value


def _flatten_list(obj: Any) -> list[Any]:
    """
    Flatten nested lists into a single list.
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        result: list[Any] = []
        for item in obj:
            result.extend(_flatten_list(item))
        return result
    return [obj]
