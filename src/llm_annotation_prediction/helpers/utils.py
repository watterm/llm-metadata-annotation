import importlib
from typing import Any, Type, Union


def load_class(class_path: str) -> Type[Any]:
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
def get(obj: Union[dict[str, Any], object], *keys: str) -> Any:
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
    for key in keys:
        if isinstance(obj, dict):
            # Handle dictionary access
            obj = obj.get(key, None)
        else:
            # Handle object attribute access
            obj = getattr(obj, key, None)

        if obj is None:
            break
    return obj


def set_if_none(obj: Union[dict[str, Any], object], key: str, value: Any) -> None:
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
        if obj.get(key, None) is None:
            obj[key] = value
    else:
        if getattr(obj, key, None) is None:
            setattr(obj, key, value)
