"""Type stubs for transliterate package."""

def translit(
    value: str,
    language_code: str,
    reversed: bool = False,
    strict: bool = False,
) -> str:
    """
    Transliterates the given value from one script to another.

    Args:
        value: The string to transliterate
        language_code: The language code (e.g., 'el' for Greek)
        reversed: If True, transliterate from Latin to the target script
        strict: If True, raise an exception if transliteration fails

    Returns:
        The transliterated string
    """
    ...

def slugify(
    value: str,
    language_code: str,
    reversed: bool = False,
    strict: bool = False,
) -> str:
    """
    Create a URL-safe slug from the given value.

    Args:
        value: The string to slugify
        language_code: The language code
        reversed: If True, transliterate from Latin to the target script
        strict: If True, raise an exception if transliteration fails

    Returns:
        A URL-safe slug
    """
    ...

def detect_language(value: str) -> str | None:
    """
    Detect the language of the given value.

    Args:
        value: The string to analyze

    Returns:
        The detected language code, or None if detection fails
    """
    ...

def get_available_language_codes() -> list[str]:
    """Get a list of available language codes."""
    ...

def get_available_language_packs() -> dict[str, str]:
    """Get a dictionary of available language packs."""
    ...
