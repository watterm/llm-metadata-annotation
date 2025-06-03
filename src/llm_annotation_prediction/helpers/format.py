import re

doi_regex = re.compile(r'10.\d{4,9}/[^\s"<>]+')


def format_doi(doi: str) -> str:
    """
    Guarantees a DOI string to be in link-form.
    """
    # Using a regex to extract saves us testing for
    without_prefix = doi_regex.search(doi)
    if not without_prefix:
        raise ValueError(f"No DOI detected in {doi}")
    return f"https://doi.org/{without_prefix.group()}"


def sanitize_folder_name(name: str) -> str:
    """
    Replace any annoying characters for folder names
    """
    # Replace any non-alphanumeric characters (except underscore and hyphen) with underscores
    sanitized_name = re.sub(r"[^\w\-]", "_", name)

    # Trim any leading or trailing underscores
    return sanitized_name.strip("_")
