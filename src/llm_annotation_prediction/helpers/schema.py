# This regex is a simplification of references used in fredato schemas. They can point
# to GitLab projects, which we don't support here. We just need to make sure, that all
# references are within the same project.
# Example: gitlab://?excel2schema/organs.json#/kidney/tissue/kidneyTissueList"
import re

_gitlab_ref_regex = re.compile(
    r"""
        (?:gitlab://    # Our config references don't have to start with the prefix.
            (?P<gitlab> # The gitlab group matches anything up to "?"". It's irrelevant
                [^?]*   # here, we just need it to be empty.
            )
            \?
        )?
        (?P<file>       # The file group matches a file path up to the "#". Optional.
            [^#]*
        )?
        \#              # The references require "#" to be valid.
        (?P<object>     # The remaining part describes the object path in the JSON file.
        .*
        )
    """,
    re.VERBOSE,
)


def match_gitlab_regex(target: str) -> re.Match[str]:
    """
    Parses and validates a reference string
    """
    match = _gitlab_ref_regex.search(target)

    if not match:
        raise ValueError(f"Invalid reference: {target}")

    if match.group("gitlab"):
        raise ValueError(f"Reference links a GitLab project: {target}")

    if not match.group("object"):
        raise ValueError(f"Reference without object path: {target}")

    return match
