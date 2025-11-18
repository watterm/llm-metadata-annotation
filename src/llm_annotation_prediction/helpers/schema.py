# This regex is a simplification of references used in fredato schemas. They can point
# to GitLab projects, which we don't support here. We just need to make sure, that all
# references are within the same project.
# Example: gitlab://?excel2schema/organs.json#/kidney/tissue/kidneyTissueList"
import re
import unicodedata

from transliterate import translit

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


_sep_re = re.compile(r"[\s_\-./]+")
_punct_re = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize(s: str) -> str:
    """
    Normalize a string for comparison:
    - Unicode normalization (NFKD)
    - Transliteration from Greek to Latin characters
    - Remove diacritics
    - Case folding (lowercasing, etc.)
    - Replace separators (whitespace, underscores, hyphens, dots, slashes) with a single space
    - Remove punctuation
    """
    s = unicodedata.normalize("NFKD", s)
    s = translit(s, "el", reversed=True)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold()
    s = _sep_re.sub(" ", s)
    s = _punct_re.sub(" ", s)
    return " ".join(s.split())


def contains(query: str, target: str) -> bool:
    q = normalize(query)
    t = normalize(target)
    return q in t


def mnemonic(label: str) -> str:
    """
    Generate a mnemonic from a label (up to 3 letters).
    - If multiple words: use first letter of each word (max 3)
    - If single word: consonants first, then other letters (max 3)
    """
    words: list[str] = normalize(label).split()

    if len(words) >= 2:
        # Multiple words: take first letter of each word, max 3
        mm: str = "".join(word[0] for word in words[:3])
    elif len(words) == 1:
        # Single word: consonants first, then letters, max 3
        cons: str = "".join(
            [c for c in words[0] if c.lower() in "bcdfghjklmnpqrstvwxyz"]
        )
        mm = cons[:3] or words[0][:3]
    else:
        mm = "XX"

    return mm.upper()
