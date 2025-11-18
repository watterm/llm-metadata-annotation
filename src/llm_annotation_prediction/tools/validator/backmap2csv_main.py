"""Attach researcher validations back onto merged_data_files.csv.

Reads per‑paper sheets from validation_<dataset>/to_validate_for_researchers/
and writes merged_data_files_with_validation.csv to validation_<dataset>/other/.

Defaults:
- --experiments-root saved_experiments
- --dataset nephgen-nine (creates validation_nephgen-nine)

Usage (installed console script):
- uv run backmap --experiments-root saved_experiments --dataset nephgen-nine
- uv run backmap  # uses defaults
"""

import argparse
from collections.abc import Sequence
import csv
from pathlib import Path
import sys
from typing import Any

from pydantic import BaseModel  # type: ignore

DELIM = ";"

from llm_annotation_prediction.helpers.schema import (
    normalize,
)  # canonical normalization only


class BackmapRow(BaseModel):
    """Pydantic model for representing a CSV row in backmap2csv conversion."""

    paper: str = ""
    entity: str = ""
    validation: str = ""

    class Config:
        extra = "allow"

    def csv_record(self) -> dict[str, Any]:
        """Return the row as a dictionary for CSV writing."""
        return self.dict()  # type: ignore


def load_validation_mapping(sheets_dir: Path) -> dict[tuple[str, str], str]:
    """Read per-paper sheets and build (paper, normalized_entity) -> validation map."""
    mapping: dict[tuple[str, str], str] = {}
    if not sheets_dir.exists():
        return mapping
    for p in sorted(sheets_dir.glob("*.csv")):
        if p.name.startswith("merged_data_files"):
            continue  # skip merged files
        paper_key = p.stem
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=DELIM)
            for row in reader:
                model = BackmapRow(
                    paper=paper_key,
                    entity=row.get("entity_name", "").strip(),
                    validation=row.get("validation", "").strip(),
                )
                if not model.entity or not model.validation:
                    continue
                mapping[(paper_key, normalize(model.entity))] = model.validation
    return mapping


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for backmapping validations."""
    p = argparse.ArgumentParser(
        description=(
            "Attach researcher validations to merged_data_files.csv. "
            "Use SAME --experiments-root / --dataset (or --sheets-dir) as json2csv unless relying on auto-detect."
        )
    )
    p.add_argument(
        "--experiments-root",
        default="saved_experiments",
        help="Root experiments directory (match json2csv)",
    )
    p.add_argument(
        "--dataset",
        default="nephgen-nine",
        help="Dataset name (validation_<dataset> folder; match json2csv)",
    )
    p.add_argument(
        "--sheets-dir",
        default=None,
        help="Explicit per-paper sheets dir (skips dataset-based inference)",
    )
    return p


def write_backmapped_csv(
    out_csv: Path, fieldnames: list[str], rows: list[dict[str, Any]]
) -> int:
    """Write rows with validation to CSV and return row count."""
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=DELIM)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: Sequence[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)

    experiments_root = Path(args.experiments_root)
    # Determine primary directories
    if args.sheets_dir:
        sheets_dir = Path(args.sheets_dir)
        validation_root = sheets_dir.parent
    else:
        validation_root = experiments_root / f"validation_{args.dataset}"
        sheets_dir = validation_root / "to_validate_for_researchers"

    merged_dir = validation_root / "other"

    # Expect strict validation_<dataset> layout or explicit --sheets-dir; no legacy or auto-discovery.
    merged_csv_in = merged_dir / "merged_data_files.csv"
    if not merged_csv_in.exists():
        print(
            f"ERROR: Expected merged CSV at {merged_csv_in}. Run json2csv first or pass --sheets-dir."
        )
        sys.exit(1)
    if not sheets_dir.exists():
        print(
            f"ERROR: Sheets directory missing: {sheets_dir}. Run json2csv first or pass --sheets-dir."
        )
        sys.exit(1)

    # Output always written next to source merged file
    out_csv = merged_csv_in.parent / "merged_data_files_with_validation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    valmap = load_validation_mapping(sheets_dir)

    with merged_csv_in.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=DELIM)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        rows = list(reader)
    if "validation" not in fieldnames:
        fieldnames.append("validation")

    # Normalize fields using get with explicit defaults
    total = len(rows)
    matched = 0
    for r in rows:
        paper_key = r.get("paper_key_name", "").strip()
        raw_entity = r.get("entity_name", "").strip()
        norm_entity = (
            r.get("entity_name_normalize", "") or normalize(raw_entity)
        ).strip()
        validation = valmap.get((paper_key, norm_entity), "")
        r["validation"] = validation
        if validation:
            matched += 1

    # Write output using helper
    rows_written = write_backmapped_csv(out_csv, fieldnames, rows)
    print(f"Wrote {rows_written} rows with validation to: {out_csv}")
    print(
        f"Attached validation to {matched}/{total} rows (paper_key_name + normalized entity)."
    )
    print(f"Sheets directory: {sheets_dir}")
    print(f"Merged directory: {merged_csv_in.parent}")


if __name__ == "__main__":
    main()
