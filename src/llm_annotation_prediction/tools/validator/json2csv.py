"""Aggregate experiment data.json files and emit:

1. merged_data_files.csv to validation_<dataset>/other/
2. Per-paper CSV sheets (entity_name; schema_category; validation) to validation_<dataset>/to_validate_for_researchers/

Only consolidated_list entities are considered; backmapping is handled by backmap2csv_main.py.

Defaults:
- --experiments-root saved_experiments
- --dataset nephgen-nine (creates validation_nephgen-nine)

Usage (installed console script):
- uv run json2csv --experiments-root saved_experiments --dataset nephgen-nine
- uv run json2csv  # uses defaults
"""

import argparse
from collections.abc import Iterable, Sequence
import csv
import json
from pathlib import Path
import sys
from typing import Any

from pydantic import BaseModel  # type: ignore[reportMissingImports]

from llm_annotation_prediction.helpers.schema import (
    normalize,
)  # rely on canonical normalization

DELIM = ";"  # CSV delimiter used throughout


class Row(BaseModel):  # type: ignore[reportGeneralTypeIssues]
    """Row of merged entities and per‑paper sheets."""

    entity_name: str
    entity_name_normalize: str
    paper_key_name: str
    schema_category: str | None = None
    llm_name: str

    def merged_record(self) -> dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "entity_name_normalize": self.entity_name_normalize,
            "paper_key_name": self.paper_key_name,
            "schema_category": self.schema_category,
            "llm_name": self.llm_name,
        }

    def sheet_record(self) -> dict[str, str]:
        record: dict[str, Any] = self.dict()  # type: ignore
        return {
            "entity_name": self.entity_name,
            "schema_category": record.get("schema_category", ""),  # type: ignore
            "validation": record.get("validation", ""),  # type: ignore
        }


def iter_data_jsons(dataset_root: Path) -> Iterable[Path]:
    if not dataset_root.exists():
        return
    yield from sorted(dataset_root.glob("**/data.json"))


def extract_rows_from_data_json(data_json_path: Path) -> list[Row]:
    llm_name = data_json_path.parent.name
    with data_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows: list[Row] = []
    for paper_key, paper_entries in data.items():
        for entry in paper_entries:
            consolidated = entry.get("consolidated_list", {})
            for ent in consolidated.get("entity_list", []):
                entity_name = ent.get("entity_name")
                if not entity_name:
                    continue
                rows.append(
                    Row(
                        entity_name=entity_name,
                        entity_name_normalize=normalize(entity_name),
                        paper_key_name=paper_key,
                        schema_category=ent.get("schema_category"),
                        llm_name=llm_name,
                    )
                )
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate per-paper validation sheets and a merged entities CSV. Layout: "
            "validation_<dataset>/to_validate_for_researchers (CSVs) and validation_<dataset>/other (merged). "
            "If you override --experiments-root / --dataset / --sheets-dir you must pass the SAME values to 'backmap'."
        )
    )
    p.add_argument(
        "--experiments-root",
        default="saved_experiments",
        help="Root experiments directory (pass same to backmap)",
    )
    p.add_argument(
        "--dataset",
        default="nephgen-nine",
        help="Dataset name (validation_<dataset> folder; pass same to backmap)",
    )
    p.add_argument(
        "--sheets-dir",
        default=None,
        help="Custom per-paper sheets dir; merged goes to sibling 'other' (backmap use same)",
    )
    p.add_argument(
        "--json-sheets",
        action="store_true",
        help="Also write per-paper sheets as JSON into a 'json' subfolder",
    )
    return p


def dedupe_rows_for_sheets(rows: list[Row]) -> dict[str, dict[str, Row]]:
    """Remove duplicate rows for each paper's sheet.

    This function groups all the rows by paper (using the paper key) and then deduplicates them based on the normalized entity names.
    In cases where multiple rows share the same normalized name, it prefers a row with a schema_category if available.
    The result is a dictionary mapping each paper key to another dictionary that maps normalized entity names to the unique Row object.
    """
    grouped: dict[str, dict[str, Row]] = {}
    for r in rows:
        paper = r.paper_key_name or ""
        if not paper:
            continue
        norm = r.entity_name_normalize or normalize(r.entity_name)
        bucket = grouped.setdefault(paper, {})
        existing = bucket.get(norm)
        if existing is None or (not existing.schema_category and r.schema_category):
            bucket[norm] = r
    return grouped


def write_per_paper_sheets(grouped: dict[str, dict[str, Row]], sheets_dir: Path) -> int:
    """Write CSV sheets for each paper.

    Given an output directory and a dictionary mapping paper identifiers to
    CSV-friendly row dictionaries, this function generates individual CSV files
    for each paper.
    """
    sheets_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for paper_key, mapping in grouped.items():
        out_path = sheets_dir / f"{paper_key}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["entity_name", "schema_category", "validation"],
                delimiter=DELIM,
            )
            writer.writeheader()
            # Use dot notation and the sheet_record() method for consistency
            for row in sorted(mapping.values(), key=lambda x: x.entity_name.casefold()):
                writer.writerow(row.sheet_record())
        count += 1
    return count


def write_merged_entities_csv(rows: list[Row], out_path: Path) -> int:
    """Write merged_data_files.csv with all rows."""
    fieldnames = [
        "entity_name",
        "entity_name_normalize",
        "paper_key_name",
        "schema_category",
        "llm_name",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=DELIM)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.merged_record())
    return len(rows)


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)
    experiments_root = Path(args.experiments_root)
    dataset_root = experiments_root / args.dataset

    data_jsons = [p for p in iter_data_jsons(dataset_root) if p.is_file()]
    if not data_jsons:
        print(f"No data.json files found under: {dataset_root}")
        sys.exit(0)

    all_rows: list[Row] = []
    for pth in data_jsons:
        try:
            rows = extract_rows_from_data_json(pth)
            all_rows.extend(rows)
            print(f"Parsed {pth} -> {len(rows)} rows")
        except Exception as e:  # pragma: no cover - robust to partial failures
            print(f"ERROR parsing {pth}: {e}")

    # Dataset-specific validation directory: validation_<dataset>
    validation_root = experiments_root / f"validation_{args.dataset}"

    # Primary directories (current layout design)
    sheets_dir_default = validation_root / "to_validate_for_researchers"
    merged_dir_default = validation_root / "other"

    # Override semantics: --sheets-dir points to per-paper sheets directory; merged goes to sibling 'other'
    if args.sheets_dir:
        sheets_dir = Path(args.sheets_dir)
        merged_dir = sheets_dir.parent / "other"
    else:
        sheets_dir = sheets_dir_default
        merged_dir = merged_dir_default

    # No legacy layout detection; expect explicit validation_<dataset> structure or --sheets-dir.

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_out = merged_dir / "merged_data_files.csv"
    rows_written = write_merged_entities_csv(all_rows, merged_out)
    print(f"Merged CSV written: {merged_out} ({rows_written} rows)")
    print(f"Per-paper sheets directory: {sheets_dir}")
    if merged_dir != sheets_dir:
        print(f"Merged directory: {merged_dir}")

    grouped = dedupe_rows_for_sheets(all_rows)
    sheet_count = write_per_paper_sheets(grouped, sheets_dir)
    print(f"Per-paper sheets written: {sheet_count} in {sheets_dir}")
    print(f"Dataset root: {dataset_root}")

    # Optional JSON export of per-paper sheets
    if args.json_sheets:
        json_dir = sheets_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        exported = 0
        for paper_key, mapping in grouped.items():
            json_path = json_dir / f"{paper_key}.json"
            records: list[dict[str, str]] = []
            for row in sorted(mapping.values(), key=lambda x: x.entity_name.casefold()):
                records.append(row.sheet_record())
            try:
                with json_path.open("w", encoding="utf-8") as jf:
                    json.dump(records, jf, ensure_ascii=False, indent=2)
                exported += 1
            except Exception as e:  # pragma: no cover
                print(f"ERROR writing JSON sheet {json_path}: {e}")
        print(f"Per-paper JSON sheets written: {exported} in {json_dir}")


if __name__ == "__main__":
    main()
