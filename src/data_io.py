from __future__ import annotations

import gzip
import json
import random
import re
import warnings
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd


def normalized_file_stem(path: str | Path) -> str:
    """Normalize a JSONL filename for exact category matching."""
    name = Path(path).name.lower()
    for suffix in (".jsonl.gz", ".jsonl"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return re.sub(r"[^a-z0-9]", "", name)


def find_jsonl_file(
    category: str,
    roots: Iterable[str | Path],
    metadata: bool = False,
    manual_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """Find the exact review or meta_<category> JSONL file under supplied roots."""
    if manual_path:
        path = Path(manual_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configured path does not exist: {path}")
        return path

    expected = normalized_file_stem(f"meta_{category}" if metadata else category)
    candidates: list[Path] = []
    for root_value in roots:
        root = Path(root_value).expanduser()
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if lower.endswith(".jsonl") or lower.endswith(".jsonl.gz"):
                if normalized_file_stem(path) == expected:
                    candidates.append(path)

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: (len(p.parts), str(p)))
    if len(candidates) > 1:
        warnings.warn(
            f"Multiple exact matches found for {category} (metadata={metadata}); "
            f"using the first deterministic match: {candidates[0]}"
        )
    return candidates[0]


def open_jsonl(path: str | Path):
    """Open plain or gzip-compressed JSONL as UTF-8 text."""
    path = Path(path)
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8")


def reservoir_jsonl(path: str | Path, k: int, seed: int) -> tuple[list[dict], dict]:
    """
    Uniformly sample up to k valid JSONL records without loading the full file.
    Blank and malformed lines are counted and skipped.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    rng = random.Random(seed)
    sample: list[dict] = []
    valid_count = 0
    blank_count = 0
    bad_count = 0

    with open_jsonl(path) as handle:
        for line in handle:
            if not line.strip():
                blank_count += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_count += 1
                continue
            if not isinstance(row, dict):
                bad_count += 1
                continue

            valid_count += 1
            if len(sample) < k:
                sample.append(row)
            else:
                replacement_index = rng.randrange(valid_count)
                if replacement_index < k:
                    sample[replacement_index] = row

    return sample, {
        "valid_rows_scanned": valid_count,
        "sample_rows": len(sample),
        "blank_lines": blank_count,
        "invalid_json_lines": bad_count,
        "sample_fraction": len(sample) / valid_count if valid_count else np.nan,
    }


def first_nonempty(row: Mapping, keys: Iterable[str]) -> Optional[str]:
    """Return the first nonempty value among candidate keys as a string."""
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            continue
        value_string = str(value).strip()
        if value_string and value_string.lower() != "nan":
            return value_string
    return None


def scan_metadata_for_sample(
    path: str | Path,
    target_keys: set[str],
) -> tuple[list[dict], dict]:
    """Stream a metadata file and retain only products referenced by sampled reviews."""
    records: list[dict] = []
    valid_count = 0
    bad_count = 0
    blank_count = 0

    with open_jsonl(path) as handle:
        for line in handle:
            if not line.strip():
                blank_count += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_count += 1
                continue
            if not isinstance(row, dict):
                bad_count += 1
                continue

            valid_count += 1
            key = first_nonempty(row, ["parent_asin", "asin"])
            if key is not None and key in target_keys:
                records.append(row)

    return records, {
        "valid_rows_scanned": valid_count,
        "matched_metadata_rows_before_deduplication": len(records),
        "blank_lines": blank_count,
        "invalid_json_lines": bad_count,
    }


def build_reviews_dataframe(
    categories: list[str],
    roots: Iterable[str | Path],
    sample_size_per_category: int = 40_000,
    seed: int = 42,
    review_paths: Optional[Mapping[str, Optional[str | Path]]] = None,
    metadata_paths: Optional[Mapping[str, Optional[str | Path]]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Reconstruct the Phase 1 sample from Kaggle JSONL inputs.

    Returns:
        analysis_df: review rows left-joined to deduplicated product metadata.
        file_audit: per-category file and sampling audit.
        file_paths: resolved paths keyed by (category, "reviews"/"metadata").
        join_audit: metadata duplicate and match counts.
    """
    review_paths = review_paths or {}
    metadata_paths = metadata_paths or {}

    review_frames = []
    metadata_frames = []
    file_paths: dict = {}
    audit_rows = []
    metadata_duplicate_rows = 0

    for category in categories:
        review_path = find_jsonl_file(
            category, roots, metadata=False, manual_path=review_paths.get(category)
        )
        if review_path is None:
            raise FileNotFoundError(
                f"Could not find review JSONL for {category} under {list(map(str, roots))}. "
                "Check the Kaggle input attachment or set REVIEW_PATHS manually."
            )

        metadata_path = find_jsonl_file(
            category, roots, metadata=True, manual_path=metadata_paths.get(category)
        )
        file_paths[(category, "reviews")] = review_path
        file_paths[(category, "metadata")] = metadata_path

        review_rows, review_audit = reservoir_jsonl(
            review_path, k=sample_size_per_category, seed=seed
        )
        review_frame = pd.DataFrame(review_rows).copy()
        review_frame["dataset_category"] = category
        review_frame["_join_key"] = review_frame.apply(
            lambda row: first_nonempty(row, ["parent_asin", "asin"]), axis=1
        )
        review_frames.append(review_frame)

        if metadata_path is not None:
            target_keys = {
                first_nonempty(row, ["parent_asin", "asin"]) for row in review_rows
            }
            target_keys.discard(None)
            metadata_rows, metadata_audit = scan_metadata_for_sample(
                metadata_path, target_keys
            )
            metadata_frame = pd.DataFrame(metadata_rows).copy()
            if not metadata_frame.empty:
                metadata_frame["_join_key"] = metadata_frame.apply(
                    lambda row: first_nonempty(row, ["parent_asin", "asin"]), axis=1
                )
                metadata_frame["dataset_category"] = category
                metadata_frames.append(metadata_frame)
        else:
            metadata_audit = {
                "valid_rows_scanned": np.nan,
                "matched_metadata_rows_before_deduplication": 0,
                "blank_lines": np.nan,
                "invalid_json_lines": np.nan,
            }
            warnings.warn(
                f"Product metadata file was not found for {category}; "
                "review-level analysis will continue without joined product metadata."
            )

        audit_rows.append({
            "category": category,
            "review_file": str(review_path),
            "metadata_file": str(metadata_path) if metadata_path else None,
            "valid_review_rows_scanned": review_audit["valid_rows_scanned"],
            "sampled_review_rows": review_audit["sample_rows"],
            "review_sample_fraction": review_audit["sample_fraction"],
            "blank_review_lines": review_audit["blank_lines"],
            "invalid_review_json_lines": review_audit["invalid_json_lines"],
            "valid_metadata_rows_scanned": metadata_audit["valid_rows_scanned"],
            "matched_metadata_rows_before_deduplication":
                metadata_audit["matched_metadata_rows_before_deduplication"],
            "blank_metadata_lines": metadata_audit["blank_lines"],
            "invalid_metadata_json_lines": metadata_audit["invalid_json_lines"],
        })

    reviews = pd.concat(review_frames, ignore_index=True, sort=False)
    if metadata_frames:
        metadata = pd.concat(metadata_frames, ignore_index=True, sort=False)
        metadata_duplicate_rows = int(
            metadata.duplicated(
                subset=["dataset_category", "_join_key"], keep="first"
            ).sum()
        )
        metadata = metadata.drop_duplicates(
            subset=["dataset_category", "_join_key"], keep="first"
        )
        rename_map = {
            col: f"product_{col}"
            for col in metadata.columns
            if col not in {"_join_key", "dataset_category"}
        }
        metadata = metadata.rename(columns=rename_map)
    else:
        metadata = pd.DataFrame(columns=["dataset_category", "_join_key"])

    analysis_df = reviews.merge(
        metadata,
        on=["dataset_category", "_join_key"],
        how="left",
        validate="many_to_one",
        indicator="_metadata_merge",
    )
    analysis_df["metadata_matched"] = analysis_df["_metadata_merge"].eq("both")
    analysis_df = analysis_df.drop(columns="_metadata_merge")
    analysis_df["sample_row_id"] = np.arange(len(analysis_df), dtype=np.int64)

    if "helpful_vote" not in analysis_df.columns:
        warnings.warn("The sampled review records do not contain helpful_vote.")
        analysis_df["helpful_vote"] = np.nan

    analysis_df["helpful_vote_num"] = pd.to_numeric(
        analysis_df["helpful_vote"], errors="coerce"
    )
    analysis_df["rating_num"] = pd.to_numeric(
        analysis_df.get("rating", pd.Series(index=analysis_df.index, dtype="object")),
        errors="coerce",
    )

    file_audit = pd.DataFrame(audit_rows)
    join_audit = {
        "metadata_duplicate_rows_removed": metadata_duplicate_rows,
        "review_rows": int(len(analysis_df)),
        "metadata_matched_rows": int(analysis_df["metadata_matched"].sum()),
        "metadata_match_rate": float(analysis_df["metadata_matched"].mean())
        if len(analysis_df) else np.nan,
    }
    return analysis_df, file_audit, file_paths, join_audit
