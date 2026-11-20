from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser

import numpy as np
import pandas as pd


class ReviewHTMLTextExtractor(HTMLParser):
    """Remove HTML tags while preserving readable text, spacing and entities."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"br", "p", "div", "li", "tr", "td", "hr"}:
            self.parts.append(" ")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag.lower() in {"p", "div", "li", "tr", "td"}:
            self.parts.append(" ")

    def handle_data(self, data):
        self.parts.append(data)

    def clean(self, text: str) -> str:
        self.parts = []
        self.feed(text)
        self.close()
        return re.sub(r"\s+", " ", unescape("".join(self.parts))).strip()


def clean_review_text(value) -> str:
    """Deterministically clean a review title/body without changing the raw source."""
    if not isinstance(value, str) or not value.strip():
        return ""
    extractor = ReviewHTMLTextExtractor()
    try:
        return extractor.clean(value)
    except Exception:
        return re.sub(r"\s+", " ", unescape(value)).strip()


def has_value(value) -> bool:
    """Check whether a scalar/list/dict contains a meaningful value."""
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return bool(pd.notna(value))
    except Exception:
        return False


def count_items(value) -> int:
    """Count list-like items; treat missing and empty collections as zero."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    if isinstance(value, str):
        return int(bool(value.strip()))
    try:
        return 0 if pd.isna(value) else 1
    except Exception:
        return 0


def parse_price(value):
    """Extract the first numeric price value; retain unparseable values as NaN."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value) if np.isfinite(value) else np.nan
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        return float(match.group()) if match else np.nan
    return np.nan


def normalize_bool(value):
    """Normalize common true/false encodings; unknown values remain missing."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return np.nan


def convert_timestamp(series: pd.Series) -> pd.Series:
    """
    Parse epoch seconds, epoch milliseconds or date strings to UTC timestamps.
    The numeric-unit decision follows the Phase 1 rule.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_share = numeric.notna().mean() if len(series) else 0.0
    if numeric_share >= 0.5 and numeric.notna().any():
        median_value = numeric.dropna().median()
        unit = "ms" if abs(median_value) >= 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True)


def clean_review_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Phase 1 deterministic cleaning/type conversion to a joined review table.
    Raw source columns are preserved; cleaned columns are added separately.
    """
    out = df.copy()

    if "sample_row_id" not in out:
        out["sample_row_id"] = np.arange(len(out), dtype=np.int64)

    title_source = out.get("title", pd.Series("", index=out.index))
    body_source = out.get("text", pd.Series("", index=out.index))

    out["review_title_text"] = title_source.fillna("").astype(str)
    out["review_body_text"] = body_source.fillna("").astype(str)
    out["combined_text"] = (
        out["review_title_text"].str.strip() + " " +
        out["review_body_text"].str.strip()
    ).str.strip()

    out["cleaned_title"] = out["review_title_text"].map(clean_review_text)
    out["cleaned_body"] = out["review_body_text"].map(clean_review_text)
    out["cleaned_text"] = (
        out["cleaned_title"] + " " + out["cleaned_body"]
    ).str.strip()

    out["text_title_present"] = out["cleaned_title"].ne("")
    out["text_body_present"] = out["cleaned_body"].ne("")
    out["text_present"] = out["cleaned_text"].ne("")
    out["html_cleaning_changed"] = out["combined_text"].ne(out["cleaned_text"])

    out["helpful_vote_num"] = pd.to_numeric(
        out.get("helpful_vote", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["rating_num"] = pd.to_numeric(
        out.get("rating", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["timestamp_dt"] = convert_timestamp(
        out.get("timestamp", pd.Series(index=out.index, dtype="object"))
    )

    out["product_price_num"] = out.get(
        "product_price", pd.Series(np.nan, index=out.index)
    ).map(parse_price)

    verified_source = out.get(
        "verified_purchase", pd.Series(np.nan, index=out.index)
    ).map(normalize_bool)
    # Preserve Phase 1's true/false/missing coding.
    out["verified_purchase_clean"] = verified_source

    review_images_source = out.get("images", pd.Series(None, index=out.index))
    product_images_source = out.get(
        "product_images", pd.Series(None, index=out.index)
    )
    out["review_images_field_missing"] = review_images_source.isna()
    out["product_images_field_missing"] = product_images_source.isna()
    out["review_images_count"] = review_images_source.map(count_items).astype(int)
    out["product_images_count"] = product_images_source.map(count_items).astype(int)
    out["review_image_present"] = out["review_images_count"].gt(0)
    out["product_image_present"] = out["product_images_count"].gt(0)

    out["product_description_present"] = out.get(
        "product_description", pd.Series(None, index=out.index)
    ).map(has_value)
    if "metadata_matched" not in out:
        out["metadata_matched"] = False
    out["metadata_matched"] = out["metadata_matched"].fillna(False).astype(bool)

    return out
