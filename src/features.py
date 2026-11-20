from __future__ import annotations

import re
import numpy as np
import pandas as pd

from .cleaning import has_value, parse_price


NUMERIC_FEATURES = [
    "rating_num",
    "product_price_num",
    "text_char_count",
    "text_word_count",
    "text_sentence_count",
    "non_ascii_ratio",
    "exclamation_count",
    "question_count",
    "uppercase_ratio",
    "review_images_count",
    "product_images_count",
    "review_year",
    "review_month_sin",
    "review_month_cos",
    "review_weekday_sin",
    "review_weekday_cos",
    "review_hour_sin",
    "review_hour_cos",
    "reviewer_prior_review_count",
    "product_prior_review_count",
    "user_product_prior_interaction_count",
    "log1p_text_word_count",
    "log1p_product_price",
    "log1p_reviewer_prior_review_count",
    "log1p_product_prior_review_count",
    "log1p_user_product_prior_interaction_count",
    "title_body_length_ratio",
    "metadata_completeness_score",
]

CATEGORICAL_FEATURES = [
    "dataset_category",
    "product_main_category",
    "verified_purchase_status",
    "review_text_available",
    "review_image_available",
    "product_image_available",
    "product_description_available",
    "product_metadata_available",
    "product_price_missing",
]


FEATURE_REGISTRY = [
    ("text_char_count", "Number of characters in cleaned title + body", "Captures review length beyond word count."),
    ("text_word_count", "Number of word-like tokens in cleaned title + body", "A basic measure of review detail."),
    ("text_sentence_count", "Number of sentence-like segments", "Captures writing structure."),
    ("non_ascii_ratio", "Fraction of cleaned characters with code point > 127", "Captures character composition; not a language label."),
    ("exclamation_count", "Number of exclamation marks", "Captures emphatic punctuation."),
    ("question_count", "Number of question marks", "Captures interrogative style."),
    ("uppercase_ratio", "Uppercase alphabetic characters / alphabetic characters", "Captures emphasis in writing."),
    ("title_body_length_ratio", "Cleaned title length / (cleaned body length + 1)", "Captures how much information is concentrated in the title."),
    ("review_age_days_proxy", "Days before the latest timestamp observed in this sample", "A relative recency/exposure proxy, not verified time since publication at the dataset snapshot."),
    ("review_month_sin/cos", "Cyclical encoding of publication month", "Preserves the circular relationship between December and January."),
    ("review_weekday_sin/cos", "Cyclical encoding of weekday", "Represents weekly publication patterns."),
    ("review_hour_sin/cos", "Cyclical encoding of publication hour in UTC", "Represents daily publication patterns."),
    ("reviewer_prior_review_count", "Number of earlier sampled reviews by the same reviewer", "A past-only reviewer activity proxy; raw reviewer ID is not a model input."),
    ("product_prior_review_count", "Number of earlier sampled reviews for the same product", "A past-only product activity proxy; raw product ID is not a model input."),
    ("user_product_prior_interaction_count", "Number of earlier sampled reviews for the same reviewer-product pair", "A past-only repeated-interaction proxy."),
    ("log1p_*", "Log1p transforms of nonnegative, right-skewed counts/price/length", "Reduces the influence of extreme magnitudes without deleting valid observations."),
    ("metadata_completeness_score", "Count of selected product metadata fields available", "Summarizes product metadata coverage."),
    ("product_main_category", "First available product taxonomy label", "Provides a coarse product-type descriptor."),
    ("verified_purchase_status", "Verified / not verified / missing", "Preserves the distinction between false and unrecorded."),
    ("availability indicators", "Availability of review text, images, description and joined metadata", "Makes missingness/availability informative rather than silently hiding it."),
]


def _to_yes_no(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool).map({True: "yes", False: "no"})


def _primary_category(value):
    """Extract a coarse first taxonomy label from common nested category formats."""
    if not has_value(value):
        return np.nan
    current = value
    while isinstance(current, (list, tuple)) and len(current):
        current = current[0]
    if isinstance(current, dict):
        current = current.get("name", current.get("title", np.nan))
    if current is None:
        return np.nan
    result = str(current).strip()
    return result if result else np.nan


def _prior_count_for_groups(
    df: pd.DataFrame,
    group_columns: list[str],
    timestamp_column: str = "timestamp_dt",
) -> pd.Series:
    """
    Count records with strictly earlier timestamps for each group.
    Reviews sharing the same timestamp do not count one another.
    Counts are based only on the sampled review rows.
    """
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    if not all(col in df.columns for col in group_columns) or timestamp_column not in df:
        return result

    work = df[group_columns + [timestamp_column]].copy()
    work["_row_index"] = df.index
    work["_ts"] = pd.to_datetime(work[timestamp_column], errors="coerce", utc=True)

    valid = work["_ts"].notna()
    for col in group_columns:
        valid &= work[col].map(has_value)

    if not valid.any():
        return result

    valid_work = work.loc[valid].copy()
    for col in group_columns:
        valid_work[col] = valid_work[col].astype(str)

    counts = (
        valid_work.groupby(group_columns + ["_ts"], dropna=False)
        .size()
        .rename("n_at_timestamp")
        .reset_index()
        .sort_values(group_columns + ["_ts"])
    )
    counts["prior_count"] = (
        counts.groupby(group_columns, dropna=False)["n_at_timestamp"].cumsum()
        - counts["n_at_timestamp"]
    )

    mapped = valid_work[["_row_index"] + group_columns + ["_ts"]].merge(
        counts[group_columns + ["_ts", "prior_count"]],
        on=group_columns + ["_ts"],
        how="left",
        validate="many_to_one",
    )
    result.loc[mapped["_row_index"].to_numpy()] = mapped["prior_count"].to_numpy()
    return result


def engineer_review_features(
    df: pd.DataFrame,
    reference_timestamp=None,
) -> pd.DataFrame:
    """
    Create deterministic, target-free review features.

    helpful_vote_num is preserved for post-hoc cluster interpretation only and is
    never used to construct any feature in this function.
    """
    out = df.copy()

    if "cleaned_text" not in out:
        out["cleaned_text"] = ""
    out["cleaned_text"] = out["cleaned_text"].fillna("").astype(str)

    if "cleaned_title" not in out:
        out["cleaned_title"] = ""
    if "cleaned_body" not in out:
        out["cleaned_body"] = ""

    text = out["cleaned_text"]
    title = out["cleaned_title"].fillna("").astype(str)
    body = out["cleaned_body"].fillna("").astype(str)

    out["text_char_count"] = text.str.len().astype(float)
    out["text_word_count"] = text.str.findall(r"\b\w+\b").str.len().astype(float)
    out["text_sentence_count"] = text.apply(
        lambda value: float(len([part for part in re.split(r"[.!?]+", value) if part.strip()]))
        if value else 0.0
    )
    out["non_ascii_ratio"] = text.apply(
        lambda value: sum(ord(char) > 127 for char in value) / len(value)
        if value else 0.0
    )
    out["exclamation_count"] = text.str.count("!").astype(float)
    out["question_count"] = text.str.count(r"\?").astype(float)
    out["uppercase_ratio"] = text.apply(
        lambda value: sum(char.isupper() for char in value) /
        sum(char.isalpha() for char in value)
        if any(char.isalpha() for char in value) else 0.0
    )
    out["title_body_length_ratio"] = title.str.len() / (body.str.len() + 1.0)

    if "timestamp_dt" not in out:
        out["timestamp_dt"] = pd.NaT
    timestamps = pd.to_datetime(out["timestamp_dt"], errors="coerce", utc=True)

    if reference_timestamp is None:
        reference = timestamps.max()
        reference_label = "latest timestamp in sampled data (proxy)"
    else:
        reference = pd.to_datetime(reference_timestamp, utc=True)
        reference_label = "user-specified reference timestamp"

    if pd.isna(reference):
        out["review_age_days_proxy"] = np.nan
    else:
        out["review_age_days_proxy"] = (
            (reference - timestamps).dt.total_seconds() / 86400.0
        )
    out.attrs["review_age_reference"] = reference.isoformat() if pd.notna(reference) else None
    out.attrs["review_age_reference_label"] = reference_label

    out["review_year"] = timestamps.dt.year.astype(float)
    month = timestamps.dt.month.astype(float)
    weekday = timestamps.dt.dayofweek.astype(float)
    hour = timestamps.dt.hour.astype(float)

    out["review_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["review_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    out["review_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["review_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    out["review_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["review_hour_cos"] = np.cos(2 * np.pi * hour / 24)

    if "product_price_num" not in out:
        out["product_price_num"] = out.get(
            "product_price", pd.Series(np.nan, index=out.index)
        ).map(parse_price)
    out["product_price_num"] = pd.to_numeric(
        out["product_price_num"], errors="coerce"
    )
    out.loc[out["product_price_num"] < 0, "product_price_num"] = np.nan

    if "review_images_count" not in out:
        out["review_images_count"] = 0
    if "product_images_count" not in out:
        out["product_images_count"] = 0
    out["review_images_count"] = pd.to_numeric(
        out["review_images_count"], errors="coerce"
    )
    out["product_images_count"] = pd.to_numeric(
        out["product_images_count"], errors="coerce"
    )

    if "verified_purchase_clean" not in out:
        out["verified_purchase_clean"] = np.nan
    out["verified_purchase_status"] = (
        out["verified_purchase_clean"]
        .map({True: "verified", False: "not_verified"})
        .fillna("missing")
        .astype(str)
    )

    out["review_text_available"] = _to_yes_no(text.ne(""))
    out["review_image_available"] = _to_yes_no(out["review_images_count"].fillna(0).gt(0))
    out["product_image_available"] = _to_yes_no(out["product_images_count"].fillna(0).gt(0))

    description_flag = out.get(
        "product_description_present", pd.Series(False, index=out.index)
    )
    out["product_description_available"] = _to_yes_no(description_flag)

    metadata_flag = out.get("metadata_matched", pd.Series(False, index=out.index))
    out["product_metadata_available"] = _to_yes_no(metadata_flag)

    taxonomy = out.get(
        "product_categories", pd.Series(np.nan, index=out.index)
    )
    out["product_main_category"] = taxonomy.map(_primary_category)
    # Explicit missingness category; never replace an unknown price by zero.
    out["product_price_missing"] = _to_yes_no(out["product_price_num"].isna())

    user_col = "user_id" if "user_id" in out else None
    product_col = "_join_key" if "_join_key" in out else None

    out["reviewer_prior_review_count"] = (
        _prior_count_for_groups(out, [user_col]) if user_col else np.nan
    )
    out["product_prior_review_count"] = (
        _prior_count_for_groups(out, ["dataset_category", product_col]) if (product_col and "dataset_category" in out) else (
            _prior_count_for_groups(out, [product_col]) if product_col else np.nan
        )
    )
    if user_col and product_col:
        out["user_product_prior_interaction_count"] = _prior_count_for_groups(
            out, ["dataset_category", user_col, product_col]
        )
    else:
        out["user_product_prior_interaction_count"] = np.nan

    out["log1p_text_word_count"] = np.log1p(out["text_word_count"].clip(lower=0))
    out["log1p_product_price"] = np.log1p(out["product_price_num"].clip(lower=0))
    out["log1p_reviewer_prior_review_count"] = np.log1p(
        out["reviewer_prior_review_count"].clip(lower=0)
    )
    out["log1p_product_prior_review_count"] = np.log1p(
        out["product_prior_review_count"].clip(lower=0)
    )
    out["log1p_user_product_prior_interaction_count"] = np.log1p(
        out["user_product_prior_interaction_count"].clip(lower=0)
    )

    available_metadata_fields = pd.DataFrame({
        "price": out["product_price_num"].notna(),
        "description": description_flag.fillna(False).astype(bool),
        "product_image": out["product_images_count"].fillna(0).gt(0),
        "main_category": out["product_main_category"].notna(),
    }, index=out.index)
    out["metadata_completeness_score"] = available_metadata_fields.sum(axis=1).astype(float)

    # The target is retained unchanged for later descriptive profiling only.
    if "helpful_vote_num" not in out:
        out["helpful_vote_num"] = np.nan

    return out


def get_feature_columns() -> tuple[list[str], list[str]]:
    """Return the fixed numeric and categorical predictor columns."""
    return NUMERIC_FEATURES.copy(), CATEGORICAL_FEATURES.copy()


def get_feature_registry() -> pd.DataFrame:
    """Return a human-readable feature dictionary."""
    return pd.DataFrame(
        FEATURE_REGISTRY,
        columns=["feature_or_group", "definition", "why_it_may_help"],
    )
