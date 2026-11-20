from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def _take_rows(X, indices):
    return X[indices]


def evaluate_kmeans_candidates(
    X,
    k_values=range(2, 9),
    random_state: int = 42,
    n_init: int = 10,
    max_fit_rows: int = 15_000,
    silhouette_sample_size: int = 3_000,
) -> pd.DataFrame:
    """Evaluate K-means candidates using inertia, silhouette and cluster balance."""
    rng = np.random.default_rng(random_state)
    if X.shape[0] > max_fit_rows:
        fit_indices = np.sort(
            rng.choice(X.shape[0], size=max_fit_rows, replace=False)
        )
        X_eval = _take_rows(X, fit_indices)
    else:
        X_eval = X

    rows = []
    for k in k_values:
        model = KMeans(
            n_clusters=int(k),
            random_state=random_state,
            n_init=n_init,
        )
        labels = model.fit_predict(X_eval)
        counts = np.bincount(labels, minlength=int(k))
        fractions = counts / counts.sum()

        if len(np.unique(labels)) > 1 and X_eval.shape[0] > int(k):
            silhouette = silhouette_score(
                X_eval,
                labels,
                sample_size=min(silhouette_sample_size, X_eval.shape[0]),
                random_state=random_state,
            )
        else:
            silhouette = np.nan

        rows.append({
            "k": int(k),
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette),
            "smallest_cluster_n": int(counts.min()),
            "smallest_cluster_fraction": float(fractions.min()),
            "largest_cluster_fraction": float(fractions.max()),
            "fit_rows": int(X_eval.shape[0]),
        })

    return pd.DataFrame(rows)


def evaluate_seed_stability(
    X,
    k: int,
    seeds=(42, 52, 62),
    n_init: int = 10,
) -> dict:
    """Compare K-means assignments across random seeds using pairwise ARI."""
    label_sets = []
    for seed in seeds:
        model = KMeans(
            n_clusters=int(k),
            random_state=int(seed),
            n_init=n_init,
        )
        label_sets.append(model.fit_predict(X))

    scores = [
        adjusted_rand_score(label_sets[i], label_sets[j])
        for i, j in combinations(range(len(label_sets)), 2)
    ]
    return {
        "k": int(k),
        "seed_count": len(seeds),
        "mean_pairwise_ari": float(np.mean(scores)) if scores else np.nan,
        "min_pairwise_ari": float(np.min(scores)) if scores else np.nan,
        "pairwise_ari_values": scores,
    }


def evaluate_subsample_stability(
    X,
    k: int,
    baseline_model=None,
    fractions=(0.6, 0.8),
    seeds=(42, 52, 62),
    reference_size: int = 5_000,
    n_init: int = 10,
) -> pd.DataFrame:
    """Fit on different training subsamples and compare predictions on fixed rows."""
    rng = np.random.default_rng(42)
    reference_indices = np.sort(
        rng.choice(X.shape[0], size=min(reference_size, X.shape[0]), replace=False)
    )
    X_reference = _take_rows(X, reference_indices)

    if baseline_model is None:
        baseline_model = KMeans(
            n_clusters=int(k), random_state=42, n_init=n_init
        ).fit(X)
    baseline_labels = baseline_model.predict(X_reference)

    rows = []
    for fraction in fractions:
        sample_size = max(int(k) + 1, int(X.shape[0] * fraction))
        sample_size = min(sample_size, X.shape[0])
        for seed in seeds:
            local_rng = np.random.default_rng(seed)
            sample_indices = local_rng.choice(
                X.shape[0], size=sample_size, replace=False
            )
            model = KMeans(
                n_clusters=int(k),
                random_state=int(seed),
                n_init=n_init,
            ).fit(_take_rows(X, sample_indices))
            predicted = model.predict(X_reference)
            rows.append({
                "training_fraction": float(fraction),
                "seed": int(seed),
                "training_rows": int(sample_size),
                "reference_rows": int(len(reference_indices)),
                "ari_vs_full_training_solution": float(
                    adjusted_rand_score(baseline_labels, predicted)
                ),
            })
    return pd.DataFrame(rows)


def evaluate_scaling_sensitivity(
    X,
    k: int,
    baseline_model=None,
    random_state: int = 42,
    n_init: int = 10,
) -> dict:
    """Compare the baseline clustering with an alternative global standard scaling."""
    if baseline_model is None:
        baseline_model = KMeans(
            n_clusters=int(k), random_state=random_state, n_init=n_init
        ).fit(X)

    baseline_labels = baseline_model.predict(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.asarray(X))

    scaled_model = KMeans(
        n_clusters=int(k), random_state=random_state, n_init=n_init
    ).fit(X_scaled)
    scaled_labels = scaled_model.predict(X_scaled)

    return {
        "comparison": "standard scaling of reduced components vs baseline representation",
        "ari_vs_baseline": float(
            adjusted_rand_score(baseline_labels, scaled_labels)
        ),
        "scaled_inertia": float(scaled_model.inertia_),
    }


def summarize_clusters(
    df: pd.DataFrame,
    labels,
    numeric_columns,
    target_column: str = "helpful_vote_num",
    cluster_column: str = "cluster",
) -> pd.DataFrame:
    """
    Summarize feature distributions and, separately, the held-out target outcome.
    The target is never passed to K-means or used to select k.
    """
    work = df.copy()
    work[cluster_column] = np.asarray(labels)
    numeric_columns = [c for c in numeric_columns if c in work.columns]

    grouped = work.groupby(cluster_column, dropna=False)
    summary = grouped.size().rename("n_reviews").to_frame()
    summary["cluster_share"] = summary["n_reviews"] / len(work)

    if numeric_columns:
        medians = grouped[numeric_columns].median(numeric_only=True)
        medians = medians.add_prefix("median_")
        summary = summary.join(medians)

    if target_column in work.columns:
        target_rows = []
        for cluster, group in grouped:
            votes = pd.to_numeric(group[target_column], errors="coerce").dropna()
            votes = votes[votes >= 0]
            target_rows.append({
                cluster_column: cluster,
                "target_valid_n": int(len(votes)),
                "helpful_vote_mean_posthoc": float(votes.mean()) if len(votes) else np.nan,
                "helpful_vote_median_posthoc": float(votes.median()) if len(votes) else np.nan,
                "helpful_vote_p90_posthoc": float(votes.quantile(0.90)) if len(votes) else np.nan,
                "helpful_vote_nonzero_rate_posthoc": float((votes > 0).mean()) if len(votes) else np.nan,
                "mean_log1p_helpful_vote_posthoc": float(np.log1p(votes).mean()) if len(votes) else np.nan,
            })
        summary = summary.join(
            pd.DataFrame(target_rows).set_index(cluster_column)
        )

    return summary.reset_index()


def cluster_category_mix(
    df: pd.DataFrame,
    labels,
    category_column: str = "dataset_category",
) -> pd.DataFrame:
    """Return row-normalized category composition for each cluster."""
    work = pd.DataFrame({
        "cluster": np.asarray(labels),
        category_column: df[category_column].astype(str).to_numpy(),
    })
    counts = pd.crosstab(work["cluster"], work[category_column])
    proportions = counts.div(counts.sum(axis=1), axis=0)
    return proportions.reset_index()


def top_terms_by_cluster(
    tfidf_matrix,
    labels,
    vectorizer,
    top_n: int = 12,
) -> pd.DataFrame:
    """Find high-average-TF-IDF terms per cluster for qualitative interpretation."""
    terms = np.asarray(vectorizer.get_feature_names_out())
    labels = np.asarray(labels)
    rows = []

    for cluster in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == cluster)
        if len(indices) == 0:
            continue
        mean_weights = np.asarray(tfidf_matrix[indices].mean(axis=0)).ravel()
        top_indices = np.argsort(mean_weights)[::-1][:top_n]
        for rank, term_index in enumerate(top_indices, start=1):
            rows.append({
                "cluster": int(cluster),
                "rank": rank,
                "term": str(terms[term_index]),
                "mean_tfidf": float(mean_weights[term_index]),
            })

    return pd.DataFrame(rows)
