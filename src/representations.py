from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize


def fit_svd_reduction(
    X_train,
    X_validation=None,
    X_test=None,
    n_components: int = 30,
    random_state: int = 42,
):
    """Fit TruncatedSVD on training rows only and transform held-out rows."""
    n_samples, n_features = X_train.shape
    effective_components = min(n_components, n_features - 1, n_samples - 1)
    if effective_components < 2:
        raise ValueError(
            f"Not enough rows/features for SVD: shape={X_train.shape}"
        )

    reducer = TruncatedSVD(
        n_components=effective_components,
        algorithm="randomized",
        n_iter=7,
        random_state=random_state,
    )
    train_reduced = reducer.fit_transform(X_train)
    validation_reduced = (
        reducer.transform(X_validation) if X_validation is not None else None
    )
    test_reduced = reducer.transform(X_test) if X_test is not None else None

    return reducer, train_reduced, validation_reduced, test_reduced


def fit_pca_projection(
    X_train,
    X_validation=None,
    X_test=None,
    random_state: int = 42,
):
    """Fit a two-dimensional PCA projection on the reduced training representation."""
    train_array = np.asarray(X_train)
    if train_array.ndim != 2 or train_array.shape[1] < 2:
        raise ValueError("PCA projection requires a 2D matrix with at least 2 columns")

    projection = PCA(n_components=2, random_state=random_state)
    train_2d = projection.fit_transform(train_array)
    validation_2d = (
        projection.transform(np.asarray(X_validation))
        if X_validation is not None else None
    )
    test_2d = (
        projection.transform(np.asarray(X_test))
        if X_test is not None else None
    )
    return projection, train_2d, validation_2d, test_2d


def combine_modalities(
    text_features,
    structured_features,
    text_weight: float = 1.0,
    structured_weight: float = 1.0,
):
    """L2-normalize each modality separately and concatenate sparse matrices."""
    text_matrix = sparse.csr_matrix(text_features)
    structured_matrix = sparse.csr_matrix(structured_features)

    text_matrix = normalize(text_matrix, norm="l2", axis=1, copy=False)
    structured_matrix = normalize(
        structured_matrix, norm="l2", axis=1, copy=False
    )

    return sparse.hstack(
        [
            text_matrix.multiply(text_weight),
            structured_matrix.multiply(structured_weight),
        ],
        format="csr",
    )


def extract_sentence_embeddings(
    texts,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str | None = None,
):
    """
    Optional frozen pretrained text embeddings.

    Requires the optional sentence-transformers package and access to the model
    weights (locally cached or downloadable). No model is trained in this project.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install requirements-embeddings.txt to use pretrained text embeddings."
        ) from exc

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    metadata = {
        "model_name": model_name,
        "n_rows": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "normalized": True,
    }
    return embeddings.astype(np.float32), metadata


def fit_umap_projection(
    X_train,
    X_validation=None,
    X_test=None,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
):
    """Optional UMAP visualization; fit on training rows and transform held-out rows."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "Install umap-learn to use UMAP; PCA is already provided as the default."
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
        transform_seed=random_state,
    )
    train_2d = reducer.fit_transform(X_train)
    validation_2d = (
        reducer.transform(X_validation) if X_validation is not None else None
    )
    test_2d = reducer.transform(X_test) if X_test is not None else None
    return reducer, train_2d, validation_2d, test_2d
