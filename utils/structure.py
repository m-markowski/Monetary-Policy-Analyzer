import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def standardize(frame: pd.DataFrame) -> pd.DataFrame | None:
    """
    Z-score every column so distance-based methods weight features equally.

    Clustering and PCA are scale-sensitive. Zero-variance columns are
    dropped, as they carry no information and would divide by zero.

    Args:
        frame (pd.DataFrame): Numeric, missing-free feature frame (dates x features).

    Returns:
        pd.DataFrame | None: Standardised frame with the original index, or None if
        fewer than two rows or columns remain.
    """
    usable = frame.loc[:, frame.std(ddof=0) > 0]
    if usable.shape[0] < 2 or usable.shape[1] < 2:
        return None
    scaled = StandardScaler().fit_transform(usable)
    return pd.DataFrame(scaled, index=usable.index, columns=usable.columns)


def pca_summary(scaled: pd.DataFrame, n_components: int | None = None) -> dict | None:
    """
    Fit PCA and return scores, explained variance and loadings.

    Principal components are orthogonal linear combinations of the (standardised)
    features ordered by the variance share they capture, so a handful often
    summarise many collinear macro/rate series. Loadings show how each original
    feature contributes to each component, which is how the components are read.

    Args:
        scaled (pd.DataFrame): Standardised feature frame.
        n_components (int | None): Components to keep; None keeps the maximum

    Returns:
        dict | None: 'scores' (dates x PC), 'explained_variance_ratio' (per PC), and
        'loadings' (feature x PC); None if fewer than two features.
    """
    if scaled.shape[1] < 2 or scaled.shape[0] < 2:
        return None
    max_k = min(scaled.shape[0] - 1, scaled.shape[1])  # (min of rows - 1 and feature count)
    k = max_k if n_components is None else min(n_components, max_k)

    pca = PCA(n_components=k)
    scores = pca.fit_transform(scaled)
    names = [f"PC{i + 1}" for i in range(k)]
    return {
        "scores": pd.DataFrame(scores, index=scaled.index, columns=names),
        "explained_variance_ratio": pd.Series(pca.explained_variance_ratio_, index=names),
        "loadings": pd.DataFrame(pca.components_.T, index=scaled.columns, columns=names),
    }


def kmeans_sweep(scaled: pd.DataFrame, k_min: int = 2, k_max: int = 8, random_state: int = 0) -> pd.DataFrame | None:
    """
    Fit K-Means across a range of k and score each fit.

    Reports inertia (within-cluster sum of squares, for the elbow) and the mean
    silhouette (cluster separation, -1 to 1, higher is better) so the page can
    suggest a k and let the user override it.

    Args:
        scaled (pd.DataFrame): Standardised feature frame.
        k_min (int): Smallest number of clusters to try.
        k_max (int): Largest number of clusters to try.
        random_state (int): Seed for reproducible fits.

    Returns:
        pd.DataFrame | None: One row per k with 'inertia' and 'silhouette',
        indexed by k; None if no candidate has a valid silhouette score.
    """
    data = scaled.to_numpy()
    top = min(k_max, data.shape[0] - 1, len(np.unique(data, axis=0)))
    if top < k_min:
        return None

    rows = []
    for k in range(k_min, top + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(data)
        try:
            score = float(silhouette_score(data, labels, sample_size=min(2000, len(data)), random_state=random_state))
        except ValueError:
            # A subsample can miss a very small cluster, making silhouette undefined.
            continue
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": score})
    return pd.DataFrame(rows).set_index("k") if rows else None


def kmeans_labels(scaled: pd.DataFrame, k: int, random_state: int = 0) -> dict | None:
    """
    Fit K-Means for a chosen k and return labels and quality scores.

    Args:
        scaled (pd.DataFrame): Standardised feature frame.
        k (int): Number of clusters.
        random_state (int): Seed for a reproducible fit.

    Returns:
        dict | None: 'labels' (1-based cluster id per date, categorical),
        'silhouette', 'inertia' and 'sizes' (count per cluster);
        None if k or the silhouette sample is invalid.
    """
    data = scaled.to_numpy()
    if not 2 <= k <= min(data.shape[0] - 1, len(np.unique(data, axis=0))):
        return None

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    raw = model.fit_predict(data)
    labels = pd.Series(raw + 1, index=scaled.index, name="cluster").astype("category")
    try:
        score = float(silhouette_score(data, labels, sample_size=min(2000, len(data)), random_state=random_state))
    except ValueError:
        return None
    return {
        "labels": labels,
        "silhouette": score,
        "inertia": float(model.inertia_),
        "sizes": labels.value_counts().sort_index(),
    }


def cluster_agreement(labels: pd.Series, regime: pd.Series) -> dict | None:
    """
    Measure how closely data-driven clusters match a rule-based regime label.

    The adjusted Rand index compares two partitions of the same dates, correcting
    for chance: 1 is identical grouping, 0 is no better than random, negative is
    worse than random. It quantifies whether the unsupervised structure recovers
    the known economic regimes.

    Args:
        labels (pd.Series): Cluster ids, date-indexed.
        regime (pd.Series): Categorical regime labels, date-indexed.

    Returns:
        dict | None: 'ari' and the overlap count 'n'; None if fewer than two
        common, non-missing dates or only one regime present.
    """
    pair = pd.DataFrame({"cluster": labels, "regime": regime}).dropna()
    if pair.shape[0] < 2 or pair["regime"].nunique() < 2:
        return None
    ari = adjusted_rand_score(pair["regime"].astype(str), pair["cluster"].astype(str))
    return {"ari": float(ari), "n": int(pair.shape[0])}
