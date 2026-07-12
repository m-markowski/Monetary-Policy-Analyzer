import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
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


def hopkins_statistic(scaled: pd.DataFrame, sample_size: int | None = None, random_state: int = 0) -> float | None:
    """
    Hopkins statistic: does the data have any clustering tendency at all?

    Compares nearest-neighbour distances of real points with those of uniform
    random points drawn from the same bounding box. Values near 1 indicate strong
    clustering structure, near 0.5 a uniform (structureless) cloud. Run before
    clustering to confirm clusters are worth seeking.

    Args:
        scaled (pd.DataFrame): Standardised feature frame.
        sample_size (int | None): Number of points sampled for the comparison.
        random_state (int): Seed for reproducible sampling.

    Returns:
        float | None: The Hopkins statistic in [0, 1], or None if too few rows.
    """
    data = scaled.to_numpy()
    n, d = data.shape
    m = min(sample_size or max(50, n // 10), n - 1)  # ~10% of points, floor 50
    if m < 5:
        return None

    rng = np.random.default_rng(random_state)
    nbrs = NearestNeighbors(n_neighbors=2).fit(data)

    idx = rng.choice(n, size=m, replace=False)
    real_dist, _ = nbrs.kneighbors(data[idx])
    w = real_dist[:, 1].sum()  # column 0 is the point itself (distance 0)

    synthetic = rng.uniform(data.min(axis=0), data.max(axis=0), size=(m, d))
    synth_dist, _ = nbrs.kneighbors(synthetic)
    u = synth_dist[:, 0].sum()  # no self-distance here as synthetic samples are not in the training data

    total = u + w
    return float(u / total) if total > 0 else None


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
    if scaled.shape[1] < 2:
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
        indexed by k; None if there are too few rows for the range.
    """
    data = scaled.to_numpy()
    top = min(k_max, data.shape[0] - 1)
    if top < k_min:
        return None

    rows = []
    for k in range(k_min, top + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(data)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                # sample to reduce computation cost
                "silhouette": float(
                    silhouette_score(data, labels, sample_size=min(2000, len(data)), random_state=random_state)
                ),
            }
        )
    return pd.DataFrame(rows).set_index("k")


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
         None if k is invalid for the sample.
    """
    data = scaled.to_numpy()
    if not 2 <= k <= data.shape[0] - 1:
        return None

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    raw = model.fit_predict(data)
    labels = pd.Series(raw + 1, index=scaled.index, name="cluster").astype("category")
    return {
        "labels": labels,
        "silhouette": float(
            silhouette_score(data, labels, sample_size=min(2000, len(data)), random_state=random_state)
        ),  # sample to reduce computation cost
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
