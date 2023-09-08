"""Dunn Index for unsupervised evaluation metrics."""

# Author: Noberto Maciel <nobertomaciel@hotmail.com>
# License: BSD 3 clause


# import functools
# from numbers import Integral

# import numpy as np
# from scipy.sparse import issparse

# from preprocessing import LabelEncoder
# from utils import _safe_indexing, check_random_state, check_X_y
# from utils._param_validation import (
#     Interval,
#     StrOptions,
#     validate_params,
# )
# from ..pairwise import _VALID_METRICS, pairwise_distances, pairwise_distances_chunked


def check_number_of_labels(n_labels, n_samples):
    """Check that number of labels are valid.

    Parameters
    ----------
    n_labels : int
        Number of labels.

    n_samples : int
        Number of samples.
    """
    if not 1 < n_labels < n_samples:
        raise ValueError(
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)"
            % n_labels
        )

@validate_params(
    {
        "X": ["array-like"],
        "labels": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def dunn_index(X, labels):
    """Compute the Dunn index.

    The index is defined as the average similarity measure of each cluster with
    its most similar cluster, where similarity is the ratio of within-cluster
    distances to between-cluster distances. Thus, clusters which are farther
    apart and less dispersed will result in a better index.

    The minimum index is zero, with lower values indicating better clustering.

    Read more in the :ref:`User Guide <dunn_index>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    index: float
        The resulting Dunn Index.

    References
    ----------
    .. [1] J. C. Dunn (1973).
       "A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters"
        Journal of Cybernetics, 3:3, 32-57
        DOI: 10.1080/01969727308546046
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=float)
    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = max(pairwise_distances(cluster_k, [centroid]))

    centroid_distances = pairwise_distances(centroids)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    #scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    scores = np.min(centroid_distances / combined_intra_dists, axis=1)
    return np.mean(scores)
