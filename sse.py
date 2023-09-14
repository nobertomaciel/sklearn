# Author: Noberto Maciel <nobertomaciel@hotmail.com>
# License: BSD 3 clause

@validate_params(
    {
        "X": ["array-like"],
        "labels": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def sse(X, labels):
    """Compute the Sum of Squared Error.

    The Sum of Squared Error in cluster analisys get all distances between 
    an element and the center of cluster.

    The minimum score is zero (when elements are equal to center of cluster), 
    with lower values indicating better clustering.

    Read more in the :ref:`User Guide <sse>`.

    .. versionadded:: 0.10

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    score: float
        The resulting SSE value.

    References
    ----------
    .. [1] 
    """
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
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    intra_dists = np.zeros(n_labels) # same cluster dist
    centroids = np.zeros((n_labels, len(X[0])), dtype=float)
    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.sum((pairwise_distances(cluster_k, [centroid]))**2)


    if np.allclose(intra_dists, 0):
        return 0.0


    scores = np.sum(intra_dists)
    return scores
