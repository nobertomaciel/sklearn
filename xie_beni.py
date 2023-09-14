# Author: Noberto Maciel <nobertomaciel@hotmail.com>
# License: BSD 3 clause

@validate_params(
    {
        "X": ["array-like"],
        "labels": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def xie_beni_index(X, labels):
    """Compute the Xie Beni Index.

    Xie and Beni introduced Xie-Beni (XB) index method in 1991. 
    XB index is focus on separation and compactness. 
    Separation is a measure of the distance between one cluster and 
    another cluster and compactness is a measure of proximity between data 
    points in a cluster (Lathief 2020).

    The minimum score is zero, with lower values indicating better clustering.

    Read more in the :ref:`User Guide <xie-beni_index>`.

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
        The resulting Xie Beni Index.

    References
    ----------
    .. [1] XIE, Xuanli Lisa ; BENI, Gerardo. 
        A validity measure for fuzzy clustering. 
        IEEE Transactions on Pattern Analysis & Machine Intelligence, 
        v. 13, n. 08, p. 841-847, 1991.
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
        intra_dists[k] = np.sum((pairwise_distances(cluster_k, [centroid])**2))

    centroid_distances = pairwise_distances(centroids)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf

    scores = (np.sum(intra_dists))/(n_labels*(np.min((centroid_distances**2))))
    return scores
