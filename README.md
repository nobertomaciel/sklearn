# sklearn complementary clustering measures

## Xie Beni Index
**Compute the Xie Beni Index.**
https://github.com/nobertomaciel/sklearn/blob/main/xie_beni.py

    Xie and Beni introduced Xie-Beni (XB) index method in 1991. 
    XB index is focus on separation and com- pactness. 
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

## SSE
**Compute the Sum of Squared Error.**
https://github.com/nobertomaciel/sklearn/blob/main/sse.py
    The Sum of Squared Error in cluster analisys get all distances between an element and the center of cluster.

    The minimum score is zero (when elements are equal to center of cluster), with lower values indicating better clustering.

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

## Dunn Index

(WARNING: routine still in development - being adjusted)
<br>
Extracted from https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d/sklearn/metrics/cluster/_unsupervised.py#L360
