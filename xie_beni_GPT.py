# code generated from Chat GPT
def xie_beni_index(data, cluster_centers):
    """
    Calcula o Índice de Xie Beni para a clusterização de dados.

    Args:
    data (list of lists): Os dados, onde cada sublista representa um ponto de dados.
    cluster_centers (list of lists): Os centros dos clusters, onde cada sublista representa um centro de cluster.

    Returns:
    float: O valor do Índice de Xie Beni.

    data = X
    cluster_centers = labels
    """

    num_clusters = len(cluster_centers)
    num_data_points = len(data)

    # Inicialize uma lista para armazenar as distâncias mínimas de cada ponto aos centros dos clusters
    min_distances = []

    for i in range(num_data_points):
        distances = [np.linalg.norm(np.array(data[i]) - np.array(center)) for center in cluster_centers]
        min_distance = min(distances)
        min_distances.append(min_distance)

    # Calcule o denominador do Índice de Xie Beni
    denominator = sum(min_distances)

    # Calcule o numerador do Índice de Xie Beni
    numerator = 0
    for i in range(num_clusters):
        for j in range(num_data_points):
            numerator += (np.linalg.norm(np.array(data[j]) - np.array(cluster_centers[i])) / min_distances[j]) ** 2

    # Calcule o Índice de Xie Beni
    xie_beni_index = numerator / (num_clusters * denominator)

    return xie_beni_index
