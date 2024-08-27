import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score  # type:ignore

# debut

vec_embeddings_reduit10 = np.load(
    "../data/vec_embeddings_norm_reduit_dim100_neighbors15_mindist005.npy",
    allow_pickle=True,
)

labels_hdbscan = np.load(
    "../data/labels_hdbscan_dim10_eps0.2_min200.npy",
    allow_pickle=True,
)


def get_graph_with_silhouette_score(labels, vec_embeddings):
    silhouette_avg = silhouette_score(vec_embeddings, labels)
    print(f"Le score moyen de silhouette pour tous les clusters est : {silhouette_avg}")

    print("Calcul des scores de silhouette pour chaque point : ")

    sample_silhouette_values = silhouette_samples(vec_embeddings, labels)

    n_clusters = len(set(labels))
    fig, ax = plt.subplots(figsize=(18, 8))

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax.set_title("Silhouette plot pour les clusters")
    ax.set_xlabel("Coefficient de silhouette")
    ax.set_ylabel("Cluster")

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.show()
