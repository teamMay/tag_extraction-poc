import numpy as np
from sklearn.cluster import KMeans  # type:ignore
from sklearn.metrics import davies_bouldin_score, silhouette_score  # type:ignore
from sklearn.preprocessing import normalize  # type:ignore
from visualisation.visu_with_plotly import (  # type:ignore
    get_plotly_graph,
    get_tab_one_cluster,
)

# Import des données

if True:
    data_message = np.load(
        "../data/data_numpy_msg_and_categories_ipde_0824.npy", allow_pickle=True
    )

    vec_embeddings_reduit50 = np.load(
        "../data/vec_embeddings_norm_reduit_dim50_neighbors15_mindist005.npy",
        allow_pickle=True,
    )

    vec_embeddings_reduit2 = np.load(
        "../data/vec_embeddings_norm_reduit_dim2_2_neighbors15_mindist005.npy",
        allow_pickle=True,
    )


K_table = [50, 70, 100, 150, 180, 200]


def fit_kmeans(vec_embeddings):
    vec_embeddings_norm = normalize(vec_embeddings, norm="l2")
    dico_des_resultats = {}

    for k in K_table:
        kmeans = KMeans(n_clusters=k, random_state=11)
        kmeans.fit(vec_embeddings_norm)
        labels = kmeans.labels_

        metric = davies_bouldin_score(vec_embeddings_norm, labels)
        metric = silhouette_score(vec_embeddings_norm, labels)

        print(metric)
        dico_des_resultats[k] = metric

    return dico_des_resultats, max(dico_des_resultats, key=dico_des_resultats.get)


# class restructuring if some cluster are not satifiying :


# First visualisation to know how to change
def class_restructuring_with_kmeans(
    vec_embeddings,
    vec_embeddings_dim2,
    labels_initial,
    cluster_to_change,
    number_of_new_cluster,
):
    vec_embeddings_norm = normalize(vec_embeddings, norm="l2")
    table_indice_of_the_cluster_to_change = []
    for i, label in enumerate(labels_initial):
        if label == cluster_to_change:
            table_indice_of_the_cluster_to_change.append(i)

    get_plotly_graph(
        vec_embeddings_dim2[table_indice_of_the_cluster_to_change],
        np.array(len(vec_embeddings_dim2[table_indice_of_the_cluster_to_change])),
        data_message[table_indice_of_the_cluster_to_change],
    )

    kmeans = KMeans(n_clusters=number_of_new_cluster, random_state=11)
    kmeans.fit(vec_embeddings_norm[table_indice_of_the_cluster_to_change])
    new_labels_of_the_cluster_to_change = kmeans.labels_

    get_plotly_graph(
        vec_embeddings_dim2[table_indice_of_the_cluster_to_change],
        new_labels_of_the_cluster_to_change,
        data_message[table_indice_of_the_cluster_to_change],
    )

    for label in np.unique(new_labels_of_the_cluster_to_change):
        get_plotly_graph(
            get_tab_one_cluster(
                vec_embeddings_dim2[table_indice_of_the_cluster_to_change],
                new_labels_of_the_cluster_to_change,
                data_message[table_indice_of_the_cluster_to_change],
                label,
            )
        )


# Then change it with the dico
def change_the_big_label_tab(
    dico_nouveau_ancien_label,
    table_indice_of_the_cluster_to_change,
    new_labels_of_the_cluster_to_change,
    labels_total,
):
    nouveau_labels = []
    for label in new_labels_of_the_cluster_to_change:
        nouveau_labels.append(dico_nouveau_ancien_label[label])

    labels_total_2 = labels_total.copy()
    for i, indice_global in enumerate(table_indice_of_the_cluster_to_change):
        labels_total_2[indice_global] = nouveau_labels[i]

    table_labels_total_2_np = np.array(labels_total_2)
    np.save(
        "../data/table_all_labels_dim50_eps0.005_min100_restructuring.npy",
        table_labels_total_2_np,
    )


dico_nouveau_ancien_label_test = {
    0: 86,
    1: 88,
    2: 89,
}
