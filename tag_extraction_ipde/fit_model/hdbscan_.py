import os
import sys
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hdbscan  # type:ignore
import numpy as np
from joblib import Memory  # type:ignore
from sklearn.neighbors import KNeighborsClassifier  # type:ignore
from sklearn.preprocessing import normalize  # type:ignore
from visualisation.visu_with_plotly import get_plotly_graph  # type:ignore

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


memory = Memory("cache_directory", verbose=0)


def fit_hdbscan(vec_embeddings, epsilon_table, min_table, dim=50, save=False):
    vec_embeddings_norm = normalize(vec_embeddings, norm="l2")
    for eps in epsilon_table:
        for min in min_table:
            print(f"\nepsilon : {eps} et min : {min}")
            clusterer = hdbscan.HDBSCAN(
                algorithm="best",
                cluster_selection_epsilon=eps,
                max_cluster_size=100000,
                alpha=0.5,
                approx_min_span_tree=True,
                gen_min_span_tree=True,
                leaf_size=40,
                memory=memory,
                metric="euclidean",
                min_cluster_size=min,
                min_samples=None,
                p=None,
            )

            clusterer.fit(vec_embeddings_norm)
            labels = clusterer.labels_
            labels_np = np.array(labels, dtype=int)
            if save:
                np.save(
                    f"../data/labels_thg_hdbscan/labels_thg_hdbscan_dim{dim}_eps{eps}_min{min}.npy",
                    labels_np,
                )

            counter = Counter(labels)

            print(counter)
            print(f"nbr de clusters : {len(counter)}")


# with hdbscan, there are outillers, to be able to remove them, I re-do an allocation with a knn based on the already classified.

# Allocation of unclassified (in tow functions (and a function that connects the two)) :


def get_dic_and_table_of_allocations_and_non_allocations(labels):
    dico_indice_vec_non_classer = {}
    dico_indice_label_classer = {}
    table_vec_classer = []
    table_label_classer = []

    for i, label in enumerate(labels):
        if label == -1:
            dico_indice_vec_non_classer[i] = vec_embeddings_reduit50[i]
        if label != -1:
            dico_indice_label_classer[i] = label
            table_vec_classer.append(vec_embeddings_reduit50[i])
            table_label_classer.append(label)

    table_vec_classer = np.array(table_vec_classer)
    table_label_classer = np.array(table_label_classer)

    return (
        dico_indice_vec_non_classer,
        dico_indice_label_classer,
        table_vec_classer,
        table_label_classer,
    )


def attribution_of_the_non_classifier_with_knn(
    vec_embeddings,
    table_vec_classified,
    table_label_classified,
    dico_index_vec_non_classified,
    dico_index_label_classified,
):
    knn_perso = KNeighborsClassifier(n_neighbors=9)
    knn_perso.fit(table_vec_classified, table_label_classified)

    dico_indice_label_predict_des_pas_classer = {}
    for indice, vec in dico_index_vec_non_classified.items():
        vec_np = np.array(vec).reshape(1, -1)
        label_predict = knn_perso.predict(vec_np)[0]
        dico_indice_label_predict_des_pas_classer[indice] = label_predict

    table_labels_total = []
    indices = np.arange(len(vec_embeddings))
    for indice in indices:
        if indice in dico_indice_label_predict_des_pas_classer:
            table_labels_total.append(dico_indice_label_predict_des_pas_classer[indice])
        elif indice in dico_index_label_classified:
            table_labels_total.append(dico_index_label_classified[indice])
        else:
            print("PROBLEME")

    table_labels_total_np = np.array(table_labels_total)

    return table_labels_total_np


def get_classified_all_vector(initial_labels):
    (
        dico_indice_vec_non_classer,
        dico_indice_label_classer,
        table_vec_classer,
        table_label_classer,
    ) = get_dic_and_table_of_allocations_and_non_allocations(initial_labels)

    labels_final = attribution_of_the_non_classifier_with_knn(
        vec_embeddings_reduit50,
        table_vec_classer,
        table_label_classer,
        dico_indice_vec_non_classer,
        dico_indice_label_classer,
    )

    return labels_final


if __name__ == "__main__":
    epsilon_table = [0.2]
    min_table = [200]
    fit_hdbscan(vec_embeddings_reduit50, epsilon_table, min_table, dim=50, save=True)

    labels_test = np.load(
        "../data/labels_thg_hdbscan/labels_thg_hdbscan_dim50_eps0.2_min200.npy",
        allow_pickle=True,
    )
    labels_all_classified = get_classified_all_vector(labels_test)

    get_plotly_graph(vec_embeddings_reduit2, labels_all_classified, data_message)
