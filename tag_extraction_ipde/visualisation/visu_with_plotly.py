import pickle

import numpy as np
import pandas as pd
import plotly.express as px  # type:ignore


def insert_line_breaks(text, num_words):
    words = text.split()
    lines = [
        " ".join(words[i : i + num_words]) for i in range(0, len(words), num_words)
    ]
    return "<br>".join(lines)


def get_plotly_graph(vec_dim2, labels, data_message, dico_label_tag=None):
    if not dico_label_tag:
        df = pd.DataFrame(
            {
                "x": vec_dim2[:, 0],
                "y": vec_dim2[:, 1],
                "cluster": labels,
                "info": [
                    f"{msg[1]}<br>{insert_line_breaks(msg[2], 20)}"
                    for msg in data_message
                ],
            }
        )
    else:
        table_tag = []
        for label in labels:
            table_tag.append(dico_label_tag[label])
        df = pd.DataFrame(
            {
                "x": vec_dim2[:, 0],
                "y": vec_dim2[:, 1],
                "cluster": labels,
                "info": [
                    f"{msg[1]}<br>{insert_line_breaks(msg[2], 20)}"
                    for msg in data_message
                ],
                "additional_info": [f"tag associé : {tag}" for tag in table_tag],
            }
        )
        df["info"] = df["info"] + "<br><br>" + df["additional_info"]
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        title="Visualisation des clusters",
    )
    fig.update_traces(
        hovertemplate="<b>Cluster:</b> %{customdata[0]}<br><b>Info:</b><br>%{customdata[1]}<extra></extra>",
        customdata=np.stack((df["cluster"], df["info"]), axis=-1),
    )
    fig.update_layout(
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
    )

    fig.show()


def get_tab_one_cluster(vec_dim2, labels, data_message, desired_cluster):
    desired_index = []
    for i, label in enumerate(labels):
        if label == desired_cluster:
            desired_index.append(i)
    return vec_dim2[desired_index], labels[desired_index], data_message[desired_index]


if __name__ == "__main__":
    data_message = np.load(
        "../data/data_numpy_msg_and_categories_ipde_0824.npy", allow_pickle=True
    )

    vec_embeddgins_dim2 = np.load(
        "../data/vec_embeddings_norm_reduit_dim2_2_neighbors15_mindist005.npy",
        allow_pickle=True,
    )

    labels = np.load(
        "../data/nouveau_labels_hdbscan_dim50_eps02_mi200_V2.npy", allow_pickle=True
    )

    with open(
        "../dico_num_tag_name_tag/dico_tag_cluster_hdbscan_dim50_eps02_min200_V2.pkl",
        "rb",
    ) as f:
        dico_label_tag = pickle.load(f)

    get_plotly_graph(vec_embeddgins_dim2, labels, data_message, dico_label_tag)

    vec_dim2_part, labels_part, data_message_part = get_tab_one_cluster(
        vec_embeddgins_dim2, labels, data_message, 20
    )

    get_plotly_graph(vec_dim2_part, labels_part, data_message_part, dico_label_tag)
