import pickle

import numpy as np
from sklearn.preprocessing import normalize  # type:ignore
from umap.umap_ import UMAP  # type:ignore

# Import data:
vec_embeddings = np.load(
    "../data_midwife/embeddings_sf.npy",
    allow_pickle=True,
)


def get_reducer_with_umap(vec_embeddings, output_dim=2):
    vec_embeddings_norm = normalize(vec_embeddings, norm="l2")

    reducer = UMAP(
        n_neighbors=15,
        n_components=output_dim,
        min_dist=0.05,
        metric="cosine",
        verbose=True,
        random_state=11,
    )
    reducer.fit(vec_embeddings_norm)

    with open(f"umap_model_sf_dim{output_dim}.pkl", "wb") as f:
        pickle.dump(reducer, f)


def dim_reduction_of_the_vector(vec_embeddings, save=False):
    vec_embeddings_norm = normalize(vec_embeddings, norm="l2")

    with open("umap_model_sf_dim2.pkl", "rb") as f:
        reducer = pickle.load(f)

    vec_embeddings_norm_reduit = reducer.transform(vec_embeddings_norm)

    vec_embeddings_norm_reduit_np = np.array(vec_embeddings_norm_reduit)
    print(vec_embeddings_norm_reduit_np.shape)

    if save:
        np.save(
            "../data_midwife/vec_embeddings_sf_norm_reduit_dim2_neighbors15_mindist005.npy",
            vec_embeddings_norm_reduit_np,
        )
