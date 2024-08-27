import pickle

import numpy as np
from sklearn.preprocessing import normalize  # type:ignore
from umap.umap_ import UMAP  # type:ignore

# Import data:
vec_embeddings = np.load(
    "../data/embeddings_numpy_update.npy",
    allow_pickle=True,
)

vec_embeddings_reduit2_question = np.load(
    "../data/vec_embeddings_norm_reduit_dim2_neighbors15_mindist005.npy",
    allow_pickle=True,
)


def get_reducer_with_umap(vec_embeddings):
    vec_embeddings_norm = normalize(vec_embeddings, norm="l2")

    reducer = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.05,
        metric="cosine",
        verbose=True,
        random_state=11,
    )
    reducer.fit(vec_embeddings_norm)

    with open("umap_model_dim2.pkl", "wb") as f:
        pickle.dump(reducer, f)


def dim_reduction_of_the_vector(vec_embeddings, save=False):
    vec_embeddings_norm = normalize(vec_embeddings, norm="l2")

    with open("umap_model_dim2.pkl", "rb") as f:
        reducer = pickle.load(f)

    vec_embeddings_norm_reduit = reducer.transform(vec_embeddings_norm)

    vec_embeddings_norm_reduit_np = np.array(vec_embeddings_norm_reduit)
    print(vec_embeddings_norm_reduit_np.shape)

    if save:
        np.save(
            "../data/vec_embeddings_norm_reduit_dim2_2_neighbors15_mindist005.npy",
            vec_embeddings_norm_reduit_np,
        )
