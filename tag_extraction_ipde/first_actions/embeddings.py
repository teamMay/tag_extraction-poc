import numpy as np
from sentence_transformers import SentenceTransformer  # type:ignore

model_embeddings = SentenceTransformer(
    "../../models/output_triplet_flaubert-base-uncased-xnli-sts_2024-07-04_15-10-51"
)

tab_msg_and_cat = np.load(
    "../data/data_numpy_msg_and_categories_ipde_0824.npy", allow_pickle=True
)

msg = tab_msg_and_cat[:, 2]
print(msg[2])
print(len(msg))

embeddings_sentence = model_embeddings.encode(
    msg,
    batch_size=64,
    convert_to_numpy=True,
    show_progress_bar=True,
)

np.save("../data/embeddings_msg_ipde_0824.npy", embeddings_sentence)
