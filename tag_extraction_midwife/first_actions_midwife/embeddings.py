import numpy as np
from sentence_transformers import SentenceTransformer

data_message_sf = np.load(
    "../data/data_numpy_clean_sf.npy",
    allow_pickle=True,
)

questions_sf = data_message_sf[:, 2]
print(questions_sf[10])


model = SentenceTransformer(
    "../models/output_triplet_flaubert-base-uncased-xnli-sts_2024-07-04_15-10-51"
)


embeddings_sf = model.encode(
    data_message_sf[:, 2],
    batch_size=64,
    convert_to_numpy=True,
    show_progress_bar=True,
)

np.save("./embeddings_sf.npy", embeddings_sf)

# Réalisé sur datalba car sinon trop long
