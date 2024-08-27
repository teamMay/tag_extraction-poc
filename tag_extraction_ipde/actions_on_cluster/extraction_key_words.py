import pickle
from collections import defaultdict

import numpy as np
from keybert import KeyBERT  # type:ignore
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer  # type:ignore

print("Import models and data")


data_message = np.load(
    "../data/data_numpy_clean_update.npy",
    allow_pickle=True,
)

model = SentenceTransformer(
    "../models/output_triplet_flaubert-base-uncased-xnli-sts_2024-07-04_15-10-51"
)

vec_embeddings_reduit10 = np.load(
    "../data/vec_embeddings_norm_reduit_dim10_neighbors15_mindist005.npy",
    allow_pickle=True,
)


labels_hdbscan = np.load(
    "../data/labels_hdbscan_dim10_eps0.2_min200.npy",
    allow_pickle=True,
)


def get_dico_label_message(labels):
    dico_label_message = {}
    for i, label in enumerate(labels):
        if label not in dico_label_message:
            dico_label_message[label] = [data_message[i][2]]
        else:
            dico_label_message[label].append(data_message[i][2])
    return dico_label_message


stop_words_fr = [
    "a",
    "à",
    "abord",
    "afin",
    "ah",
    "ai",
    "ainsi",
    "allez",
    "am",
    "an",
    "and",
    "as",
    "au",
    "aucun",
    "aucune",
    "aussi",
    "autre",
    "avant",
    "avec",
    "avez",
    "b",
    "bah",
    "beaucoup",
    "bien",
    "bof",
    "bon",
    "c",
    "car",
    "ce",
    "ces",
    "cette",
    "ch",
    "chez",
    "ci",
    "comme",
    "comment",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "est",
    "et",
    "etant",
    "etre",
    "eu",
    "eux",
    "exemple",
    "fait",
    "faire",
    "faites",
    "faut",
    "fois",
    "font",
    "je",
    "jusqu",
    "la",
    "le",
    "les",
    "lors",
    "lui",
    "ma",
    "maintenant",
    "mais",
    "me",
    "merci",
    "mes",
    "moi",
    "mon",
    "meme",
    "ne",
    "nous",
    "ou",
    "où",
    "par",
    "parce",
    "pas",
    "peu",
    "plus",
    "pour",
    "pourquoi",
    "qu",
    "que",
    "quel",
    "quelle",
    "quels",
    "qui",
    "sa",
    "sans",
    "se",
    "si",
    "son",
    "sont",
    "sous",
    "sua",
    "sur",
    "ta",
    "tandis",
    "te",
    "tes",
    "toi",
    "ton",
    "tous",
    "tout",
    "très",
    "un",
    "une",
    "v",
    "vous",
    "y",
    "à",
    "également",
    "savoir",
    "seulement",
    "auprès",
    "également",
    "bien",
    "jusqu'à",
    "sauf",
    "dès",
    "partout",
    "entre",
    "c'est",
    "à",
    "cette",
    "celles",
    "celui",
    "celle",
    "ceux",
    "cela",
    "dailleurs",
    "tout",
    "dont",
    "ils",
    "elles",
    "nous",
    "bonjour",
    "bonjoir",
    "merci",
    "bonsoir",
    "rebonjour",
    "vous",
    "ici",
    "bjr",
    "mervi",
    "il",
    "elle",
    "on",
    "ça",
    "depuis",
    "remercie",
    "mercii",
    "erci",
    "oui",
    "ok",
    "bonjoue",
]


# with KeyBert


def keyword_with_keybert(dico_label_message, save=False):
    kw_model = KeyBERT(model=model)

    dico_label_keyword_bert = defaultdict(list)
    for cluster, phrases in dico_label_message.items():
        if cluster != -1:
            combined_text = " ".join(phrases)
            combined_text_petit = combined_text.lower()
            keywords = kw_model.extract_keywords(
                combined_text_petit,
                keyphrase_ngram_range=(1, 1),
                stop_words=stop_words_fr,
                top_n=7,
                # use_mmr=True,
                diversity=0.5,
            )
            keywords_only = [kw[0] for kw in keywords]
            dico_label_keyword_bert[cluster] = keywords_only

    if save:
        with open(
            "../keywords/dico_label_keyword_bert_hdbscan_dim10_eps0.2_min200.pkl", "wb"
        ) as f:
            pickle.dump(dico_label_keyword_bert, f)

    return dico_label_keyword_bert


# with TF-iDF


def keyword_with_tfidf(dico_label_message, save=False):
    vectorizer = TfidfVectorizer(
        stop_words=stop_words_fr, max_df=0.6, ngram_range=(1, 2)
    )

    documents = []
    dico_documents = {}

    index_docs = 0
    for cluster, sentence in dico_label_message.items():
        text_cluster = " ".join(sentence)
        text_cluster_petit = text_cluster.lower()
        documents.append(text_cluster_petit)
        dico_documents[(cluster, index_docs)] = text_cluster_petit
        index_docs += 1

    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    dico_label_keywords_tfidf = defaultdict(list)
    for tuple_cluster_index, doc in dico_documents.items():
        cluster, doc_idx = tuple_cluster_index
        if cluster != -1:
            sorted_items = np.argsort(tfidf_matrix[doc_idx].toarray()).flatten()[::-1]

            liste_keyword_one_label = []
            for idx in sorted_items[:7]:  # Number of keyword to extract
                liste_keyword_one_label.append(feature_names[idx])
            dico_label_keywords_tfidf[cluster] = liste_keyword_one_label

    if save:
        with open(
            "../keywords/dico_label_keyword_tfidf_hdbscan_dim10_eps0.2_min200.pkl", "wb"
        ) as f:
            pickle.dump(dico_label_keywords_tfidf, f)

    return dico_label_keywords_tfidf


# with TF-IDF et KeyBERT :


# Using TF-IDF to obtain candidate terms
def get_tfidf_candidates(dico_label_message, top_n=10):
    tfidf_candidates = defaultdict(list)
    vectorizer = TfidfVectorizer(
        stop_words=stop_words_fr, max_df=0.6, ngram_range=(1, 1)
    )

    documents = []
    dico_documents = {}

    index_docs = 0
    for cluster, sentence in dico_label_message.items():
        text_cluster = " ".join(sentence)
        text_cluster_petit = text_cluster.lower()
        documents.append(text_cluster_petit)
        dico_documents[(cluster, index_docs)] = text_cluster_petit
        index_docs += 1

    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    for tuple_cluster_index, doc in dico_documents.items():
        cluster, doc_idx = tuple_cluster_index
        candidate_1_cluster = []

        sorted_items = np.argsort(tfidf_matrix[doc_idx].toarray()).flatten()[::-1]

        for idx in sorted_items[:top_n]:
            candidate_1_cluster.append(feature_names[idx])
        tfidf_candidates[cluster] = candidate_1_cluster
    return tfidf_candidates, dico_documents


# Refinement with KeyBERT
def keywords_with_tfidf_and_keybert(
    dico_label_message, top_n=5, tfidf_top_n=10, save=False
):
    tfidf_candidates, dico_label_doc = get_tfidf_candidates(
        dico_label_message, top_n=tfidf_top_n
    )

    dico_label_final_keywords = {}

    kw_model = KeyBERT()

    for tuple_cluster_index, doc in dico_label_doc.items():
        cluster, doc_idx = tuple_cluster_index
        candidates = tfidf_candidates[cluster]
        keywords = kw_model.extract_keywords(
            doc,
            candidates=candidates,
            keyphrase_ngram_range=(1, 1),
            top_n=top_n,
            stop_words=stop_words_fr,
        )
        keywords_only = [kw[0] for kw in keywords]
        dico_label_final_keywords[cluster] = keywords_only

    if save:
        with open(
            "../keywords/dico_label_keyword_bert_tfidf_hdbscan_dim10_eps0.2_min200.pkl",
            "wb",
        ) as f:
            pickle.dump(dico_label_keyword_bert_tfidf, f)

    return dico_label_final_keywords


# random verification :


with open(
    "../keywords/dico_label_keyword_tfidf_hdbscan_dim10_eps0.2_min200.pkl", "rb"
) as f:
    dico_label_keywords_tfidf = pickle.load(f)

with open(
    "../keywords/dico_label_keyword_bert_hdbscan_dim10_eps0.2_min200.pkl", "rb"
) as f:
    dico_label_keyword_bert = pickle.load(f)

with open(
    "../keywords/dico_label_keyword_bert_tfidf_hdbscan_dim10_eps0.2_min200.pkl", "rb"
) as f:
    dico_label_keyword_bert_tfidf = pickle.load(f)


def pick_sentence_random(
    labels,
    data_message,
    label,
):
    label_pick = 0
    while label_pick != label:
        index_pick_random = np.random.randint(0, len(data_message))
        label_pick = labels[index_pick_random]
    print(f"\nLabel : {labels[index_pick_random]}\n")
    print(f"Sentence : {data_message[index_pick_random][2]}\n")
    print(f"Mots clefs TF associé : {dico_label_keywords_tfidf[label]}")
    print(f"Mots clefs BERT associé : {dico_label_keyword_bert[label]}")
    print(f"Mots clefs TF+BERT associé : {dico_label_keyword_bert_tfidf[label]}")
    print("")


label_random = np.random.randint(0, len(np.unique(labels_hdbscan)))
pick_sentence_random(labels_hdbscan, data_message=data_message, label=label_random)


def display_all_keywords(labels):
    labels_unique = np.unique(labels)
    for label in labels_unique:
        # print(label)
        if label != -1:
            print(f"\nLabel {label} :")
            print(f"Mots clefs TF associé : {dico_label_keywords_tfidf[label]}")
            print(f"Mots clefs BERT associé : {dico_label_keyword_bert[label]}")
            print(
                f"Mots clefs TF+BERT associé : {dico_label_keyword_bert_tfidf[label]}"
            )
            print("")


display_all_keywords(labels_hdbscan)
