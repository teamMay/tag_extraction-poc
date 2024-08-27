import numpy as np
import pandas as pd
from dateutil.parser import parse


def pd_to_np(path_csv):
    dataframe = pd.read_csv(path_csv)

    dataframe["firstMessage"] = pd.to_datetime(dataframe["firstMessage"].apply(parse))

    data_numpy = dataframe[
        ["id", "category_id", "firstMessagesBatch", "firstMessage", "subject_id"]
    ].to_numpy()

    print(f"taille des données initialles : {data_numpy.shape}")

    return data_numpy


# functions to see errors, data in this table sent by a nurse
def recherche_erreur(data):
    tab_data_mieux = []
    tab_erreur = []
    for id, cat, message, firstmessage, subject_id in data:
        if isinstance(message, str):
            if "infirmière puéricultrice chez May" in message:
                tab_erreur.append([id, cat, message, firstmessage, subject_id])
            else:
                tab_data_mieux.append([id, cat, message, firstmessage, subject_id])
    return np.array(tab_data_mieux), tab_erreur


def str_to_int(value):
    try:
        return int(float(value))
    except ValueError:
        return -5


def mettre_cat_en_int(tableau_donnee_ancien, tableau_donnee_nouveau):
    for i in range(len(tableau_donnee_ancien)):
        int_cat = str_to_int(tableau_donnee_ancien[i][1])
        tableau_donnee_nouveau[i][1] = int_cat
        int_sub = str_to_int(tableau_donnee_ancien[i][4])
        tableau_donnee_nouveau[i][4] = int_sub

    return tableau_donnee_nouveau


def pipeline_importation(path_csv):
    data_numpy = pd_to_np(path_csv)
    data_numpy_clean, tableau_des_erreurs = recherche_erreur(data_numpy)

    print(f"nbr d'erreurs enlevées : {len(tableau_des_erreurs)}")
    print(f"taille nouvelle des données : {data_numpy_clean.shape}")

    data_numpy_clean_copy = data_numpy_clean.copy()
    data_numpy_clean_final = mettre_cat_en_int(data_numpy_clean, data_numpy_clean_copy)

    print(f"taille nouvelle des données : {data_numpy_clean_final.shape}")

    return data_numpy_clean_final


path_csv = "../data/messages_and_categories_ipde_0824.csv"

data_numpy_clean_final = pipeline_importation(path_csv)

np.save(
    "../data/data_numpy_msg_and_categories_ipde_0824.npy",
    data_numpy_clean_final,
)
