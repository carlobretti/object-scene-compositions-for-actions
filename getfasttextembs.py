#!/usr/bin/env python
# coding: utf-8



import os
os.system('pip install fasttext')

import fasttext.util
import fasttext
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import cdist


langs = {"en":"English"}
# langs = {"en":"English", "af":"Afrikaans", "nl":"Dutch", "fr":"French", "pt":"Portuguese"}

# download pretrained ft models for languages of choice
for lang_short in langs.keys():
    fasttext.util.download_model(lang_short, if_exists='ignore')  # English




def custom_get_sentence_vector(ft, text):
    # custom get sentence vector to avoid adding the EOS token like in ft.get_sentence_vector
    vecs = []
    norms = []
    for token in text.split():
        vec = ft.get_word_vector(token)
        vecs.append(vec)
        norms.append(LA.norm(vec))
    avged = np.mean([vec/norm if norm>0 else vec for vec, norm in zip(vecs,norms)], axis = 0)
    return avged

# generate and store embeddings for all labels for the five datasets
for dataset in ["imagenet", "places-365", "ucf-101", "kinetics", "ucf-sports", "j-hmdb"]:
    ds_folder = f"data/{dataset}/"
    ds_wd_folder = ds_folder+"words/"
    ds_ft_folder = ds_folder+"fasttext/"

    os.system(f"mkdir -p {ds_ft_folder}")

    dataset = "imagenet12988" if dataset == "imagenet" else dataset.replace("-", "")

    for lang_short, lang in langs.items():
        with open(ds_wd_folder+f"{dataset}-words-{lang}.txt", 'r') as f:
            labels = f.readlines()
        labels = [label.strip().replace("_", " ") for label in labels]
        ft = fasttext.load_model(f'cc.{lang_short}.300.bin')


        embeddings = np.array([custom_get_sentence_vector(ft, label) for label in labels])
        np.save(ds_ft_folder+f"fasttext-{dataset}-{lang}.npy", embeddings)


# generate and store embeddings for object-scene label pairs for the three datasets
# for lang_short, lang in langs.items():
#
#     imagenet_vecs = np.load(f"data/imagenet/fasttext/fasttext-imagenet12988-{lang}.npy")
#     places_vecs = np.load(f"data/places-365/fasttext/fasttext-places365-{lang}.npy")
#
#     ds_ft_folder = "data/imagenet_places/fasttext/"
#     os.system(f"mkdir -p {ds_ft_folder}")
#
#     # averaging the embedding for object and for scene without dividing by norm
#     embeddings = np.array([np.mean([places_vec, imagenet_vec], axis = 0) for imagenet_vec in imagenet_vecs for places_vec in places_vecs])
#
#     np.save(ds_ft_folder+f"fasttext-imagenet12988places365pairs-{lang}.npy", embeddings)


#
# Pair-wise similarity between (action and scene), (action and object), (object and object), (scene and scene), (scene and object) word embeddings.
#
def wtv_mapping(wtv1, wtv2):
    wtvmap = cdist(wtv1, wtv2, metric = "cosine")
    return 1 - wtvmap

for ds1, ds2 in [("imagenet", "places-365"),
                 ("ucf-101", "places-365"),
                 ("ucf-101", "imagenet"),
                 ("kinetics", "places-365"),
                 ("kinetics", "imagenet"),
                 ("ucf-sports", "places-365"),
                 ("ucf-sports", "imagenet"),
                 ("j-hmdb", "places-365"),
                 ("j-hmdb", "imagenet"),
                 ("imagenet", "imagenet"),
                 ("places-365", "places-365")]:
    ds1_ft_folder = f"data/{ds1}/fasttext/"
    ds2_ft_folder = f"data/{ds2}/fasttext/"

    ds1 = "imagenet12988" if ds1 == "imagenet" else ds1.replace("-", "")
    ds2 = "imagenet12988" if ds2 == "imagenet" else ds2.replace("-", "")

    for lang_short, lang in langs.items():
        ds1_emb = np.load(ds1_ft_folder+f"fasttext-{ds1}-{lang}.npy")
        ds2_emb = np.load(ds2_ft_folder+f"fasttext-{ds2}-{lang}.npy")

        emb2emb = wtv_mapping(ds1_emb, ds2_emb)

        corr = {"imagenet12988":"o", "places365":"s", "ucf101":"a", "kinetics":"a", "ucfsports":"a", "jhmdb":"a"}

        np.save(ds1_ft_folder+f"{corr[ds1]}2{corr[ds2]}_ft_{ds2}_{lang}.npy", emb2emb)



# computing pairwise similarity for object-scene pairs and actions
for dataset in ["kinetics", "ucf-101", "ucf-sports", "j-hmdb"]:
    for lang_short, lang in langs.items():
        a = dataset.replace("-", "") if (dataset == "ucf-101" or dataset == "ucf-sports" or dataset == "j-hmdb") else dataset
        ds1_emb = np.load(f"data/{dataset}/fasttext/fasttext-{a}-{lang}.npy")
        ds2_emb = np.load(f"data/imagenet_places/fasttext/fasttext-imagenet12988places365pairs-{lang}.npy")


        emb2emb = wtv_mapping(ds2_emb, ds1_emb)
        np.save(f"data/{dataset}/fasttext/a2ospairs_ft_imagenet12988places365pairs_{lang}.npy", emb2emb.T)
