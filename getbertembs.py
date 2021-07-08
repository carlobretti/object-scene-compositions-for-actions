import os
os.system('pip install -U sentence-transformers')



from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
import numpy as np



model_name = "paraphrase-MiniLM-L6-v2" # best model in terms of tradeoff speed performance

# model_name = "distiluse-base-multilingual-cased-v1" # model to be used for multilingual
# langs = {"en":"English", "af":"Afrikaans", "nl":"Dutch", "fr":"French", "pt":"Portuguese"}

model = SentenceTransformer(model_name)

langs = {"en":"English"}


# generate and store embeddings for all labels for all the datasets

for dataset in ["imagenet", "places-365", "ucf-101", "kinetics", "ucf-sports","j-hmdb"]:
    ds_folder = f"data/{dataset}/"
    ds_wd_folder = ds_folder+"words/"
    ds_sbert_folder = ds_folder+"sbert/"


    os.system(f"mkdir -p {ds_sbert_folder}")

    dataset = "imagenet12988" if dataset == "imagenet" else dataset.replace("-", "")

    for lang_short, lang in langs.items():
        with open(ds_wd_folder+f"{dataset}-words-{lang}.txt", 'r') as f:
            labels = f.readlines()

        labels = [label.strip().replace("_", " ") for label in labels]
        embeddings = model.encode(labels)
        np.save(ds_sbert_folder+f"sbert-{dataset}-{lang}.npy", embeddings)



# generate and store embeddings for object-scene label pairs

for lang_short, lang in langs.items():

    imagenet_vecs = np.load(f"data/imagenet/sbert/sbert-imagenet12988-{lang}.npy")
    places_vecs = np.load(f"data/places-365/sbert/sbert-places365-{lang}.npy")

    ds_sbert_folder = "data/imagenet_places/sbert/"
    os.system(f"mkdir -p {ds_sbert_folder}")

    # averaging the embedding for object and for scene without dividing by norm
    embeddings = np.array([np.mean([places_vec, imagenet_vec], axis = 0) for imagenet_vec in imagenet_vecs for places_vec in places_vecs])

    np.save(ds_sbert_folder+f"sbert-imagenet12988places365pairs-{lang}.npy", embeddings)



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
    ds1_sbert_folder = f"data/{ds1}/sbert/"
    ds2_sbert_folder = f"data/{ds2}/sbert/"

    ds1 = "imagenet12988" if ds1 == "imagenet" else ds1.replace("-", "")
    ds2 = "imagenet12988" if ds2 == "imagenet" else ds2.replace("-", "")

    for lang_short, lang in langs.items():
        ds1_emb = np.load(ds1_sbert_folder+f"sbert-{ds1}-{lang}.npy")
        ds2_emb = np.load(ds2_sbert_folder+f"sbert-{ds2}-{lang}.npy")

        emb2emb = wtv_mapping(ds1_emb, ds2_emb)

        corr = {"imagenet12988":"o", "places365":"s", "ucf101":"a", "kinetics":"a", "ucfsports":"a", "jhmdb":"a"}

        np.save(ds1_sbert_folder+f"{corr[ds1]}2{corr[ds2]}_sbert_{ds2}_{lang}.npy", emb2emb)


# computing pairwise similarity for object-scene pairs and actions
for dataset in ["kinetics", "ucf-101", "ucf-sports", "j-hmdb"]:
    for lang_short, lang in langs.items():
        a = dataset.replace("-", "") if (dataset == "ucf-101" or dataset == "ucf-sports" or dataset == "j-hmdb") else dataset
        ds1_emb = np.load(f"data/{dataset}/sbert/sbert-{a}-{lang}.npy")
        ds2_emb = np.load(f"data/imagenet_places/sbert/sbert-imagenet12988places365pairs-{lang}.npy")


        emb2emb = wtv_mapping(ds2_emb, ds1_emb)
        np.save(f"data/{dataset}/sbert/a2ospairs_sbert_imagenet12988places365pairs_{lang}.npy", emb2emb.T)
