import os
from tqdm import tqdm

configfiles = ["config/ucf-101-fasttext.config", "config/ucf-101-sbert.config"]

nr_test_actions = [101,25,50]

seeds = [1,2,3,4,5,6,7,8,9,10]

modes = ["o", "s", "os", "or"]
topk_objects = [10,50,100]
topk_scenes = [3,5,10]
aggregates = ['simple', 'compositions']
topk_objsce = [100,250,500]

lambdas = [0.75, 1]
languages = ["English"]
xdiscrs = [0]
adiscrs = [0]
store_predss = [0]

nr_test_actions = [25]
topk_objects = [100]
topk_scenes = [5]
topk_objsce = [250]

def grid(   configfiles,
            nr_test_actions,
            seeds,
            modes,
            topk_objects,
            topk_scenes,
            aggregates,
            topk_objsce,
            lambdas,
            languages,
            xdiscrs,
            adiscrs,
            store_predss):
    for c in tqdm(configfiles, desc = "configfiles"):
        for t in tqdm(nr_test_actions, desc = "nr_test_actions"):
            for s in tqdm(seeds, desc = "seeds"):
                for store_preds in store_predss:
                    for l in languages:
                        for xdiscr in xdiscrs:
                            for adiscr in adiscrs:
                                for m in tqdm(modes, desc = "modes"):
                                    if m == "o":
                                        for kobj in topk_objects:
                                            os.system(f"python zero-shot-actions.py -c {c} -s {s} -t {t} -m {m} -l {l} --kobj {kobj} --xdiscr {xdiscr} --adiscr {adiscr} --store_preds {store_preds}")
                                    elif m == "s":
                                        for ksce in topk_scenes:
                                            os.system(f"python zero-shot-actions.py -c {c} -s {s} -t {t} -m {m} -l {l} --ksce {ksce} --xdiscr {xdiscr} --adiscr {adiscr} --store_preds {store_preds}")
                                    elif m == "os":
                                        for aggregate in aggregates:
                                            if aggregate in ['simple', 'normalized', 'weighted']:
                                                for kobj in topk_objects:
                                                    for ksce in topk_scenes:
                                                        os.system(f"python zero-shot-actions.py -c {c} -s {s} -t {t} -m {m} -l {l} -a {aggregate} --kobj {kobj} --ksce {ksce} --xdiscr {xdiscr} --adiscr {adiscr} --store_preds {store_preds}")
                                            elif aggregate in ["combined", "compositions"]:
                                                for kobjsce in topk_objsce:
                                                    if aggregate == "compositions":
                                                        if xdiscr == 1:
                                                            break
                                                        for lam in lambdas:
                                                            os.system(f"python zero-shot-actions.py -c {c} -s {s} -t {t} -m {m} -l {l} -a {aggregate} --kobjsce {kobjsce} --xdiscr {xdiscr} --adiscr {adiscr} --lambda {lam} --store_preds {store_preds}")
                                                    else:
                                                        os.system(f"python zero-shot-actions.py -c {c} -s {s} -t {t} -m {m} -l {l} -a {aggregate} --kobjsce {kobjsce} --xdiscr {xdiscr} --adiscr {adiscr} --store_preds {store_preds}")
                                    elif m == "or":
                                        for kobj in topk_objects:
                                            for ksce in topk_scenes:
                                                os.system(f"python zero-shot-actions.py -c {c} -s {s} -t {t} -m {m} -l {l} --kobj {kobj} --ksce {ksce} --xdiscr {xdiscr} --adiscr {adiscr} --store_preds {store_preds}")

                if t == 10 or t==21 or t == 101 or t == 400: # we don't need to iterate over seeds when we're selecting all the actions
                    break

# # make results for UCF-101
grid(configfiles, nr_test_actions, seeds, modes, topk_objects, topk_scenes, aggregates, topk_objsce, lambdas, languages, xdiscrs, adiscrs, store_predss)


# # # make results for kinetics
configfiles = ["config/kinetics-sbert.config", "config/kinetics-fasttext.config"]
nr_test_actions = [400,25,100]
topk_objects = [100]
topk_scenes = [5]
topk_objsce = [250]
grid(configfiles, nr_test_actions, seeds, modes, topk_objects, topk_scenes, aggregates, topk_objsce, lambdas, languages, xdiscrs, adiscrs, store_predss)




# # # make results for ucf-sports
configfiles = ["config/ucf-sports-sbert.config", "config/ucf-sports-fasttext.config"]
modes = ["o", "s", "os"]
topk_objsce = [10,50,100,250,500]
nr_test_actions = [10]
store_predss=[1]
grid(configfiles, nr_test_actions, seeds, modes, topk_objects, topk_scenes, aggregates, topk_objsce, lambdas, languages, xdiscrs, adiscrs, store_predss)


# # # make results for j-hmdb
configfiles = ["config/j-hmdb-sbert.config", "config/j-hmdb-fasttext.config"]
modes = ["o", "s", "os"]
topk_objsce = [10,50,100,250,500]
nr_test_actions = [21]
store_predss=[1]
grid(configfiles, nr_test_actions, seeds, modes, topk_objects, topk_scenes, aggregates, topk_objsce, lambdas, languages, xdiscrs, adiscrs, store_predss)
