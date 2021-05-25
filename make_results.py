import os
from tqdm import tqdm

seeds = [1,2,3,4,5]
nr_test_actions = [101,20,50]

modes = ["o", "s", "os", "or"]
topk_objects = [10,50,100]
topk_scenes = [3,5,10]
# aggregates = ['simple', 'normalized', 'weighted', 'combined', 'paired']
# aggregates = ['weighted', 'combined', 'paired']
aggregates = ['weighted', 'paired']
topk_objsce = [100,200,300,400,500]


lambdas = [0.75, 1]
# xdiscrs = [0,1]
# adiscrs = [0,1]
languages = ["English", "English-Dutch"]
xdiscrs = [0]
adiscrs = [0]

for t in tqdm(nr_test_actions, desc = "nr_test_actions"):
    for s in tqdm(seeds, desc = "seeds"):
        for l in languages:
            for xdiscr in xdiscrs:
                for adiscr in adiscrs:
                    for m in tqdm(modes, desc = "modes"):
                        if m == "o":
                            for kobj in topk_objects:
                                os.system(f"python zero-shot-actions.py -s {s} -t {t} -m {m} -l {l} --kobj {kobj} --xdiscr {xdiscr} --adiscr {adiscr}")
                        elif m == "s":
                            for ksce in topk_scenes:
                                os.system(f"python zero-shot-actions.py -s {s} -t {t} -m {m} -l {l} --ksce {ksce} --xdiscr {xdiscr} --adiscr {adiscr}")
                        elif m == "os":
                            for aggregate in aggregates:
                                if aggregate in ['simple', 'normalized', 'weighted']:
                                    for kobj in topk_objects:
                                        for ksce in topk_scenes:
                                            os.system(f"python zero-shot-actions.py -s {s} -t {t} -m {m} -l {l} -a {aggregate} --kobj {kobj} --ksce {ksce} --xdiscr {xdiscr} --adiscr {adiscr}")
                                elif aggregate in ["combined", "paired"]:
                                    for kobjsce in topk_objsce:
                                        if aggregate == "paired":
                                            if xdiscr == 1:
                                                break
                                            for lam in lambdas:
                                                os.system(f"python zero-shot-actions.py -s {s} -t {t} -m {m} -l {l} -a {aggregate} --kobjsce {kobjsce} --xdiscr {xdiscr} --adiscr {adiscr} --lambda {lam}")
                                        else:
                                            os.system(f"python zero-shot-actions.py -s {s} -t {t} -m {m} -l {l} -a {aggregate} --kobjsce {kobjsce} --xdiscr {xdiscr} --adiscr {adiscr}")
                        elif m == "or":
                            for kobj in topk_objects:
                                for ksce in topk_scenes:
                                    os.system(f"python zero-shot-actions.py -s {s} -t {t} -m {m} -l {l} --kobj {kobj} --ksce {ksce} --xdiscr {xdiscr} --adiscr {adiscr}")

        if t == 101: # we don't need to iterate over seeds when we're selecting all the actions
            break
