import os
from tqdm import tqdm

seeds = [1,2,3,4,5]
nr_test_actions = [101,1,20,50]
modes = ["o", "s", "os", "or"]
topk_objects = [10,50,100]
topk_scenes = [3,5,10]
aggregates = ['simple', 'normalized', 'weighted', 'combined', 'paired']
topk_objsce = [10,20,50,75,100,300,500,700,1000]

for t in tqdm(nr_test_actions, desc = "nr_test_actions"):
    for s in tqdm(seeds, desc = "seeds"):
        for m in tqdm(modes, desc = "modes"):
            if m == "o":
                for kobj in topk_objects:
                    os.system(f"python zero-shot-actions.py -s {s} -t {t} -m {m} --kobj {kobj}")
            elif m == "s":
                for ksce in topk_scenes:
                    os.system(f"python zero-shot-actions.py -s {s} -t {t} -m {m} --ksce {ksce}")
            elif m == "os":
                for aggregate in aggregates:
                    if aggregate in ['simple', 'normalized', 'weighted']:
                        for kobj in topk_objects:
                            for ksce in topk_scenes:
                                os.system(f"python zero-shot-actions.py -s {s} -t {t} -m {m} -a {aggregate} --kobj {kobj} --ksce {ksce}")
                    elif aggregate in ["combined", "paired"]:
                        for kobjsce in topk_objsce:
                            os.system(f"python zero-shot-actions.py -s {s} -t {t} -m {m} -a {aggregate} --kobjsce {kobjsce}")
            elif m == "or":
                for kobj in topk_objects:
                    for ksce in topk_scenes:
                        os.system(f"python zero-shot-actions.py -s {s} -t {t} -m {m} --kobj {kobj} --ksce {ksce}")

        if t == 101: # we don't need to iterate over seeds when we're selecting all the actions
            break
