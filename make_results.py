import os

seeds = [1,25]
nr_test_actions = [1,20,50,101]
modes = ["o", "s", "os"]
topk_objects = [10,50,100]
topk_scenes = [1,5,10]


for t in nr_test_actions:
    for s in seeds:
        for m in modes:
            if m == "o":
                for kobj in topk_objects:
                    os.system(f"python zero-shot-actions.py -s {s} -t {t} -m {m} --kobj {kobj}")
            elif m == "s":
                for ksce in topk_scenes:
                    os.system(f"python zero-shot-actions.py -s {s} -t {t} -m {m} --ksce {ksce}")
            elif m == "os":
                for kobj in topk_objects:
                    for ksce in topk_scenes:
                        os.system(f"python zero-shot-actions.py -s {s} -t {t} -m {m} --kobj {kobj} --ksce {ksce}")
        if t == 101: # we don't need to iterate over seeds when we're selecting all the actions
            break
