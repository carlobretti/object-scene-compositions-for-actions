#
# Zero-shot action classification using standard action to object semantics.
#

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import configparser
from   datetime import datetime
from   six.moves import cPickle as pickle
from   scipy.spatial.distance import cosine
from   scipy.stats import beta
from   nltk.corpus import wordnet as wn
from   sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from collections import defaultdict

#
# Obtain action score for scenes and objects
#
def video_action_score(x_scores, action_x_scores, top_x):
    return np.dot(x_scores[top_x], action_x_scores[top_x])

#
# Obtain action score for scenes and objects pairs
# here we have to assume that the pairs were computed by taking the first object and combining it with all scenes, then the second object, and combining it with all the scene
# i.e. the objscepairs were created with [objlabel+" "+scelabel for objlabel in objlabels for scelabel in scelabels] as in Fasttext for all labels.ipynb
#
def video_action_score_paired(obj_scores, sce_scores, action_x_scores, top_objscepairs):
    n_s = len(sce_scores)
    oidxs, sidxs = zip(*[divmod(top_objscepair, n_s) for top_objscepair in top_objscepairs])
    oidxs = list(oidxs)
    sidxs = list(sidxs)
    pairs_scores = np.multiply(obj_scores[oidxs], sce_scores[sidxs])


    return np.dot(pairs_scores, action_x_scores[top_objscepairs])

#
# # attempt at normalizing?
# # divide by sum of probability of the topk most related xs?
# # # divide by sum of probability of the topk most prominent xs (so for each video the most probable objects?)?
# def video_action_score_normalized(x_scores, action_x_scores, top_x):
#     return np.dot(x_scores[top_x]/np.sum(x_scores[top_x]), action_x_scores[top_x])
# # # attempt

#
# Zero-shot action classification class.
#
class ZeroShotActionClassifier(object):

    #
    # Initialize by setting paths and loading data.
    #
    def __init__(self, configfile):
        # Parse configfile.
        parser = configparser.ConfigParser()
        parser.read(configfile)

        # Directories and videos.
        self.objfeatdir = parser.get('actions', 'objectscores')
        self.scefeatdir = parser.get('actions', 'scenescores')
        self.videos  = parser.get('actions', 'videofile')
        self.ext     = parser.get('actions', 'objscefilename')
        self.videos  = [l.strip().split()[0] for l in open(self.videos)]
        self.videos  = np.array(self.videos)

        # Actions.
        self.actions = np.array([l.split("/")[0] for l in self.videos])
        self.actions, self.aidxs = np.unique(self.actions, return_inverse=True)

        # Action to object fasttext.
        languages = parser.get('actions', 'fasttext_languages')
        languages = languages.split(", ")
        self.a2o_ft = {}
        self.a2s_ft = {}
        self.a2ospairs_ft = {}
        self.o2o_ft = {}
        self.s2s_ft = {}
        for language in languages:
            a2oftfile = parser.get('actions', f'a2oft_{language}')
            self.a2o_ft[language] = np.load(a2oftfile)
            a2sftfile = parser.get('actions', f'a2sft_{language}')
            self.a2s_ft[language] = np.load(a2sftfile)
            a2ospairsftfile = parser.get('actions', f'a2ospairsft_{language}')
            self.a2ospairs_ft[language] = np.load(a2ospairsftfile)
            o2oftfile = parser.get('actions', f'o2oft_{language}')
            self.o2o_ft[language] = np.load(o2oftfile)
            s2sftfile = parser.get('actions', f's2sft_{language}')
            self.s2s_ft[language] = np.load(s2sftfile)

    #
    # Predict the class of each test action.
    #
    def predict(self, seed, nr_test_actions, topk_objects, topk_scenes, topk_objsce, xdiscr, adiscr, aggregate, mode, languages):
        # Set seed and randomly select test actions.
        np.random.seed(seed)
        aa = np.arange(len(self.actions))
        test_actions = np.random.choice(aa, nr_test_actions, replace=False)
        test_actions = np.sort(test_actions)

        # Gather test indices.
        teidxs = []
        for test_action in test_actions:
            teidxs.append(np.where(self.aidxs == test_action)[0])
        teidxs = np.concatenate(teidxs).astype(int)

        # Find video and label subsets.
        self.videos = self.videos[teidxs]
        self.aidxs  = self.aidxs[teidxs]

        a2xscores = defaultdict(list)
        top_x = defaultdict(list)
        # all_records = []

        dweights = [xdiscr, adiscr]

        for i in range(len(test_actions)):
            if mode in ["o", "os", "or"] and aggregate != "paired":
                a2os = self.a2x_scores(test_actions, i, languages, dweights, "o")
                a2xscores["a2oscores"].append(a2os)
                oidxs = np.argsort(a2os)[-topk_objects:]
                top_x["top_objects"].append(oidxs)

            if mode in ["s", "os", "or"] and aggregate != "paired":
                a2ss = self.a2x_scores(test_actions, i, languages, dweights, "s")
                a2xscores["a2sscores"].append(a2ss)
                sidxs = np.argsort(a2ss)[-topk_scenes:]
                top_x["top_scenes"].append(sidxs)

            if mode == "os" and aggregate == "combined":
                a2oss = np.concatenate((a2os, a2ss))
                a2xscores["a2osscores"].append(a2oss)
                osidxs = np.argsort(a2oss)[-topk_objsce:]
                top_x["top_objsce"].append(osidxs)

            if mode == "os" and aggregate == "paired":
                a2ospairss = self.a2x_scores(test_actions, i, languages, dweights, "osp")
                a2xscores["a2ospairscores"].append(a2ospairss)
                osidxs = np.argsort(a2ospairss)[-topk_objsce:]
                top_x["top_objscepairs"].append(osidxs)



            # # Save top scoring objects and scenes for each action.
            # wnids = [c.strip() for c in open("data/imagenet/wnids-12988.txt")]
            # scenes = [c.strip() for c in open("data/places-365/words/places365-words-English.txt")]
            # allsidxs = np.argsort(a2ss)[::-1]
            # alloidxs = np.argsort(a2os)[::-1]
            # record = {}
            # record["action_name"] = self.actions[test_actions[i]]
            # for j in range(5):
            #     record[f"top_scene_{j}"] = scenes[allsidxs[j]]
            #     record[f"top_object_{j}"] = wn.synset_from_pos_and_offset('n',int(wnids[alloidxs[j]][1:])).lemma_names()[0]

            # all_records.append(record)

        # os.system("mkdir -p results")
        # pd.DataFrame(all_records).sort_index(axis=1).to_csv(f"results/top_object_scene_action.csv")

        # Gather video predictions.
        predictions = np.zeros(len(self.videos), dtype=int)

        # oracle
        if mode == "or":
            results_peraction_path = "results/accuracies_per_action.csv"
            accs_df = pd.read_csv(results_peraction_path, index_col = 0)
        # oracle
        for i in range(len(self.videos)):
            print("Video %d/%d\r" %(i+1, len(self.videos)), end="")
            sys.stdout.flush()

            # Load object/scene scores.
            vidfile = self.videos[i].split("/")[-1][:-4]
            objavgfeat = np.load(self.objfeatdir + vidfile + self.ext)
            sceavgfeat = np.load(self.scefeatdir + vidfile + self.ext)

            # Score per action.
            action_scores = np.zeros(len(test_actions))
            for j in range(len(test_actions)):
                if mode == "o":
                    action_scores[j] = video_action_score(objavgfeat, a2xscores["a2oscores"][j], top_x["top_objects"][j])
                elif mode == "s":
                    action_scores[j] = video_action_score(sceavgfeat, a2xscores["a2sscores"][j], top_x["top_scenes"][j])
                elif mode == "os":
                    if aggregate == "simple":
                        # simplest average, take the two scores, add them up and divide by 2
                        action_scores[j] = (video_action_score(objavgfeat, a2xscores["a2oscores"][j], top_x["top_objects"][j]) + video_action_score(sceavgfeat, a2xscores["a2sscores"][j], top_x["top_scenes"][j]))/2
                    elif aggregate == "normalized":
                        # ""normalized"" average, take the two scores, divide them by the respective k, add them up and divide by 2
                        action_scores[j] = (video_action_score(objavgfeat, a2xscores["a2oscores"][j], top_x["top_objects"][j])/len(top_x["top_objects"][j]) + video_action_score(sceavgfeat, a2xscores["a2sscores"][j], top_x["top_scenes"][j])/len(top_x["top_scenes"][j]) )/2
                    elif aggregate == "weighted":
                        # weighted average, take the two scores, multiply them by the respective weight z, add them up
                        # don't have yet a scientific way of finding the best values for zs and zo, but ballpark estimate is that zs=0.1 and zo=0.9 works best
                        zs = 0.1
                        zo = 0.9
                        action_scores[j] = video_action_score(objavgfeat, a2xscores["a2oscores"][j], top_x["top_objects"][j])*zo + video_action_score(sceavgfeat, a2xscores["a2sscores"][j], top_x["top_scenes"][j])*zs
                    elif aggregate == "combined":
                        # unifying object and scenes in a single table
                        objsceavgfeat = np.concatenate((objavgfeat, sceavgfeat))
                        action_scores[j] = video_action_score(objsceavgfeat, a2xscores["a2osscores"][j], top_x["top_objsce"][j])
                        # action_scores[j] = video_action_score_normalized(objsceavgfeat, a2xscores["a2osscores"][j], top_x["top_objsce"][j])
                    elif aggregate == "paired":
                        action_scores[j] = video_action_score_paired(objavgfeat, sceavgfeat, a2xscores["a2ospairscores"][j], top_x["top_objscepairs"][j])


                    # # smart, semantically-aware average
                    # raise NotImplementedError
                elif mode == "or":
                    # oracle mode
                    # take a peek at the truth label of the video and pick whether to use the score from the object-only or scene-only based model
                    # this is based on the accuracy each

                    try:
                        # hacky way of doing this by hardcoding the default value for kobj and ksce
                        if accs_df[f"o_{args.aggregate}a_{args.topk_objects}kobj_5ksce_{args.topk_objsce}kobjsce_{args.xdiscr}xdiscr_{args.adiscr}adiscr_{args.language}l_accuracy"][self.aidxs[i]] >= accs_df[f"s_{args.aggregate}a_100kobj_{args.topk_scenes}ksce_{args.topk_objsce}kobjsce_{args.xdiscr}xdiscr_{args.adiscr}adiscr_{args.language}l_accuracy"][self.aidxs[i]]:
                            action_scores[j] = video_action_score(objavgfeat, a2xscores["a2oscores"][j], top_x["top_objects"][j])
                        else:
                            action_scores[j] = video_action_score(sceavgfeat, a2xscores["a2sscores"][j], top_x["top_scenes"][j])
                    except KeyError as e:
                        print(str(e))
                        raise KeyError("To use the oracle (mode or), you need to have first run both in mode s and mode o with the same kobj and ksce configuration")

            # Select highest scoring action.
            predictions[i] = np.argmax(action_scores)
        print()

        # Map predictions to correct labels.
        predictions = test_actions[predictions]
        return self.aidxs, predictions


    def a2x_discriminate(self, actions, actionindex, languages, x_priors, a2x_ft, x2x, dweights):
        # action-based discrimination prior
        if dweights[0] > 0:
            tmpa2x = a2x_ft[languages[0]].copy()
            other_actions = np.setdiff1d(actions, [actions[actionindex]])
            entropies = x_priors - np.max(tmpa2x[other_actions,:], axis=0)
            x_priors = x_priors + entropies * dweights[0]
        # x-based discrimination prior
        if dweights[1] > 0:
            entropies = x_priors - (np.mean(x2x[languages[0]], axis=0) ** 0.5)
            x_priors = x_priors + entropies * dweights[1]
        return x_priors

    #
    #
    #
    def a2x_scores(self, actions, actionindex, languages, dweights, x):
        languages = languages.split("-")
        if x == "o":
            x_priors = self.a2o_ft[languages[0]][actions[actionindex]]
            x_priors = self.a2x_discriminate(actions, actionindex, languages, x_priors, self.a2o_ft, self.o2o_ft, dweights)
        elif x == "s":
            x_priors = self.a2s_ft[languages[0]][actions[actionindex]]
            x_priors = self.a2x_discriminate(actions, actionindex, languages, x_priors, self.a2s_ft, self.s2s_ft, dweights)
        elif x == "osp":
            x_priors = self.a2ospairs_ft[languages[0]][actions[actionindex]]
            # not sure about having an object-scenepair x object-scenepair matrix of size (4m x 4m)
        # Deal with scenes/objects/actions with no (Working) word embedding.
        x_priors[np.isnan(x_priors)] = -10
        for i in range(1, len(languages)):
            if x == "o":
                new_priors = self.a2o_ft[languages[i]][actions[actionindex]]
            elif x == "s":
                new_priors = self.a2s_ft[languages[i]][actions[actionindex]]
            elif x == "osp":
                new_priors = self.a2ospairs_ft[languages[i]][actions[actionindex]]
            new_priors[np.isnan(new_priors)] = -10
            x_priors += new_priors
        #x_priors /= len(languages)
        return x_priors


#
# Parser arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot actions from objects and scenes")
    parser.add_argument("-c", dest="configfile", help="Configuration file", default="ucf-101.config", type=str)
    parser.add_argument("-t", dest="nr_test_actions", help="Number of test actions", default=50, type=int)
    parser.add_argument("--kobj", dest="topk_objects", help="Top k objects per action", default=100, type=int)
    parser.add_argument("--ksce", dest="topk_scenes", help="Top k scenes per action", default=5, type=int)
    parser.add_argument("--kobjsce", dest="topk_objsce", help="Top k objects and scenes per action", default=100, type=int)
    parser.add_argument("--xdiscr", dest="xdiscr", help="x-based (object/scene) discrimination, should be either 1 or 0", default=0, type=int)
    parser.add_argument("--adiscr", dest="adiscr", help="action-based discrimination, should be either 1 or 0", default=0, type=int)
    parser.add_argument("-s", dest="seed", help="Random seed", default=100, type=int)
    parser.add_argument("-m", dest="mode", help="Mode used, can be Objects (o); Scenes (s); Objects and scenes (os); Objects and scenes oracle (or)", default="o", type=str)
    parser.add_argument("-a", dest="aggregate", help="Way of aggregating scores used in Objects and scenes (os) mode", default="NA", type=str)
    parser.add_argument("-l", dest="language", help="Used language", default="English", type=str)
    args = parser.parse_args()
    return args

#
# Main script.
#
if __name__ == "__main__":
    # Parse arguments.
    args   = parse_args()

    if args.mode not in ["s","o", "os", "or"]:
        raise ValueError("Mode used, can be Objects (o); Scenes (s); Objects and scenes (os), Objects and scenes oracle (or)")

    if args.mode == "os":
        if args.aggregate not in ['simple', 'normalized', 'weighted', 'combined', 'paired']:
            raise ValueError("When using os mode an aggregation method needs to be specified.\n Can be 'simple', 'normalized', 'weighted', 'combined', 'paired'")


    # Initialize zero-shot classifier.
    model = ZeroShotActionClassifier(args.configfile)

    # Perform zero-shot action prediction.
    ty, tp = model.predict( seed = args.seed,
                            nr_test_actions = args.nr_test_actions,
                            topk_objects = args.topk_objects,
                            topk_scenes = args.topk_scenes,
                            topk_objsce = args.topk_objsce,
                            xdiscr = args.xdiscr,
                            adiscr = args.adiscr,
                            aggregate = args.aggregate,
                            mode = args.mode,
                            languages = args.language)

    # Print the results.
    acc = np.mean(ty == tp)

    results = { "datetime": datetime.now().isoformat(' ', 'seconds'),
                "mode":args.mode,
                "a": args.aggregate,
                "t":args.nr_test_actions,
                "kobj":args.topk_objects,
                "ksce":args.topk_scenes,
                "kobjsce": args.topk_objsce,
                "xdiscr": args.xdiscr,
                "adiscr": args.adiscr,
                "s":args.seed,
                "l":args.language,
                "acc":acc}
    print(f"Setting: [mode:{args.mode}, a: {args.aggregate} t:{args.nr_test_actions}, kobj:{args.topk_objects}, ksce:{args.topk_scenes}, kobjsce: {args.topk_objsce}, xdiscr: {args.xdiscr}, adiscr: {args.adiscr}, s:{args.seed}, l:{args.language}]: acc: {acc:.4f}")
    # print(classification_report(ty, tp, target_names = model.actions))

    results_path = "results/accuracies.csv"
    df = pd.read_csv(results_path, index_col=0) if os.path.exists(results_path) else pd.DataFrame(columns = results.keys())
    df = df.append(results, ignore_index=True)
    df.to_csv(results_path)

    # Print confusion matrix
    if args.nr_test_actions==101:
        # plotting out confusion matrices
        plt.rcParams["figure.figsize"] = (70,70)
        cm = confusion_matrix(ty, tp, normalize = "true")
        pd.DataFrame(cm, index = model.actions, columns = model.actions).to_csv(f'results/{args.mode}_{args.aggregate}a_{args.topk_objects}kobj_{args.topk_scenes}ksce_{args.topk_objsce}kobjsce_{args.xdiscr}xdiscr_{args.adiscr}adiscr_{args.language}l_confmatrix.csv')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= model.actions)
        disp.plot(xticks_rotation='vertical')
        plt.savefig(f'results/{args.mode}_{args.aggregate}a_{args.topk_objects}kobj_{args.topk_scenes}ksce_{args.topk_objsce}kobjsce_{args.xdiscr}xdiscr_{args.adiscr}adiscr_{args.language}l_confmatrix.png')


        # Store accuracy per action and top objects/scenes.
        colname = f"{args.mode}_{args.aggregate}a_{args.topk_objects}kobj_{args.topk_scenes}ksce_{args.topk_objsce}kobjsce_{args.xdiscr}xdiscr_{args.adiscr}adiscr_{args.language}l_accuracy"
        accs = np.bincount(ty[ty==tp], minlength=len(model.actions)) / np.bincount(ty, minlength=len(model.actions))
        accs = np.stack((model.actions,accs), axis = -1)
        accs = [{"action_name": line[0], colname: line[1]} for line in accs]
        accs_df = pd.DataFrame(accs)



        results_peraction_path = "results/accuracies_per_action.csv"
        if os.path.exists(results_peraction_path):
            prev_accs_df = pd.read_csv(results_peraction_path, index_col=0)
            prev_accs_df[colname] = accs_df[colname]
            prev_accs_df.to_csv(results_peraction_path)
        else:
            accs_df.to_csv(results_peraction_path)
