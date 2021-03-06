#
# Zero-shot action classification using standard action to object semantics.
#

import os
import sys
import numpy as np
import scipy as sp
import scipy.spatial.distance
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import configparser
from   datetime import datetime
from   six.moves import cPickle as pickle
from   nltk.corpus import wordnet as wn
from   sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from   collections import defaultdict
from   math import prod
from tqdm import tqdm
### FROM https://github.com/cgnorthcutt/forum-diversification/blob/master/mmr_experiment/mmr.py arXiv:2002.12457v1
### implementation of https://doi.org/10.1145/290941.291025
def mmr(pairwise_matrix, w, K, lam):
  '''
  pairwise_matrix - precomputed cosine similarity for all pairs of comments
  w - weights for each document which rank its goodness
  K - number of diverse documents to select
  lam - lambda tradeoff between score (weight) and diversity
  '''

  VERY_NEGATIVE_NUMBER = -1e10
  VERY_LARGE_NUMBER = 1e10

  # Init
  N = len(pairwise_matrix)
  mu_i = np.zeros(K, dtype=np.int) #index of selected comments
  weights = np.copy(w)
  P = np.copy(pairwise_matrix)

  #Make highest-weighted datum first selected document
  mu_i[0] = np.argmax(weights)

  #Update mask (remove doc from set) and set weight to neg inf
  weights[mu_i[0]] = VERY_NEGATIVE_NUMBER
  prev_c = np.empty(N) #Stores the max cossim across previous docs for each of the N docs
  prev_c.fill(VERY_NEGATIVE_NUMBER) #Initialization to a value that will never be max
  P[mu_i[0]].fill(VERY_LARGE_NUMBER) #effectively remove pairwise row of selected docs

  #MMR algorithm
  for k in range(1, K):

      #Reduce the computation to O(KN) instead of O(K2N) by dynamically building max.
      P_k = P[:,mu_i[k-1]] #All cosine similarities for (k-1)th doc
      prev_max_concat_new_cossim = np.column_stack((prev_c, P_k)) #shape is (N, 2)
      c = np.max(prev_max_concat_new_cossim, axis = 1) #max cosine sim b
      prev_c = c #update dynamic max cossim for each doc across all selected docs

      #Compute MMR scores for each document
      #Scores for previously selected documents will remain VERY_NEGATIVE_NUMBER
      scores = lam * weights - (1-lam) * c

      #Select document with maximum score
      mu_i[k] = np.argmax(scores)

      #Update mask (remove doc from set) and set weight to neg inf
      weights[mu_i[k]] = VERY_NEGATIVE_NUMBER
      P[mu_i[k]].fill(VERY_LARGE_NUMBER) #effectively remove pairwise row of selected doc

  return mu_i
### END FROM https://github.com/cgnorthcutt/forum-diversification/blob/master/mmr_experiment/mmr.py




#
# Obtain action score for scenes and objects
#
def video_action_score(x_scores, action_x_scores, top_x):
    return np.dot(x_scores[top_x], action_x_scores[top_x])

#
# Obtain action score for scenes and objects pairs
# here we have to assume that the pairs were computed by taking the first object and combining it with all scenes, then the second object, and combining it with all the scene
# i.e. the objscepairs were created with [objlabel+" "+scelabel for objlabel in objlabels for scelabel in scelabels] as in getfasttextembs.py and getsbertembs.py
#
def video_action_score_compositions(obj_scores, sce_scores, action_x_scores, top_objscepairs):
    n_s = len(sce_scores)


    osidxs = np.array([list(divmod(top_objscepair, n_s)) for top_objscepair in top_objscepairs])
    pairs_scores = np.multiply(obj_scores[osidxs[:,0]], sce_scores[osidxs[:,1]])
    return np.dot(pairs_scores, action_x_scores[top_objscepairs])

#
# Zero-shot action classification class.
#
class ZeroShotActionClassifier(object):

    #
    # Initialize by setting paths and loading data.
    #
    def __init__(self, configfile, mode, aggregate, xdiscr):
        # Parse configfile.
        parser = configparser.ConfigParser()
        parser.read(configfile)

        # Directories and videos.
        self.configfile = configfile.split("/")[1]
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
        self.ospairs_ft = {}
        self.o2o_ft = {}
        self.s2s_ft = {}
        # self.o2s_ft = {}
        for language in languages:
            if (mode == "o") or ((mode in ["os", "or"]) and (aggregate != "compositions")):
                a2oftfile = parser.get('actions', f'a2oft_{language}')
                self.a2o_ft[language] = np.load(a2oftfile)
            if (mode == "s") or ((mode in ["os", "or"]) and (aggregate != "compositions")):
                a2sftfile = parser.get('actions', f'a2sft_{language}')
                self.a2s_ft[language] = np.load(a2sftfile)
            if mode == "os" and aggregate == "compositions":
                a2ospairsftfile = parser.get('actions', f'a2ospairsft_{language}')
                self.a2ospairs_ft[language] = np.load(a2ospairsftfile)
                ospairsftfile = parser.get('actions', f'ospairsft_{language}')
                self.ospairs_ft[language] = np.load(ospairsftfile)
            if xdiscr > 0:
                o2oftfile = parser.get('actions', f'o2oft_{language}')
                self.o2o_ft[language] = np.load(o2oftfile)
                s2sftfile = parser.get('actions', f's2sft_{language}')
                self.s2s_ft[language] = np.load(s2sftfile)

    #
    # Predict the class of each test action.
    #
    def predict(self, seed, nr_test_actions, topk_objects, topk_scenes, topk_objsce, xdiscr, adiscr, aggregate, mode, lam, languages, store_preds, full):
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

        dweights = [adiscr, xdiscr]

        for i in tqdm(range(len(test_actions)), desc = "computing topks for actions"):
            if (mode == "o") or ((mode in ["os", "or"]) and (aggregate != "compositions")):
                a2os = self.a2x_scores(test_actions, i, languages, dweights, "o")
                a2xscores["a2oscores"].append(a2os)
                oidxs = np.argsort(a2os)[-topk_objects:]
                top_x["top_objects"].append(oidxs)

            if (mode == "s") or ((mode in ["os", "or"]) and (aggregate != "compositions")):
                a2ss = self.a2x_scores(test_actions, i, languages, dweights, "s")
                a2xscores["a2sscores"].append(a2ss)
                sidxs = np.argsort(a2ss)[-topk_scenes:]
                top_x["top_scenes"].append(sidxs)

            if mode == "os" and aggregate == "combined":
                a2oss = np.concatenate((a2os, a2ss))
                a2xscores["a2osscores"].append(a2oss)
                osidxs = np.argsort(a2oss)[-topk_objsce:]
                top_x["top_objsce"].append(osidxs)

            if mode == "os" and aggregate == "compositions":
                a2ospairss = self.a2x_scores(test_actions, i, languages, dweights, "osp")
                a2xscores["a2ospairscores"].append(a2ospairss)
                if lam == 1:
                    osidxs = np.argsort(a2ospairss)[-topk_objsce:]
                    top_x["top_objscepairs"].append(osidxs)
                else:
                    # # attempt at diversifying topk
                    topq_objsce = topk_objsce*5
                    osidxs = np.argsort(a2ospairss)[-topq_objsce:]
                    firstlang = languages.split("-")[0]

                    similarities = 1 - sp.spatial.distance.cdist(self.ospairs_ft[firstlang][osidxs],self.ospairs_ft[firstlang][osidxs], "cosine")

                    # # MMR approach
                    # # values for lambda from arXiv:2002.12457v1
                    ids = mmr(pairwise_matrix = similarities, w = a2ospairss[osidxs], K = topk_objsce, lam = lam)

                    top_x["top_objscepairs"].append(osidxs[ids])


        # Gather video predictions.
        predictions = np.zeros(len(self.videos), dtype=int)

        # oracle
        if mode == "or":
            results_peraction_path = "results/accuracies_per_action.csv"
            accs_df = pd.read_csv(results_peraction_path, index_col = 0)
        # oracle

        video_action_scores = []
        for i in tqdm(range(len(self.videos)), desc = "Scoring videos"):
            # Load object/scene scores.
            vidfile = self.videos[i]
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
                    # elif aggregate == "normalized":
                    #     # ""normalized"" average, take the two scores, divide them by the respective k, add them up and divide by 2
                    #     action_scores[j] = (video_action_score(objavgfeat, a2xscores["a2oscores"][j], top_x["top_objects"][j])/len(top_x["top_objects"][j]) + video_action_score(sceavgfeat, a2xscores["a2sscores"][j], top_x["top_scenes"][j])/len(top_x["top_scenes"][j]) )/2
                    # elif aggregate == "weighted":
                    #     # weighted average, take the two scores, multiply them by the respective weight z, add them up
                    #     # don't have yet a scientific way of finding the best values for zs and zo, but ballpark estimate is that zs=0.1 and zo=0.9 works best
                    #     zs = 0.1
                    #     zo = 0.9
                    #     action_scores[j] = video_action_score(objavgfeat, a2xscores["a2oscores"][j], top_x["top_objects"][j])*zo + video_action_score(sceavgfeat, a2xscores["a2sscores"][j], top_x["top_scenes"][j])*zs
                    # elif aggregate == "combined":
                    #     # unifying object and scenes in a single table
                    #     objsceavgfeat = np.concatenate((objavgfeat, sceavgfeat))
                    #     action_scores[j] = video_action_score(objsceavgfeat, a2xscores["a2osscores"][j], top_x["top_objsce"][j])
                    #     # action_scores[j] = video_action_score_normalized(objsceavgfeat, a2xscores["a2osscores"][j], top_x["top_objsce"][j])
                    elif aggregate == "compositions":
                        # with o2s passed for semantic similarity between objects and scenes in a pair
                        # action_scores[j] = video_action_score_compositions(objavgfeat, sceavgfeat, a2xscores["a2ospairscores"][j], top_x["top_objscepairs"][j], self.o2s_ft[languages.split("-")[0]])
                        # without o2s passed
                        action_scores[j] = video_action_score_compositions(objavgfeat, sceavgfeat, a2xscores["a2ospairscores"][j], top_x["top_objscepairs"][j])

                elif mode == "or":
                    # oracle mode
                    # take a peek at the truth label of the video and pick whether to use the score from the object-only or scene-only based model
                    # this is based on the accuracy each

                    try:
                        # hacky way of doing this by hardcoding the default value for kobj and ksce
                        if accs_df[f"{self.configfile}_o_{aggregate}a_{topk_objects}kobj_5ksce_{topk_objsce}kobjsce_{xdiscr}xdiscr_{adiscr}adiscr_{lam}lambda_{languages}l_accuracy"][self.aidxs[i]] >= accs_df[f"{self.configfile}_s_{aggregate}a_100kobj_{topk_scenes}ksce_{topk_objsce}kobjsce_{xdiscr}xdiscr_{adiscr}adiscr_{lam}lambda_{languages}l_accuracy"][self.aidxs[i]]:
                            action_scores[j] = video_action_score(objavgfeat, a2xscores["a2oscores"][j], top_x["top_objects"][j])
                        else:
                            action_scores[j] = video_action_score(sceavgfeat, a2xscores["a2sscores"][j], top_x["top_scenes"][j])
                    except KeyError as e:
                        print(str(e))
                        raise KeyError("To use the oracle (mode or), you need to have first run both in mode s and mode o with the same kobj and ksce configuration")
            if store_preds:
                video_action_scores.append(action_scores)
            # Select highest scoring action.
            predictions[i] = np.argmax(action_scores)

        # store per video action predictions
        if (store_preds and full):
            preds_path = f"{self.configfile}_{mode}_{aggregate}a_{topk_objects}kobj_{topk_scenes}ksce_{topk_objsce}kobjsce_{xdiscr}xdiscr_{adiscr}adiscr_{lam}lambda_{languages}l_preds"
            np.save(f"results/video_action_scores/{preds_path}", np.array(video_action_scores))

        # Map predictions to correct labels.
        predictions = test_actions[predictions]
        return self.aidxs, predictions


    def a2x_discriminate(self, actions, actionindex, languages, x_priors, a2x_ft, x2x, dweights):
        # action-based discrimination prior
        if dweights[0] > 0:
            # print(f"started actionbased discrimination for action {actionindex}\t{datetime.now().isoformat(' ', 'seconds')}")
            other_actions = np.setdiff1d(actions, [actions[actionindex]])
            entropies = x_priors - np.max(a2x_ft[languages[0]][other_actions,:], axis=0)
            # entropies = x_priors - np.max(a2x_ft[languages[0]][other_actions,:], axis=0)
            x_priors = x_priors + entropies * dweights[0]
            # print(f"finished actionbased discrimination for action {actionindex}\t{datetime.now().isoformat(' ', 'seconds')}")
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
            x_priors = self.a2x_discriminate(actions, actionindex, languages, x_priors, self.a2ospairs_ft, "", dweights)
            # not sure about having an object-scenepair x object-scenepair matrix of size (4m x 4m)
        # Deal with scenes/objects/actions with no (Working) word embedding.
        x_priors[np.isnan(x_priors)] = -10
        for i in range(1, len(languages)):
            if x == "o":
                new_priors = self.a2o_ft[languages[i]][actions[actionindex]]
            elif x == "s":
                new_priors = self.a2s_ft[languages[i]][actions[actionindex]]
            elif x == "osp":
                # raise NotImplementedError("Using more than one language with -m os -a compositions would take too long given my memory restrictions")
                new_priors = self.a2ospairs_ft[languages[i]][actions[actionindex]]
            new_priors[np.isnan(new_priors)] = -10
            x_priors += new_priors
        return x_priors


#
# Parser arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot actions from objects and scenes")
    parser.add_argument("-c", dest="configfile", help="Configuration file", default="config/ucf-101-sbert.config", type=str)
    parser.add_argument("-t", dest="nr_test_actions", help="Number of test actions", default=101, type=int)
    parser.add_argument("--kobj", dest="topk_objects", help="Top k objects per action", default=100, type=int)
    parser.add_argument("--ksce", dest="topk_scenes", help="Top k scenes per action", default=5, type=int)
    parser.add_argument("--kobjsce", dest="topk_objsce", help="Top k objects and scenes per action", default=250, type=int)
    parser.add_argument("--xdiscr", dest="xdiscr", help="Not in use: x-based (object/scene) discrimination, should be either 1 or 0", default=0, type=int)
    parser.add_argument("--adiscr", dest="adiscr", help="Not in use: action-based discrimination, should be either 1 or 0", default=0, type=int)
    parser.add_argument("-s", dest="seed", help="Random seed", default=100, type=int)
    parser.add_argument("-m", dest="mode", help="Mode used, can be Objects (o); Scenes (s); Objects and scenes (os); Objects and scenes oracle (or)", default="os", type=str)
    parser.add_argument("-a", dest="aggregate", help="Way of aggregating scores used in Objects and scenes (os) mode. Can be either 'simple' or 'compositions'", default="compositions", type=str)
    parser.add_argument("--lambda", dest="lam", help="Value for Lambda used to generate diverse top k. Must be between 0 and 1. If lambda is set to 1, no diversity is injected, otherwise, the lower lambda is, the more diverse the top k", default="0.75", type=float)
    parser.add_argument("-l", dest="language", help="Not in use: Used languages", default="English", type=str)
    parser.add_argument("--store_preds", dest="store_preds", help="Whether to save video action scores or not, can be yes (1) or no (0)", default=0, type=int)
    args = parser.parse_args()
    return args

#
# Main script.
#
if __name__ == "__main__":
    # Parse arguments.
    args = parse_args()

    if args.mode not in ["s","o", "os", "or"]:
        raise ValueError("Mode used, can be Objects (o); Scenes (s); Objects and scenes (os), Objects and scenes oracle (or)")

    if args.mode == "os":
        if args.aggregate not in ['simple', 'compositions']:
            raise ValueError("When using os mode an aggregation method needs to be specified.\n Can be 'simple', 'compositions'")
        if (args.aggregate == "compositions") & (args.xdiscr == 1) :
            raise NotImplementedError("cannot use x-based discrimination with -m os -a compositions")

    if (args.lam < 0):
        raise ValueError("Lambda must be between 0 and 1")

    # Initialize zero-shot classifier.
    model = ZeroShotActionClassifier(args.configfile, args.mode, args.aggregate, args.xdiscr)

    full = 1 if (args.nr_test_actions==101 or args.nr_test_actions==400 or args.nr_test_actions==10 or args.nr_test_actions==21) else 0

    os.makedirs("results/confusion_matrix", exist_ok = True)
    os.makedirs("results/classification_report", exist_ok = True)
    os.makedirs("results/video_action_scores", exist_ok = True)
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
                            lam = args.lam,
                            languages = args.language,
                            store_preds = args.store_preds,
                            full = full)

    # Print the results.
    acc = np.mean(ty == tp)

    results = { "datetime": datetime.now().isoformat(' ', 'seconds'),
                "configfile":args.configfile.split("/")[1],
                "mode":args.mode,
                "a": args.aggregate,
                "t":args.nr_test_actions,
                "kobj":args.topk_objects,
                "ksce":args.topk_scenes,
                "kobjsce": args.topk_objsce,
                "xdiscr": args.xdiscr,
                "adiscr": args.adiscr,
                "lam": args.lam,
                "s":args.seed,
                "l":args.language,
                "acc":acc
                }

    print(f"Setting: [configfile:{args.configfile.split('/')[1]}, mode:{args.mode}, a: {args.aggregate} t:{args.nr_test_actions}, kobj:{args.topk_objects}, ksce:{args.topk_scenes}, kobjsce: {args.topk_objsce}, xdiscr: {args.xdiscr}, adiscr: {args.adiscr}, lambda:{args.lam}, s:{args.seed}, l:{args.language}]: \tacc: {acc:.4f}")


    results_path = "results/accuracies.csv"
    df = pd.read_csv(results_path, index_col=0) if os.path.exists(results_path) else pd.DataFrame(columns = results.keys())
    df = df.append(results, ignore_index=True)
    df.to_csv(results_path)

    # Print confusion matrix
    if full:
        root_name = f"{args.configfile.split('/')[1]}_{args.mode}_{args.aggregate}a_{args.topk_objects}kobj_{args.topk_scenes}ksce_{args.topk_objsce}kobjsce_{args.xdiscr}xdiscr_{args.adiscr}adiscr_{args.lam}lambda_{args.language}l"

        # plotting out confusion matrices and storing classification reports
        plt.rcParams["figure.figsize"] = (70,70)
        cm = confusion_matrix(ty, tp, normalize = "true")
        pd.DataFrame(cm, index = model.actions, columns = model.actions).to_csv(f'results/confusion_matrix/{root_name}_confmatrix.csv')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= model.actions)
        disp.plot(xticks_rotation='vertical')
        plt.savefig(f'results/confusion_matrix/{root_name}_confmatrix.png')
        clsf_report = pd.DataFrame(classification_report(y_true = ty, y_pred = tp, output_dict=True, target_names =  model.actions, zero_division = 0)).transpose()
        clsf_report.to_csv(f'results/classification_report/{root_name}_classreport.csv', index= True)

        # Store accuracy per action and top objects/scenes.
        colname = f"{root_name}_accuracy"
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
