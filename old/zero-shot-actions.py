#
# Zero-shot action classification using standard action to object semantics.
#

import os
import sys
import numpy as np
import pandas as pd
import argparse
import configparser
from   six.moves import cPickle as pickle
from   scipy.spatial.distance import cosine
from   scipy.stats import beta
from   nltk.corpus import wordnet as wn

#
# Pair-wise similarity between actions and object word embeddings.
#
def wtv_mapping(wtv1, wtv2):
    wtvmap = np.zeros((wtv1.shape[0], wtv2.shape[0]), dtype=np.float32)
    for i in range(wtv1.shape[0]):
        for j in range(wtv2.shape[0]):
            wtvmap[i,j] = 1 - cosine(wtv1[i], wtv2[j])
    return wtvmap

#
# Obtain action score for
#
def video_action_score(object_scores, action_object_scores, top_objects):
    return np.dot(object_scores[top_objects], action_object_scores[top_objects])

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
        self.featdir = parser.get('actions', 'objectscores')
        self.videos  = parser.get('actions', 'videofile')
        self.ext     = parser.get('actions', 'objectfilename')
        self.videos  = [l.strip().split()[0] for l in open(self.videos)]
        self.videos  = np.array(self.videos)

        # Actions.
        self.actions = np.array([l.split("/")[0] for l in self.videos])
        self.actions, self.aidxs = np.unique(self.actions, return_inverse=True)

        # Action to object fasttext.
        languages = parser.get('actions', 'fasttext_languages')
        languages = languages.split(", ")
        self.a2o_ft = {}
        for language in languages:
            a2oftfile = parser.get('actions', 'a2oft_%s' %(language))
            self.a2o_ft[language] = np.load(a2oftfile)

    #
    # Predict the class of each test action.
    #
    def predict(self, seed, nr_test_actions, topk, languages):
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


        # Find top objects per action.

        top_objects = []
        a2oscores = []

        all_records = []

        for i in range(len(test_actions)):
            a2os = self.a2o_scores(test_actions, i, languages)
            a2oscores.append(a2os)
            oidxs = np.argsort(a2os)[-topk:]

            # Print top scoring objects for each action.
            #wnids = [c.strip() for c in open("../../data/imagenet/wnids-12988.txt")]
            #oidxs = np.argsort(a2op)[::-1]
            #print(self.actions[test_actions[i]], end=": ")
            #for j in range(5):
            #    print(wn._synset_from_pos_and_offset('n',int(wnids[oidxs[j]][1:])).lemma_names()[0], end=", ")
            #print()

            # Print top scoring scene for each action.
            scenes = [c.strip() for c in open("data/places-365/words/places365-words-English.txt")]
            alloidxs = np.argsort(a2os)[::-1]
            record = {}
            record["action_name"] = self.actions[test_actions[i]]
            for j in range(5):
                record[f"top_scene_{j}"] = scenes[alloidxs[j]]
            all_records.append(record)


            top_objects.append(oidxs)

        pd.DataFrame(all_records).to_csv("scene_action.csv")
        # Gather video predictions.
        predictions = np.zeros(len(self.videos), dtype=int)
        for i in range(len(self.videos)):
            print("Video %d/%d\r" %(i+1, len(self.videos)), end="")
            sys.stdout.flush()

            # Load object scores.
            try:
                avgfeat = np.load(self.featdir + self.videos[i] + self.ext)
            except:
                vidfile = self.videos[i].split("/")[-1][:-4]
                avgfeat = np.load(self.featdir + vidfile + self.ext)

            # Score per action.
            action_scores = np.zeros(len(test_actions))
            for j in range(len(test_actions)):
                action_scores[j] = video_action_score(avgfeat, a2oscores[j], top_objects[j])

            # Select highest scoring action.
            predictions[i] = np.argmax(action_scores)
        print()

        # Map predictions to correct labels.
        predictions = test_actions[predictions]
        return self.aidxs, predictions

    #
    #
    #
    def a2o_scores(self, actions, actionindex, languages):
        languages = languages.split("-")#list(self.a2o_ft.keys())
        object_priors = self.a2o_ft[languages[0]][actions[actionindex]]
        # Deal with objects/actions with no (Working) word embedding.
        object_priors[np.isnan(object_priors)] = -10
        for i in range(1, len(languages)):
            new_priors = self.a2o_ft[languages[i]][actions[actionindex]]
            new_priors[np.isnan(new_priors)] = -10
            object_priors += new_priors
        #object_priors /= len(languages)
        return object_priors


#
# Parser arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot actions from objects")
    parser.add_argument("-c", dest="configfile", help="Configuration file", default="ucf-101.config", type=str)
    parser.add_argument("-t", dest="nr_test_actions", help="Number of test actions", default=50, type=int)
    parser.add_argument("-k", dest="topk", help="Top k objects per action", default=100, type=int)
    parser.add_argument("-s", dest="seed", help="Random seed", default=100, type=int)
    parser.add_argument("-l", dest="language", help="Used language", default="English", type=str)
    args = parser.parse_args()
    return args

#
# Main script.
#
if __name__ == "__main__":
    # Parse arguments.
    args   = parse_args()

    # Initialize zero-shot classifier.
    model = ZeroShotActionClassifier(args.configfile)

    # Perform zero-shot action prediction.
    ty, tp = model.predict(args.seed, args.nr_test_actions, args.topk, args.language)

    # Print the results.
    acc = np.mean(ty == tp)
    print("Setting: [t:%d, k:%d, s:%d, l:%s]: acc: %.4f" \
            %(args.nr_test_actions, args.topk, args.seed, args.language, acc))


    # Store accuracy per action.
    # print(model.actions)
    accs = np.bincount(ty[ty==tp], minlength=len(model.actions)) / np.bincount(ty, minlength=len(model.actions))
    # print(accs)
    accs = np.stack((model.actions,accs), axis = -1)
    accs = [{"action_name": result[0], "accuracy": result[1]} for result in accs]
    accs_df = pd.DataFrame(accs)
    top_sc_df = pd.read_csv("scene_action.csv", index_col = 0)
    pd.merge(accs_df, top_sc_df).sort_values("accuracy").to_csv("scene_action.csv")
