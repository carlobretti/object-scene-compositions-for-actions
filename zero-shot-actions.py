#
# Zero-shot action classification using standard action to object semantics.
#

import os
import sys
import numpy as np
import argparse
import configparser
from   six.moves import cPickle as pickle
from   scipy.spatial.distance import cosine
from   scipy.stats import beta
from   nltk.corpus import wordnet as wn

#
# Obtain action score for scenes and objects
#
def video_action_score(x_scores, action_x_scores, top_x):
    return np.dot(x_scores[top_x], action_x_scores[top_x])

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
        for language in languages:
            a2oftfile = parser.get('actions', f'a2oft_{language}')
            self.a2o_ft[language] = np.load(a2oftfile)
            a2sftfile = parser.get('actions', f'a2sft_{language}')
            self.a2s_ft[language] = np.load(a2sftfile)

    #
    # Predict the class of each test action.
    #
    def predict(self, seed, nr_test_actions, topk_objects, topk_scenes, mode, languages):
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
        top_scenes = []
        a2sscores = []
        for i in range(len(test_actions)):
            a2os = self.a2x_scores(test_actions, i, languages, "o")
            a2oscores.append(a2os)
            oidxs = np.argsort(a2os)[-topk_objects:]
            top_objects.append(oidxs)

            a2ss = self.a2x_scores(test_actions, i, languages, "s")
            a2sscores.append(a2ss)
            oidxs = np.argsort(a2ss)[-topk_scenes:]
            top_scenes.append(oidxs)
            # Print top scoring objects for each action.
            #wnids = [c.strip() for c in open("../../data/imagenet/wnids-12988.txt")]
            #oidxs = np.argsort(a2op)[::-1]
            #print(self.actions[test_actions[i]], end=": ")
            #for j in range(5):
            #    print(wn._synset_from_pos_and_offset('n',int(wnids[oidxs[j]][1:])).lemma_names()[0], end=", ")
            #print()



        # Gather video predictions.
        predictions = np.zeros(len(self.videos), dtype=int)
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
                    action_scores[j] = video_action_score(objavgfeat, a2oscores[j], top_objects[j])
                elif mode == "s":
                    action_scores[j] = video_action_score(sceavgfeat, a2sscores[j], top_scenes[j])
                elif mode == "os":
                    raise NotImplementedError
                    action_scores[j] = (video_action_score(objavgfeat, a2oscores[j], top_objects[j]) + video_action_score(sceavgfeat, a2sscores[j], top_scenes[j]))/2

            # Select highest scoring action.
            predictions[i] = np.argmax(action_scores)
        print()

        # Map predictions to correct labels.
        predictions = test_actions[predictions]
        print(predictions.shape)
        return self.aidxs, predictions

    #
    #
    #
    def a2x_scores(self, actions, actionindex, languages, x):
        languages = languages.split("-")#list(self.a2o_ft.keys())
        x_priors = self.a2o_ft[languages[0]][actions[actionindex]] if x == "o" else self.a2s_ft[languages[0]][actions[actionindex]]
        # Deal with scenes/objects/actions with no (Working) word embedding.
        x_priors[np.isnan(x_priors)] = -10
        for i in range(1, len(languages)):
            new_priors = self.a2o_ft[languages[0]][actions[actionindex]] if x == "o" else self.a2s_ft[languages[0]][actions[actionindex]]
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
    parser.add_argument("-s", dest="seed", help="Random seed", default=100, type=int)
    parser.add_argument("-m", dest="mode", help="Mode used, can be Objects (o); Scenes (s); Objects and scenes (os)", default="o", type=str)
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
    ty, tp = model.predict( seed = args.seed,
                            nr_test_actions = args.nr_test_actions,
                            topk_objects = args.topk_objects,
                            topk_scenes = args.topk_scenes,
                            mode = args.mode,
                            languages = args.language)

    # Print the results.
    acc = np.mean(ty == tp)
    print(f"Setting: [mode:{args.mode} t:{args.nr_test_actions}, kobj:{args.topk_objects}, ksce:{args.topk_scenes}, s:{args.seed}, l:{args.language}]: acc: {acc:.4f}")
    # Print accuracy per action.
    #print model.actions
    #accs = np.bincount(ty[ty==tp], minlength=len(model.actions)) / np.bincount(ty, minlength=len(model.actions))
    #print(accs)
