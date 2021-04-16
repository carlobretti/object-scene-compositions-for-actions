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
from   sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

        # Find top objects and scenes per action.
        top_objects = []
        a2oscores = []
        top_scenes = []
        a2sscores = []
        all_records = []

        for i in range(len(test_actions)):
            a2os = self.a2x_scores(test_actions, i, languages, "o")
            a2oscores.append(a2os)
            oidxs = np.argsort(a2os)[-topk_objects:]
            top_objects.append(oidxs)

            a2ss = self.a2x_scores(test_actions, i, languages, "s")
            a2sscores.append(a2ss)
            sidxs = np.argsort(a2ss)[-topk_scenes:]
            top_scenes.append(sidxs)

            # Save top scoring objects and scenes for each action.
            wnids = [c.strip() for c in open("data/imagenet/wnids-12988.txt")]
            scenes = [c.strip() for c in open("data/places-365/words/places365-words-English.txt")]
            allsidxs = np.argsort(a2ss)[::-1]
            alloidxs = np.argsort(a2os)[::-1]
            record = {}
            record["action_name"] = self.actions[test_actions[i]]
            for j in range(5):
                record[f"top_scene_{j}"] = scenes[allsidxs[j]]
                record[f"top_object_{j}"] = wn.synset_from_pos_and_offset('n',int(wnids[alloidxs[j]][1:])).lemma_names()[0]

            all_records.append(record)

        pd.DataFrame(all_records).sort_index(axis=1).to_csv(f"results/{mode}_object_scene_action.csv")

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
                    # # simplest average, take the two scores, add them up and divide by 2
                    # action_scores[j] = (video_action_score(objavgfeat, a2oscores[j], top_objects[j]) + video_action_score(sceavgfeat, a2sscores[j], top_scenes[j]))/2

                    # ""normalized"" average, take the two scores, divide them by the respective k, add them up and divide by 2
                    # action_scores[j] = (video_action_score(objavgfeat, a2oscores[j], top_objects[j])/len(top_objects[j]) + video_action_score(sceavgfeat, a2sscores[j], top_scenes[j])/len(top_scenes[j]) )/2

                    # ""normalized"" average, take the two scores, divide them by the respective k, add them up and divide by 2
                    action_scores[j] = (video_action_score(objavgfeat, a2oscores[j], top_objects[j])*len(top_objects[j]) + video_action_score(sceavgfeat, a2sscores[j], top_scenes[j])*len(top_scenes[j]) )/2


                    # # smart, semantically-aware average
                    # raise NotImplementedError


            # Select highest scoring action.
            predictions[i] = np.argmax(action_scores)
        print()

        # Map predictions to correct labels.
        predictions = test_actions[predictions]
        return self.aidxs, predictions

    #
    #
    #
    def a2x_scores(self, actions, actionindex, languages, x):
        languages = languages.split("-")
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

    if args.mode not in ["s","o", "os"]:
        raise ValueError("Mode used, can be Objects (o); Scenes (s); Objects and scenes (os)")

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

    results = { "datetime": datetime.now().isoformat(' ', 'seconds'),
                "mode":args.mode,
                "t":args.nr_test_actions,
                "kobj":args.topk_objects,
                "ksce":args.topk_scenes,
                "s":args.seed,
                "l":args.language,
                "acc":acc}
    print(f"Setting: [mode:{args.mode} t:{args.nr_test_actions}, kobj:{args.topk_objects}, ksce:{args.topk_scenes}, s:{args.seed}, l:{args.language}]: acc: {acc:.4f}")

    results_path = "results/accuracies.csv"
    df = pd.read_csv(results_path, index_col=0) if os.path.exists(results_path) else pd.DataFrame(columns = results.keys())
    df = df.append(results, ignore_index=True)
    df.to_csv(results_path)

    # Print confusion matrix
    if args.nr_test_actions==101:
        plt.rcParams["figure.figsize"] = (70,70)
        cm = confusion_matrix(ty, tp, normalize = "true")
        pd.DataFrame(cm, index = model.actions, columns = model.actions).to_csv(f'{args.mode}_{args.topk_objects}obj_{args.topk_scenes}_sce_{args.language}_lang.csv')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels= model.actions)
        disp.plot(xticks_rotation='vertical')
        plt.savefig(f'results/{args.mode}_{args.topk_objects}obj_{args.topk_scenes}_sce_{args.language}_lang.png')


        # Store accuracy per action and top objects/scenes.
        accs = np.bincount(ty[ty==tp], minlength=len(model.actions)) / np.bincount(ty, minlength=len(model.actions))
        accs = np.stack((model.actions,accs), axis = -1)
        accs = [{"action_name": result[0], "accuracy": result[1]} for result in accs]
        accs_df = pd.DataFrame(accs)
        top_sc_df = pd.read_csv(f"results/{args.mode}_object_scene_action.csv", index_col = 0)
        pd.merge(accs_df, top_sc_df).sort_values("accuracy").to_csv(f"results/{args.mode}_object_scene_action.csv")
