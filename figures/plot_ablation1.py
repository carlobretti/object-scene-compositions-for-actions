import matplotlib.pyplot as plt
import matplotlib
import numpy as np

nractions = [25,50,101]

objectsonly = np.array([[47.1,40.9,33.5],[7.1,2.9,0.0]])
scenesonly  = np.array([[29.3,22.6,16.4],[5.6,3.6,0.0]])
fusion      = np.array([[44.6,36.8,29.1],[6.0,3.0,0.0]])
composition = np.array([[52.5,45.5,38.5],[5.7,3.3,0.0]])

colors = ["#606060", "#A0A0A0", "#D0D0D0", "#FF9933"]

x = np.arange(3)
w = 0.2
plt.figure(figsize=(8,6))

p1 = plt.bar(x, objectsonly[0], color=colors[0], width=w, yerr=objectsonly[1], ecolor="black")
p2 = plt.bar(x+w, scenesonly[0], color=colors[1], width=w, yerr=scenesonly[1], ecolor="black")
p3 = plt.bar(x+2*w, fusion[0], color=colors[2], width=w, yerr=fusion[1], ecolor="black")
p4 = plt.bar(x+3*w, composition[0], color=colors[3], width=w, yerr=composition[1], ecolor="black")

# plt.xlim([-0.2,3])
plt.ylim([0,70])


plt.rcParams.update({'font.size': 18, 'legend.fontsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14, "patch.force_edgecolor": True})

plt.xticks(x + 0.3 , nractions)
plt.xlabel("Number of test actions", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)

legendnames = ["Objects-only", "Scenes-only", "Object-scene fusion", "Object-scene composition"]
plt.legend([p1,p2,p3,p4], legendnames, loc=1)

plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig('ablation1.pdf')
