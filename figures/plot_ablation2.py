import matplotlib.pyplot as plt
import matplotlib
import numpy as np

topk = [100,250,500]

nodiversity = np.array([36.2,38.5,37.6])
diversity = np.array([36.7,38.8,38.0])

colors = ["#606060", "#A0A0A0", "#D0D0D0", "#FF9933"]

x = np.arange(3)
w = 0.4

p1 = plt.barh(x+w, nodiversity, color=colors[1], height=w)
p2 = plt.barh(x, diversity, color=colors[3], height=w)

# plt.ylim([-0.2,3])
plt.xlim([20,40])
matplotlib.rcParams.update({'font.size': 18, 'legend.fontsize': 14})

plt.yticks(x + 0.2, topk)
plt.ylabel("Top compositions per action", fontsize=18)
plt.xlabel("Accuracy", fontsize=18)

legendnames = ["without diversity", "with diversity"]
plt.legend([p1,p2], legendnames, loc=2)

plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig('ablation2.pdf')
