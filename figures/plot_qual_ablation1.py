import numpy as np
import matplotlib.pyplot as plt
import matplotlib

action_names = np.array(['apply eye makeup','apply lipstick','archery','baby crawling','balance beam','band marching','baseball pitch','basketball','basketball dunk','bench press','biking','billiards','blow dry hair','blowing candles','body weight squats','bowling','boxing punching bag','boxing speed bag','breast stroke','brushing teeth','clean and jerk','cliff diving','cricket bowling','cricket shot','cutting in kitchen','diving','drumming','fencing','field hockey penal','floor gymnastics','frisbee catch','front crawl','golf swing','haircut','hammer throw','hammering','handstand pushups','handstand walking','head massage','high jump','horse race','horse riding','hula hoop','ice dancing','javelin throw','juggling balls','jump rope','jumping jack','kayaking','knitting','long jump','lunges','military parade','mixing','mopping floor','nunchucks','parallel bars','pizza tossing','playing cello','playing daf','playing dhol','playing flute','playing guitar','playing piano','playing sitar','playing tabla','playing violin','pole vault','pommel horse','pull ups','punch','push ups','rafting','rock climbing indoor','rope climbing','rowing','salsa spin','shaving beard','shotput','skate boarding','skiing','ski jet','sky diving','soccer juggling','soccer penalty','still rings','sumo wrestling','surfing','swing','table tennis shot','tai chi','tennis swing','throw discus','trampoline jumping','typing','uneven bars','volleyball spiking','walking with dog','wall pushups','writing on board','yo yo'])

improvements = np.array([-0.013793103448276,0.0701754385964912,0.0206896551724141,-0.136363636363636,0.083333333333333,-0.412903225806451,0.026666666666666,0.104477611940298,0.251908396946564,-0.20625,0,0.68,-0.0229007633587786,0.220183486238532,-0.098214285714285,0.012903225806452,0.024539877300613,-0.0223880597014925,0,0.099236641221374,0,-0.021739130434783,0.086330935251799,0,0.109090909090909,0.04,0.0496894409937889,-0.0900900900900899,0.206349206349206,-0.024,0,0,0.194244604316547,0.361538461538461,0,-0.0357142857142857,-0.0625,0.027027027027027,-0.0612244897959183,0.268292682926829,0.096774193548387,0.079268292682927,-0.112,0.069620253164557,0.0427350427350427,0.0082644628099174,0,0,0.226950354609929,-0.276422764227642,0.351145038167938,0.0157480314960629,0.016,0,-0.018181818181818,0.0378787878787879,0.105263157894737,0,0.189024390243903,0.0397350993377483,0.0060975609756097,0.103225806451613,0.35625,-0.047619047619047,0.038216560509555,0.279279279279279,-0.36,-0.120805369127517,0.0813008130081301,0,0,0,0.837837837837837,0.222222222222223,-0.0924369747899152,0.043795620437956,0,-0.118012422360248,0.0833333333333327,-0.608333333333334,-0.148148148148148,0,0.59090909090909,-0.0068027210884354,0.627737226277373,0,-0.017241379310345,0.214285714285715,0.366412213740459,0.15,0.02,-0.030120481927711,0,-0.042016806722689,0.110294117647059,0.0192307692307692,0.39655172413793,0.065040650406504,-0.023076923076923,-0.197368421052631,0]) *100



indeces = np.argsort(improvements)
k = 101
# print(indeces[-k:],indeces[:k])
pos = improvements[indeces] >=0
# indeces = indeces[-k:]

colors = ["#606060", "#A0A0A0", "#D0D0D0", "#FF9933"]

x = np.arange(len(improvements[indeces]))
w = 0.9

plt.figure(figsize=(20,8))
plt.margins(0.01)
#
p1 = plt.bar(x[~pos], improvements[indeces[~pos]], color=colors[1], width=w)
p2 = plt.bar(x[pos], improvements[indeces[pos]], color=colors[3], width=w)
#
plt.ylim([-100,100])
# plt.xlim([-100,100])
matplotlib.rcParams.update({'font.size': 24, 'legend.fontsize': 20})
#
plt.xticks(x,action_names[indeces],rotation=90, fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel("Action",fontsize = 13)
plt.ylabel("Accuracy(%) improvement per action using compositions",fontsize = 13)
#
# legendnames = ["without diversity", "with diversity"]
# plt.legend([p1,p2], legendnames, loc=2)
#
plt.grid(True)
plt.tight_layout()



# plt.show()
plt.savefig('improvementperaction.pdf', bbox_inches='tight')
