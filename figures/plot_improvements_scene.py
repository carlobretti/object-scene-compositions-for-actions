import numpy as np
import matplotlib.pyplot as plt
import matplotlib

action_names = np.array(['apply eye makeup','apply lipstick','archery','baby crawling','balance beam','band marching','baseball pitch','basketball','basketball dunk','bench press','biking','billiards','blow dry hair','blowing candles','body weight squats','bowling','boxing punching bag','boxing speed bag','breast stroke','brushing teeth','clean and jerk','cliff diving','cricket bowling','cricket shot','cutting in kitchen','diving','drumming','fencing','field hockey penal','floor gymnastics','frisbee catch','front crawl','golf swing','haircut','hammer throw','hammering','handstand pushups','handstand walking','head massage','high jump','horse race','horse riding','hula hoop','ice dancing','javelin throw','juggling balls','jump rope','jumping jack','kayaking','knitting','long jump','lunges','military parade','mixing','mopping floor','nunchucks','parallel bars','pizza tossing','playing cello','playing daf','playing dhol','playing flute','playing guitar','playing piano','playing sitar','playing tabla','playing violin','pole vault','pommel horse','pull ups','punch','push ups','rafting','rock climbing indoor','rope climbing','rowing','salsa spin','shaving beard','shotput','skate boarding','skiing','ski jet','sky diving','soccer juggling','soccer penalty','still rings','sumo wrestling','surfing','swing','table tennis shot','tai chi','tennis swing','throw discus','trampoline jumping','typing','uneven bars','volleyball spiking','walking with dog','wall pushups','writing on board','yo yo'])

improvements = np.array([0.682758620689655,0.0789473684210526,0.462068965137931,-0.431818181545455,0.444444444444444,0,-0.026666666333334,-0.26865671677612,0.015267175587786,0,0.798507462522388,0,0,0.743119266055045,0.294642857571428,0.00645161299999997,0.276073620018404,0.0298507458358208,0,0.15267175572519,0,0.007246376710144,0.539568345323741,0.0059880239520958,-0.027272727454546,0.406666666666666,0.770186335403726,0.549549549549549,-0.269841270031747,0.168,0,0,0.057553957165467,-0.1,0,0.1,0.109375,-0.009009008954955,-0.006802721,0.626016260162601,-0.096774193322581,0.664634145975609,0.296,0,0.222222222222222,0.0330578512396694,-0.013888889,0,0.276595744680851,0.219512195121951,0.435114503816793,0.0393700787401574,0.384,0.323529412235294,0.072727272636363,0.0833333333333333,0.93859649122807,-0.123893805,0.725609755682926,0,0,0.864516129032258,0.75,0.200000000047619,0.904458598726114,0.432432432432432,0.46,0.0604026847516778,0.756097560975609,0,0,0,-0.018018018063064,0.305555555777777,0.00840336134453781,0.700729926854014,0,0.434782608819875,0.0208333331111111,0.141666666666666,0.125925925629629,0,-0.10000000009091,8.84353701380292E-11,0.036496350708029,0,0.827586206793103,0.047619047380952,0.816793893129771,0.757142857142857,-0.11,0.728915662578313,0,0.285714285529411,0.720588235705882,0,0.172413793103448,0.349593495837398,0,0.144736842473684,0]) *100



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
plt.savefig('improvementperactionSCE.pdf', bbox_inches='tight')
