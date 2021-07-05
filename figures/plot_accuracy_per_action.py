import numpy as np
import matplotlib.pyplot as plt
import matplotlib

action_names = np.array(['apply eye makeup','apply lipstick','archery','baby crawling','balance beam','band marching','baseball pitch','basketball','basketball dunk','bench press','biking','billiards','blow dry hair','blowing candles','body weight squats','bowling','boxing punching bag','boxing speed bag','breast stroke','brushing teeth','clean and jerk','cliff diving','cricket bowling','cricket shot','cutting in kitchen','diving','drumming','fencing','field hockey penal','floor gymnastics','frisbee catch','front crawl','golf swing','haircut','hammer throw','hammering','handstand pushups','handstand walking','head massage','high jump','horse race','horse riding','hula hoop','ice dancing','javelin throw','juggling balls','jump rope','jumping jack','kayaking','knitting','long jump','lunges','military parade','mixing','mopping floor','nunchucks','parallel bars','pizza tossing','playing cello','playing daf','playing dhol','playing flute','playing guitar','playing piano','playing sitar','playing tabla','playing violin','pole vault','pommel horse','pull ups','punch','push ups','rafting','rock climbing indoor','rope climbing','rowing','salsa spin','shaving beard','shotput','skate boarding','skiing','ski jet','sky diving','soccer juggling','soccer penalty','still rings','sumo wrestling','surfing','swing','table tennis shot','tai chi','tennis swing','throw discus','trampoline jumping','typing','uneven bars','volleyball spiking','walking with dog','wall pushups','writing on board','yo yo'])

improvements = np.array([0.682758621,0.078947368,0.496551724,0.045454545,0.444444444,0,0.906666667,0.358208955,0.229007634,0,0.835820896,1,0,0.743119266,0.303571429,1,0.736196319,0.037313433,0,0.152671756,0,0.31884058,0.539568345,0.005988024,0.154545455,0.406666667,0.770186335,0.54954955,0.253968254,0.176,0,0,0.942446043,0.8,0.1,0,0.109375,0.045045045,0,0.62601626,0.85483871,0.756097561,0.296,1,0.222222222,0.033057851,0,0,0.276595745,0.219512195,0.435114504,0.039370079,0.544,0.426470588,0.163636364,0.083333333,0.938596491,0,0.829268293,0,0,0.864516129,0.75,0.819047619,0.904458599,0.432432432,0.46,0.080536913,0.756097561,0,0,0,0.936936937,0.340277778,0.008403361,0.98540146,0,0.440993789,0.048611111,0.166666667,0.82962963,0,0.590909091,0.006802721,0.97080292,0,0.905172414,0.952380952,0.816793893,0.757142857,0.02,0.759036145,0,0.647058824,0.904411765,0,0.172413793,0.37398374,0,0.151315789,0]) *100



indeces = np.argsort(improvements)
k = 101
# print(indeces[-k:],indeces[:k])
# pos = improvements[indeces] >=0
# indeces = indeces[-k:]

colors = ["#606060", "#A0A0A0", "#D0D0D0", "#FF9933"]

x = np.arange(len(improvements[indeces]))
w = 0.9

plt.figure(figsize=(20,8))
plt.margins(0.01)
#
# p1 = plt.bar(x[~pos], improvements[indeces[~pos]], color=colors[1], width=w)
p1 = plt.bar(x, improvements[indeces], color=colors[3], width=w)
#
plt.ylim([0,100])
# plt.xlim([-100,100])
matplotlib.rcParams.update({'font.size': 24, 'legend.fontsize': 20})
#
plt.xticks(x,action_names[indeces],rotation=90, fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel("Action",fontsize = 13)
plt.ylabel("Accuracy(%) per action using object-scene compositions",fontsize = 13)
#
# legendnames = ["without diversity", "with diversity"]
# plt.legend([p1,p2], legendnames, loc=2)
#
plt.grid(True)
plt.tight_layout()



# plt.show()
plt.savefig('accuracyperaction.pdf', bbox_inches='tight')
