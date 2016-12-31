file = open('bleak_best', 'rb')
liste = []

for line in file:
    sp = line.split('\t')
    if (len(sp) == 2):
        liste.append[sp[1]]

import pickle
p = open('bleak_b', 'rb')
pickle.dump(liste, p)

