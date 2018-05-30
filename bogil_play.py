import numpy as np
import sys

import est.xform
import embed.schmidt
import embed.c1c2

from uhfb import UHFB
import HfbUtil


import pickle

#file_name = 'bogil_randHV_n4uP2a2b2.pk'
file_name = 'bogil_n4u4a4b3.pk'
with open(file_name,'rb') as file_object:
    bogil = pickle.load(file_object)

n = bogil.G.shape[0]

##Can indeed verify that this is a HF state with no pairing contributions
#print(bogil.G[:n//2,:n//2])
#print(bogil.G[n//2:,:n//2])
#print(bogil.H)
bogil._do_hfb()
#print(bogil.E)
#print(bogil.E_HF)
#print(bogil.E_pp)

n = bogil.n
print(n)
print(bogil.G[:2*n,:2*n])
print()
print(bogil.G[:2*n,2*n:])




