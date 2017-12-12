#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import numpy as np

w = []
e = []
i = 0
with open("../../Downloads/starburstenzo_BC5511/starburstenzo/starburstenzo.spectrum") as inp:
    lines = inp.readlines()[5:]
    for line in lines:
        # print line.split()[0]
        if (i < 1221):
            # print 'test'
            w.append(math.log10(float(line.split()[1])))
            e.append(line.split()[2])
        else:
            break
        i = i + 1
inp.close()
# print e
fig = plt.scatter(w, e, s=2, edgecolor='None')
plt.xlabel('Log Wavelength (A)')
plt.ylabel('Log Energy')
# plt.set_yscale('log')
# cb=plt.colorbar()
# plt.show()
plt.savefig('550Myr_salpeter_14_322.png')
