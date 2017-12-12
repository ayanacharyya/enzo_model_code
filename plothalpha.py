#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import sys

fn = sys.argv[1]
loga = []
logl = []
n = 1500
f_esc = 0.0
f_dust = 0.0
v_sigma = 15  # km/sec
res = 0.02  # kpc
l = [[0 for j in xrange(n)] for i in xrange(n)]
coord = [[] for i in range(4)]
lum = open('/Users/acharyya/SB99-v8-02/output/starburst03/starburst03.quanta', 'r')
tail = lum.readlines()[7:]  #
for lines in tail:
    # print float(lines.split()[0])/10**6
    loga.append(math.log10(float(lines.split()[0]) / 10 ** 6))
    # logl.append(float(lines.split()[7]) + float(lines.split()[2]))
    logl.append(float(lines.split()[1]))
f = interp1d(loga, logl, kind='cubic')
lum.close()

list = open('/Users/acharyya/models/paramlist/param_list_' + fn, 'r')
# list.readline()
for line in list:
    # print line
    if float(line.split()[5]) >= 0.1:
        x = int(np.floor((float(line.split()[1]) * 1300 - 650) / res)) + n / 2
        y = int(np.floor((float(line.split()[2]) * 1300 - 650) / res)) + n / 2
        Q_H0 = f(math.log10(float(line.split()[5]))) - 6 + math.log10(float(line.split()[4]))
        Ha = 1.37e-12 * (1 - f_esc) * (1 - f_dust) * 10 ** Q_H0
        l[x][y] = l[x][y] + Ha / ((res * 1000) ** 2)
        # print float(line.split()[1])
list.close()
for (i, j), v in np.ndenumerate(l):
    if (v > 0):
        l[i][j] = math.log10(v)
# l = np.log10(l)
img = plt.imshow(l, cmap='Blues')
# plt.scatter(coord[0],coord[1], c = coord[3], s = 2, edgecolor = 'None')
cb = plt.colorbar(img)
plt.xlabel('Particle position x (20pc grid unit)')
plt.ylabel('Particle position y (20pc grid unit)')
cb.set_label('Log H alpha surface brightness (erg/s/pc^2)')
plt.title('H alpha map of young stars of ' + fn)
plt.show()
plt.savefig('/Users/acharyya/models/halpha/' + fn + '_Ha_map.png')
plt.close()
print 'H alpha map of ' + fn + ' saved.'  ##
