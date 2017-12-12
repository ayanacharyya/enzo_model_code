import time

start_time2 = time.time()
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp

res = 0.02  # kpc
size = 1300  # kpc
ng = int(size / res)
ns = 0
var = np.zeros(ng ** 2)
with open('/Users/acharyya/models/paramlist/param_list_DD0300_hgf') as inp:
    inp.readline()
    for lines in inp:
        x = int(float(lines.split()[1]) * ng)
        y = int(float(lines.split()[2]) * ng)
        var[x * ng + y] += 1
        ns += 1
inp.close()
# ----------for hist----------------
print 'For ' + str(ns) + ' star particles...'
print np.min(var), np.max(var)
plt.hist(var, 40, log=True, edgecolor='none')
plt.xlabel('Nstar/cell')
plt.ylabel('Ncell')
plt.title('Packing of ' + str(ns) + ' star particles')
print('Done in %s minutes' % ((time.time() - start_time2) / 60))
plt.show(block=False)
