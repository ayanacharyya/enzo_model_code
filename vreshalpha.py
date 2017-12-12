#!/usr/bin/env python

import time

start_time2 = time.time()
import warnings

warnings.filterwarnings("ignore")
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import sys
import h5py
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel

# -----------------------------------------------------------------
vmin = -75  # km/s
vmax = 75  # km/s
vsigma = 15  # km/s
nv = vmax - vmin
vbin = np.linspace(vmin, vmax, nv)
ng = 1500
f_esc = 0.0
f_dust = 0.0
v_sigma = 15  # km/sec
res = 0.02  # kpc
# -----------------------------------------------------------------
fn = sys.argv[1]
loga = []
logl = []
vz = []
L_Ha = []
x = []
y = []
l = [0 for i in xrange(ng * ng * nv)]
# l = [[[0 for k in xrange(nv)] for j in xrange(ng)] for i in xrange(ng)] ###
# -----------------------------------------------------------------
lum = open('/Users/acharyya/Downloads/starburstenzo_BD4284/starburstenzo/starburstenzo.quanta', 'r')
tail = lum.readlines()[7:]
for lines in tail:
    loga.append(math.log10(float(lines.split()[0]) / 10 ** 6))
    logl.append(float(lines.split()[1]))
f = interp1d(loga, logl, kind='cubic')
lum.close()
# -----------------------------------------------------------------
list = open('/Users/acharyya/models/paramlist/param_list_' + fn, 'r')
for line in list:
    # print line
    if float(line.split()[5]) >= 0.1:
        x.append(int(np.floor((float(line.split()[1]) * 1300 - 650) / res)) + ng / 2)
        y.append(int(np.floor((float(line.split()[2]) * 1300 - 650) / res)) + ng / 2)
        Q_H0 = f(math.log10(float(line.split()[5]))) - 6 + math.log10(
            float(line.split()[4]))  # '6' bcz starburst was run for mass of 1M Msun
        L_Ha.append((1.37e-12 * (1 - f_esc) * (1 - f_dust) * 10 ** Q_H0) / ((res * 1000) ** 2))
        vz.append(float(line.split()[6]))
list.close()
# -----------------------------------------------------------------
L_Ha = L_Ha / np.sqrt(2 * np.pi * vsigma ** 2)
L = L_Ha * np.exp(-(np.subtract.outer(vbin, vz) ** 2) / (2 * vsigma ** 2))
# -----------------------------------------------------------------
for (i, j), v in np.ndenumerate(L):
    l[x[j] * ng * nv + y[j] * nv + i] = l[x[j] * ng * nv + y[j] * nv + i] + v
    # l[x[j]][y[j]][i] = l[x[j]][y[j]][i] + v ###
###-------------------for convolving with gaussian beam----------------------------------------------
# l = convolve(l, Gaussian2DKernel(5))
# fn = fn + '_convolved'
###-----------------------------------------------------------------
l = np.log10(l)
l[np.isneginf(l)] = 0  ##-300
##----------------------for writing np array-------------------------------------------
# np.save('/Users/acharyya/models/vresHalpha/vresHalpha_'+fn,l)
##-----------------------for writing fits file------------------------------------------
'''
hdu = fits.PrimaryHDU(l)
hdu.header['AXIS1'] = 'Velocity from -75 to +75 (in km/s)'
hdu.header['AXIS2'] = 'y coordinate (in units of 20pc)'
hdu.header['AXIS3'] = 'x coordinate (in units of 20pc)'
hdu.writeto('/Users/acharyya/models/vresHalpha/vresHalpha_'+fn+'.fits')
'''
##----------------------for writing hdf5 file-------------------------------------------

fout = h5py.File('/Users/acharyya/models/vresHalpha/trial_vresHalpha_' + fn + '.h5', 'w')
fout.create_dataset('lum', data=l)
fout.close()

##-----------------------------------------------------------------
print 'Velocity resolved H alpha array of ' + fn + ' saved.'
print(fn + ' in %s minutes' % ((time.time() - start_time2) / 60))
