#python routine to replot SAMI noise spectra provided by Rob
#by Ayan, June 2018

import numpy as np
from matplotlib import pyplot as plt
import os
HOME = os.getenv('HOME') + '/'

from astropy.io import fits

outpath = HOME + 'Dropbox/papers/enzo_paper/Figs/'
inpath = HOME + '/models/Noise_model/'
fs = 20 # fontsize

branch_arr = ['B', 'R']
ylim_arr = [(0,0.02), (0,0.008)]
xlim_arr = [(3733,5755), (6298, 7423)]
plt.close('all')

for (i,branch) in enumerate(branch_arr):
    fname = 'NoiseData-99259_'
    noise = fits.open(inpath+fname+branch+'.fits')[1].data[0]
    fig = plt.figure(figsize=(10,6))
    fig.subplots_adjust(hspace=0.1, wspace=0.03, top=0.97, bottom=0.15, left=0.15, right=0.95)
    plt.plot(noise[0], noise[1], c='k')
    plt.ylim(ylim_arr[i])
    plt.xlim(xlim_arr[i])
    plt.xlabel(r'Wavelength ($\AA$)', fontsize=fs)
    plt.ylabel(r'x 10$^{\rm -16}$ ergs/s/cm$^{\rm 2}$/A/spaxel', fontsize=fs)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks(), fontsize=fs)
    ax.set_yticklabels(ax.get_yticks(), fontsize=fs)
    fig.savefig(outpath + fname+branch+'.eps')
    print 'Saved file', outpath + fname+branch+'.eps'

plt.show(block=False)
