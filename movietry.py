#!/usr/bin/env python

from galanyl import GalaxyAnalyzer as gal
from matplotlib import pyplot as plt
import matplotlib.animation as anm
import numpy as np
from pylab import *
dpi=100
#def ani_make():
fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_aspect('equal')
g = []
g[0] = gal.from_hdf5_file('DD0000_toomre')
g[1] = gal.from_hdf5_file('DD0050_toomre')
g[2] = gal.from_hdf5_file('DD0060_toomre')
im=ax.imshow(g[0].gas.total_toomre_q, clim=(0,15), cmap='seismic)
ax.colorbar(im)
#plt.show(im)
tight_layout()
    
def update_img(n):
    im.set_data(g[n].gas.total_toomre_q)
    return im
        
ani=anm.funcAnimation(fig,update_img,3,interval=1)
writer=anm.writers['ffmpeg'](fps=1)
    
ani.save('try.mp4',writer=writer,dpi=dpi)
    #return ani

    