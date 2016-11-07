#!/usr/bin/env python

from galanyl import GalaxyAnalyzer
from matplotlib import pyplot as plt
for i in range(10,21):
    string='DD0'+str(i)+'0_toomre'
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_aspect('equal')
    #print string
    g = GalaxyAnalyzer.from_hdf5_file(string)
    #im=ax.imshow(g.gas.total_toomre_q, clim=(-8,15), cmap='seismic')
    im=ax.imshow(g.gas.mass_flux_density,clim=(-3e-16, 3e-16), cmap='seismic')
    cb = plt.colorbar(im)
    plt.xlabel('x (grid units)')
    plt.ylabel('y (grid units)')
    #cb.set_label('Toomre parameter')
    cb.set_label('mass flux density')
    plt.title(str(i)+'0 Myr')
    #plt.show(im1)
    #fig.savefig('toomre/nofb_20pc_DD0'+str(i)+'0_toomre.png')
    fig.savefig('toomre/nofb_20pc_DD0'+str(i)+'0_mfd.png')

