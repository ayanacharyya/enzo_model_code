import time

start_time3 = time.time()
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
import subprocess
import os
from astropy.convolution import convolve, Gaussian2DKernel

fn = sys.argv[1]
# ----------------------------------------
vmin = -75  # km/s
vmax = 75  # km/s
vsigma = 15  # km/s
nv = vmax - vmin
ng = 1500
# ----------------------------------------
vbin = np.linspace(vmin, vmax, nv)
L = [[0 for j in xrange(ng)] for i in xrange(ng)]


# ----------------------------------------
def get_lum(z):
    for i in range(ng):
        for j in range(ng):
            L[i][j] = l[i * ng * nv + j * nv + z]
    return L


# ----------------------------------------
def plot_slice(z):
    # plt.cla()
    fig, ax = plt.subplots(1, 1)
    lum = get_lum(z)
    ###-----------for convolving---------------
    lum = np.power(lum, 10)
    lum = convolve(lum, Gaussian2DKernel(2))
    lum = np.log10(lum)
    ###----------------------------------------
    im = ax.imshow(lum, clim=(11, 16), cmap='Blues')
    # if (z==0):
    cb = fig.colorbar(im)
    cb.set_label('Log H alpha surface brightness (erg/s/pc^2)')
    plt.xlabel('Particle position x (kpc)')
    plt.ylabel('Particle position y (kpc)')
    ax.set_xticklabels([0.02 * (i - 750) for i in list(ax.get_xticks())])
    ax.set_yticklabels([0.02 * (i - 750) for i in list(ax.get_yticks())])
    plt.title('H alpha map: Velocity slice ' + str(z + vmin) + ' km/s')
    # plt.show()
    plt.savefig(path + '/Ha_map_slice' + str(z) + '.png')
    plt.close()
    # print 'Displayed slice '+str(z)+' in '+str((time.time() - start_time)/60)+' minutes'


# ----------------------------------------
def plot_intensity(x, y, d):
    I = np.zeros(nv)
    for v in range(nv):
        for i in range(-d, d + 1):
            for j in range(-d, d + 1):
                I[v] = I[v] + 10 ** l[(x + i) * ng * nv + (y + j) * nv + v]
    plt.plot(vbin, np.log10(I))
    plt.xlabel('Vel channel (km/s)')
    plt.ylabel('Intensity')
    plt.title('Velocity resolved intensity of ' + fn + ' at pixel ' + str(x) + ',' + str(y))
    # plt.show()
    if (d == 0):
        plt.savefig(path + '/Intensity_at_' + str(x) + ',' + str(y) + '.png')
    else:
        plt.savefig(path + '/Smoothed_Intensity_at_' + str(x) + ',' + str(y) + ',' + str(d) + '.png')
    plt.close()


# ----------------------------------------
'''
def onclick(event):
    global z
    z = z+1
    plot_slice(z)
'''
# ----------------------------------------
fin = h5py.File('/Users/acharyya/models/vresHalpha/trial_vresHalpha_' + fn + '.h5', 'r')
l = fin['lum'][:]
m = np.max(l)
path = '/Users/acharyya/models/vresHalpha/' + fn + '_convolved'  ###
if not os.path.exists(path):
    subprocess.call(['mkdir', path])
# ----------------------------------------
for k in range(nv):
    plot_slice(k)
    # plot_intensity(740+k,740+k,2)
# ----------------------------------------
# cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
subprocess.call('convert -delay 5 $(for i in $(seq 0 1 149); do echo ' + path + '/Ha_map_slice\
${i}.png; done) -loop 1 ' + path + '/' + fn + '_halpha.gif ', shell=True)
print 'Done in ' + str((time.time() - start_time3) / 60) + ' minutes'
# subprocess.call('echo Done plot_vreshalpha in '+str((time.time() - start_time)/60)+' minutes | \
# mail -s "completion alert" acharyyaayan@gmail.com',shell=True)
