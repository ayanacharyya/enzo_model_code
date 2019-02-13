import numpy as np
from matplotlib import pyplot as plt
import os
HOME = os.getenv('HOME')
from scipy.interpolate import interp1d
from operator import itemgetter

# ------------Reading pre-defined line list-----------------------------
def readlist(wmin, wmax):
    target = []
    finp = open(HOME + '/Mappings/lab/targetlines.txt', 'r')
    l = finp.readlines()
    for lin in l:
        if len(lin.split()) > 1 and lin.split()[0][0] != '#':
            target.append([float(lin.split()[0]), lin.split()[1]])
    finp.close()
    target = sorted(target, key=itemgetter(0))
    target = np.array(target)
    llist = np.array(target)[:, 1]
    wlist = np.asarray(target[:, 0], float)

    llist = llist[np.where(np.logical_and(wlist > wmin, wlist < wmax))]  # truncate linelist as per wavelength range
    wlist = wlist[np.where(np.logical_and(wlist > wmin, wlist < wmax))]
    return wlist, llist

# -------------to get the dispersion solution------------------------------------------------------------------------------
def get_disp_array(wmin, wmax, vdel=100, nbin=1000, nhr=100, vres=30):
    c = 3e5 # km/s
    wlist, llist = readlist(wmin, wmax)
    sig = 5 * vdel / c
    w = np.linspace(wmin, wmax, nbin)
    for ii in wlist:
        w1 = ii * (1 - sig)
        w2 = ii * (1 + sig)
        highres = np.linspace(w1, w2, nhr)
        w = np.hstack((w[:np.where(w < w1)[0][-1] + 1], highres, w[np.where(w > w2)[0][0]:]))
    # --------------spectral smearing-----------------------------------------------------------------------------
    new_w = [w[0]]
    while new_w[-1] <= w[-1]:
        new_w.append(new_w[-1] * (1 + vres / c))
    dispsol = np.array(new_w[1:])
    return dispsol, w

# -------------to read in SB99 spectra------------------------------------------------------------------------------
def readSB(wmin, wmax):
    inpSB = open(HOME + '/SB99-v8-02/output/starburst08/starburst08.spectrum', 'r')  # sb08 has cont SF
    speclines = inpSB.readlines()[5:]
    age = [0., 1., 2., 3., 4., 5.]  # in Myr
    col_ar = ['r', 'g', 'b', 'orange', 'cyan', 'k']
    funcar = []
    for (i,a) in enumerate(age):
        cw, cf = [], []
        for line in speclines:
            if float(line.split()[0]) / 1e6 == a:
                if wmin - 150. <= float(line.split()[1]) <= wmax + 150.:
                    cw.append(float(line.split()[1]))  # column 1 is wavelength in A
                    cf.append(10 ** float(line.split()[3]))  # column 3 is stellar continuum in ergs/s/A
        funcar.append(interp1d(cw, cf, kind='cubic'))
        ps = plt.scatter(cw, cf, lw=0, s=10, c=col_ar[i])

        dispsol, w = get_disp_array(wmin, wmax)
        f = funcar[-1](dispsol)
        plt.plot(dispsol, f, lw=0.5, label='%.1FMyr'%a, c=col_ar[i])
    return funcar

# -------------main------------------------------------------------------------------------------
wmin, wmax = 6400., 6783.
fig = plt.figure()
readSB(wmin, wmax) # Angstroms
plt.xlim(wmin, wmax)
plt.xlabel(r'Wavelength ($\AA$)')
plt.ylabel('Flux (ergs/s/A)')
plt.legend()
plt.show(block=False)