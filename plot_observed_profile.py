from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
import os
HOME = os.getenv('HOME')
import pandas as pd
import sys


inpath = HOME + '/Downloads/'
col_ar = ['r', 'b', 'orange', 'darkgreen', 'brown']
fig, ax = plt.subplots()
if len(sys.argv) > 1 : survey = sys.argv[1]
else: survey = 'typhoon'

# ------------SAMI data-------------------- #
if survey == 'sami' or survey == 'SAMI':
    line_arr = ['NII6583', 'SII6716', 'OII3727', 'HALPHA']
    galaxy = '288461' # '79635'
    z = 0.00599 # 0.0408
    dist = z * 1e3 * 3e5 / 70. # kpc
    print 'For '+survey+' galaxy', galaxy

    for ind,line in enumerate(line_arr):
        filename = galaxy + '-line_emission_map-'+line+'-1_comp-V03.fits'
        file = fits.open(inpath + filename)
        data = file[0].data[0] * 1e-16 # ergs/s/cm^2
        header = file[0].header
        if not np.isfinite(data).any():
            print 'Data for ', line, 'does not exist. Skipping this line..'
            continue
        '''
        plt.figure()
        plt.imshow(np.log10(data), cmap='rainbow')
        plt.colorbar()
        plt.title(line)
        '''
        g = np.shape(data)[0]
        delta_pix = np.abs(float(header['CDELT1'])) # in deg
        kpc_per_pix = dist * delta_pix * np.pi/180.
        center_pix_offset = np.abs(float(header['CRPIX1']) - g/2.)
        b = np.linspace(-g / 2 + 1, g / 2, g) * kpc_per_pix  # in kpc
        d = np.sqrt((b[:, None] - delta_pix * center_pix_offset) ** 2 + (b - delta_pix * center_pix_offset) ** 2)  # kpc
        rad = np.ma.compressed(np.ma.masked_where(~(np.isfinite(data.flatten())), d.flatten()))
        flux = np.ma.compressed(np.ma.masked_where(~(np.isfinite(data.flatten())), data.flatten()))
        rad = np.ma.compressed(np.ma.masked_where((flux <= 0), rad))
        flux = np.ma.compressed(np.ma.masked_where((flux <= 0), flux))
        ax.scatter(rad, np.log10(flux), s=5, lw=0, c=col_ar[ind])
        linefit = np.polyfit(rad, np.log10(flux), 1)
        print line, linefit
        ax.plot(rad, np.poly1d(linefit)(rad), c=col_ar[ind], label=line+' slope=%.3F dex/kpc'%linefit[0])
        ax.legend()

    ax.set_xlim(0,0.7*g*kpc_per_pix) # kpc
    ax.set_xlabel('Galactocentric radius (kpc)')
    ax.set_ylabel('log (flux ergs/s/cm^2)')
    ax.set_title(survey + ' galaxy ID '+galaxy)
# --------------------TYPHOON data---------------------- #
elif survey == 'typhoon' or survey == 'TYPHOON':
    galaxy = 'N5236'# 'N5068' #'N2835'#
    filename = galaxy + '_lineflux.txt'
    line_arr = ['NII6583', 'SII6716', 'OII3727', 'HA']
    data = pd.read_table(inpath + filename, sep=',')
    print 'For '+survey+' galaxy', galaxy

    for ind,line in enumerate(line_arr):
        rad = np.ma.compressed(np.ma.masked_where(~(np.isfinite(data[line])), data['RADIUS(R/Re)'])) # in R_e
        flux = np.ma.compressed(np.ma.masked_where(~(np.isfinite(data[line])), data[line])) # flambda, in units of 10^-17 ergs/s/cm^2/spaxel with 1.65x1.65 arcsec spaxels
        rad = np.ma.compressed(np.ma.masked_where((flux <= 0), rad))
        flux = np.ma.compressed(np.ma.masked_where((flux <= 0), flux))
        flux *= 1e-17 # now in units of ergs/s/cm^2/spaxel
        ax.scatter(rad, np.log10(flux), s=1, lw=0, c=col_ar[ind])
        linefit = np.polyfit(rad, np.log10(flux), 1)
        print line, linefit
        ax.plot(rad, np.poly1d(linefit)(rad), c=col_ar[ind], label=line+' slope=%.3F dex/Re'%linefit[0], lw=3)
        ax.legend()

    ax.set_xlim(0,4) # Re
    ax.set_xlabel('Galactocentric radius (R_e)')
    ax.set_ylabel('log (flux)')
    ax.set_title(survey + ' galaxy ID '+galaxy)
else:
    print 'Choose from either TYPHOON or SAMI surveys.'


plt.show(block=False)


