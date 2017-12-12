import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os

HOME = os.getenv('HOME')
sys.path.append(HOME + '/Work/astro/mageproject/ayan')
import splot_util as su
from scipy.optimize import curve_fit
import plotobservables as po
from astropy.io import fits
import warnings

warnings.filterwarnings("ignore")
from astropy.stats import gaussian_fwhm_to_sigma as gf2s
import argparse as ap


# -------------------------------------------------------------------------------------------
def fitspaxel(args, logbook, properties):
    wave = np.array(properties.dispsol)  # converting to numpy arrays
    flam = np.array(properties.ppvcube[args.X, args.Y, :])  # in units ergs/s/A/pc^2
    idx = np.where(flam < 0)[0]
    if len(idx) > 0:
        if idx[0] == 0: idx = idx[1:]
        if idx[-1] == len(flam) - 1: idx = idx[:-1]
        flam[idx] = (flam[idx - 1] + flam[idx + 1]) / 2.  # replacing negative fluxes with average of nearest neighbours
    # -------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(17, 5))
    fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.05, right=0.95)
    plt.plot(wave, np.log10(flam), c='k', label='Spectrum at ' + str(args.X) + ', ' + str(args.Y))
    for ii in logbook.wlist:
        plt.axvline(ii, ymin=0.9, c='black')
    plt.ylabel('Log flam in erg/s/A/pc^2')
    plt.xlabel('Wavelength (A)')
    # plt.ylim(32,40)
    plt.xlim(logbook.wmin, logbook.wmax)
    plt.title('Fitted spectrum at pp ' + str(args.X) + ',' + str(args.Y) + ' for ' + logbook.fitsname)
    plt.legend()
    plt.show(block=False)
    # -------------------------------------------------------------------------------------------
    flux, flux_errors = po.fit_all_lines(args, logbook, wave, flam, args.X, args.Y, z=0., z_err=0.0001)
    return flux, flux_errors


# -------------------------------------------------------------------------------------------
if __name__ == '__main__':
    properties, logbook = ap.Namespace(), ap.Namespace()
    parser = ap.ArgumentParser(description="apaxel fitter debugging tool")
    parser.add_argument('--silent', dest='silent', action='store_true')
    parser.set_defaults(silent=False)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--showplot', dest='showplot', action='store_true')
    parser.set_defaults(showplot=False)
    parser.add_argument('--spec_smear', dest='spec_smear', action='store_true')
    parser.set_defaults(spec_smear=False)
    parser.add_argument('--smooth', dest='smooth', action='store_true')
    parser.set_defaults(smooth=False)
    parser.add_argument('--addnoise', dest='addnoise', action='store_true')
    parser.set_defaults(addnoise=False)
    parser.add_argument('--keepprev', dest='keepprev', action='store_true')
    parser.set_defaults(keepprev=False)
    parser.add_argument('--hide', dest='hide', action='store_true')
    parser.set_defaults(hide=False)
    parser.add_argument('--maketheory', dest='maketheory', action='store_true')
    parser.set_defaults(maketheory=False)

    parser.add_argument('--scale_exptime')
    parser.add_argument('--multi_realisation')
    parser.add_argument("--file")
    parser.add_argument("--Om")
    parser.add_argument("--path")
    parser.add_argument('--vdel')
    parser.add_argument('--vdisp')
    parser.add_argument('--vres')
    parser.add_argument('--nhr')
    parser.add_argument('--nres')
    parser.add_argument('--nbin')
    parser.add_argument('--res')
    parser.add_argument('--wmin')
    parser.add_argument('--wmax')
    parser.add_argument("--X")
    parser.add_argument("--Y")
    parser.add_argument("--snr")
    parser.add_argument("--exptime")
    parser.add_argument("--fixed_SNR")
    parser.add_argument('--outfile')
    parser.add_argument("--Zgrad")
    args, leftovers = parser.parse_known_args()

    if args.path is None:
        args.path = HOME + '/Desktop/bpt/'

    if args.file is None:
        args.file = 'DD0600_lgf'  # which simulation to use

    if args.Om is not None:
        args.Om = float(args.Om)
    else:
        args.Om = 0.5

    if args.nhr is not None:
        args.nhr = int(args.nhr)
    else:
        args.nhr = 100  # no. of bins used to resolve the range lamda +/- 5sigma around emission lines

    if args.nbin is not None:
        args.nbin = int(args.nbin)
    else:
        args.nbin = 1000  # no. of bins used to bin the continuum into (without lines)

    if args.nres is not None:
        args.nres = int(args.nres)
    else:
        args.nres = 5  # no. of spectral resolution elements included on either side during fitting a line/group of lines

    if args.vdisp is not None:
        args.vdisp = float(args.vdisp)
    else:
        args.vdisp = 15.  # km/s vel dispersion to be added to emission lines from MAPPINGS while making PPV

    if args.vdel is not None:
        args.vdel = float(args.vdel)
    else:
        args.vdel = 100.  # km/s; vel range in which spectral resolution is higher is sig = 5*vdel/c
        # so wavelength range of +/- sig around central wavelength of line is binned into further nhr bins

    if args.vres is not None:
        args.vres = float(args.vres)
    else:
        args.vres = 30.  # km/s instrumental vel resolution to be considered while making PPV

    if args.res is not None:
        args.res = float(args.res)
    else:
        args.res = 0.02  # kpc: simulation actual resolution

    if args.wmin is not None:
        args.wmin = float(args.wmin)
    else:
        args.wmin = None  # Angstrom; starting wavelength of PPV cube

    if args.wmax is not None:
        args.wmax = float(args.wmax)
    else:
        args.wmax = None  # Angstrom; ending wavelength of PPV cube

    if args.snr is not None:
        args.SNR_thresh = float(args.snr)
    else:
        args.SNR_thresh = None

    if args.Zgrad is not None:
        args.logOHcen, args.logOHgrad = [float(ar) for ar in args.Zgrad.split(',')]
        args.gradtext = '_Zgrad' + str(args.logOHcen) + ',' + str(args.logOHgrad)
    else:
        args.logOHcen, args.logOHgrad = logOHsun, 0.
        args.gradtext = ''

    args.toscreen = True  # debug mode forces toscreen
    args.showplot = True

    args, logbook = po.getfitsname(args, properties)  # name of fits file to be written into
    logbook.resoln = po.c / args.vres

    properties = po.get_disp_array(args, logbook, properties)
    if os.path.exists(logbook.fitsname):
        print 'Reading ppvcube from ' + logbook.fitsname
        properties.ppvcube = fits.open(logbook.fitsname)[0].data
    else:
        print 'File not found: ' + logbook.fitsname
        po.myexit(args)

    if args.X is not None:
        args.X = int(args.X)
    else:
        args.X = np.shape(properties.ppvcube)[0] / 2  # p-p values at which point to extract spectrum from the ppv cube

    if args.Y is not None:
        args.Y = int(args.Y)
    else:
        args.Y = np.shape(properties.ppvcube)[0] / 2  # p-p values at which point to extract spectrum from the ppv cube

    if not np.array(properties.ppvcube[args.X, args.Y, :]).any():
        po.myprint('Chosen spaxel is empty. Select another.', args)
        po.myprint('Non empty spaxels are:', args)
        for i in range(np.shape(properties.ppvcube)[0]):
            for j in range(np.shape(properties.ppvcube)[1]):
                if np.array(properties.ppvcube[i, j, :]).any():
                    print i, j

        po.myexit(args, text='Try again.')
    # -------------------------------------------------------------------------------------------
    flux, flux_errors = fitspaxel(args, logbook, properties)
    print np.shape(properties.ppvcube), args.X, args.Y, args.SNR_thresh, flux / flux_errors  #
    print 'Finished!'
