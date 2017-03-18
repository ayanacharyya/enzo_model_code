import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.append('/Users/acharyya/Work/astro/mageproject/ayan')
import splot_util as su
from scipy.optimize import curve_fit
import plotobservables as p
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore")
from astropy.stats import gaussian_fwhm_to_sigma as gf2s
import argparse as ap

#-------------------------------------------------------------------------------------------
def fitspaxel(ppvcube, dispsol, wlist, llist, X, Y, vres, wmin=None ,wmax=None, filename=None, silent=True):
    if wmin is None: wmin = wlist[0]-50.
    if wmax is None: wmax = wlist[-1]+50.
    resoln = c/vres
    spec = ppvcube[X,Y,:]
    idx = np.where(spec<0)[0]
    if len(idx) > 0:
        if idx[0] == 0: idx = idx[1:]
        if idx[-1] == len(spec)-1 : idx = idx[:-1]
        spec[idx]=(spec[idx-1] + spec[idx+1])/2.
    flam=spec/dispsol
    #-------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(17,5))
    fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.05, right=0.95)
    for i in wlist:
        plt.axvline(i,ymin=0.9,c='black')    
    plt.ylabel('Log flam in erg/s/A/pc^2') #label of color bar
    plt.plot(dispsol,np.log10(flam))
    plt.xlabel('Wavelength (A)')
    plt.ylim(25,32)
    plt.xlim(wmin,wmax)
    plt.title('Fitted spectrum at pp '+str(X)+','+str(Y)+' for '+filename)
    plt.show(block=False)
    #-------------------------------------------------------------------------------------------
    flux, flux_errors = p.fit_all_lines(wlist, llist, dispsol,flam,resoln,X,Y,nres=nres, z=0, z_err=z_err,silent=silent, showplot=True)
    return flux, flux_errors
#-------------------------------------------------------------------------------------------
path = '/Users/acharyya/Desktop/bpt/'
vdel, vdisp, nhr, nbin, c, nres, z_err= 100, 15, 100, 1000, 3e5, 5., 0.0004
if __name__ == '__main__':
    parser = ap.ArgumentParser(description="observables generating tool")
    parser.add_argument("--X")
    parser.add_argument("--Y")
    parser.add_argument("--wmin")
    parser.add_argument("--wmax")
    parser.add_argument("--vres")
    parser.add_argument('--silent', dest='silent', action='store_true')
    parser.set_defaults(silent=False)
    args, leftovers = parser.parse_known_args()
    fn = sys.argv[1]
    if fn[-5:] != '.fits': fn += '.fits'
    ppvcube = fits.open(path+fn)[0].data
    if len(np.shape(ppvcube)) < 3: 
        print 'This fits file is not a cube. Exitting.'
        sys.exit()
    ppvcube = np.ma.masked_where(ppvcube==0, ppvcube)
    #-------------------------------------------------------------------------------------------
    if args.X is not None:
        X = float(args.X)
    else:
        X = np.shape(ppvcube)[0]/2 #p-p values at which point to extract spectrum from the ppv cube
    if args.Y is not None:
        Y = float(args.Y)
    else:
        Y = np.shape(ppvcube)[0]/2 #p-p values at which point to extract spectrum from the ppv cube
    if args.vres is not None:
        vres = float(args.vres)
    else:
        vres = 600 #
    if args.wmin is not None:
        wmin = float(args.wmin)
    else:
        wmin = None #Angstrom; starting wavelength of PPV cube

    if args.wmax is not None:
        wmax = float(args.wmax)
    else:
        wmax = None #Angstrom; ending wavelength of PPV cube
    #-------------------------------------------------------------------------------------------
    w, dummy2, dummy3, new_w, dummy4, wlist, llist, dummy7 = p.get_disp_array(vdel, vdisp, vres, nhr, wmin=wmin, wmax=wmax, nbin=nbin, spec_smear=True)
    dispsol = np.array(new_w[1:])
    flux, flux_errors = fitspaxel(ppvcube, dispsol, wlist, llist, X, Y, vres, wmin=wmin ,wmax=wmax, filename=fn, silent=args.silent)
    print 'Finished!'