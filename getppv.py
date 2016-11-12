import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore")
import argparse as ap
parser = ap.ArgumentParser(description="observables generating tool")
#-------------------------------------------------------------------------------------------
def anim_cube(ppvcube, cmin, cmax, galsize=26.0, pause=0.1, lim=None):   
    ppvcube = np.ma.masked_where(np.log10(ppvcube)<cmin, ppvcube)
    res = galsize/np.shape(ppvcube)[0] #kpc/pix
    if lim is None: lim = np.shape(ppvcube)[2]
    ylab = 'Log(flux)'
    xlab = 'rest-wavelength slice'
    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.1, left=0.1, right=0.9)
    for k in range(lim):
        if 'ax' in locals(): fig.delaxes(ax)
        ax = plt.subplot(111)
        p = ax.imshow(np.log10(ppvcube[:,:,k]), cmap='rainbow',vmin=cmin,vmax=cmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_xticklabels([i*res - galsize/2 for i in list(ax.get_xticks())])
        ax.set_yticklabels([i*res - galsize/2 for i in list(ax.get_yticks())])
        ax.set_ylabel('y(kpc)')
        ax.set_xlabel('x(kpc)')
        plt.colorbar(p, cax=cax).set_label(ylab)
        ax.set_title('Slice '+str(k+1)+' of '+str(lim))

        ax2 = fig.add_axes([0.54,0.74,0.3,0.1])
        ax2.plot(np.arange(np.shape(ppvcube)[2]), np.log10(np.sum(ppv,axis=(0,1))),lw=1)
        ax2.set_title('Total spectrum', fontsize=10)
        plt.gca().axes.get_xaxis().set_visible(False)        
        plt.gca().axes.get_yaxis().set_visible(False)        
        l = ax2.axvline(k, c='r')

        plt.draw()
        plt.pause(0.1)
#-------------------------------------------------------------------------------------------
parser.add_argument("--cmin")
parser.add_argument("--cmax")
parser.add_argument("--path")
parser.add_argument("--galsize")
parser.add_argument("--pause")
parser.add_argument("--lim")
args, leftovers = parser.parse_known_args()

if args.cmin is not None:
    cmin = float(args.cmin)
else:
    cmin = None

if args.cmax is not None:
    cmax = float(args.cmax)
else:
    cmax = None

if args.galsize is not None:
    galsize = float(args.galsize)
else:
    galsize = 26.0 #kpc

if args.pause is not None:
    pause = float(args.pause)
else:
    pause = 0.1 #milli sec?

if args.lim is not None:
    lim = int(args.lim)
else:
    lim = None #slices till which to show plot

if args.path is not None:
    path = args.path
    print 'Using path=', path
else:
    path = '/Users/acharyya/Desktop/bpt/' #which path to use
    print 'Path not specified. Using default', path, '. Use --path option to specify path.'

#-------------------------------------------------------------------------------------------
fn = sys.argv[1]
if fn[-5:] is not '.fits': fn += '.fits'
ppv = fits.open(path+fn)[0].data
if cmin is None: cmin = np.min(np.log10(ppv))
if cmax is None: cmax = np.max(np.log10(ppv))     
anim_cube(ppv, cmin, cmax, galsize=galsize, pause=pause, lim=lim)
