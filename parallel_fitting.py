#--python utility code which takes a PPV cube as input, convolves its each spectral slice in PARALLEL
#--and returns the convolved cube
#--by Ayam, May, 2017
import time
import datetime
import os
HOME = os.getenv('HOME')
import sys
sys.path.append(HOME+'/Work/astro/mageproject/ayan/')
sys.path.append(HOME+'/models/enzo_model_code/')
import splot_util as su
import plotobservables as po
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma as gf2s
from scipy.optimize import curve_fit
from mpi4py import MPI
from astropy.io import fits
import argparse as ap
parser = ap.ArgumentParser(description="parallel cube fitting tool")
#-------------------------------------------------------------------------------------------
def print_mpi(string, args):
    comm = MPI.COMM_WORLD
    po.myprint('['+str(comm.rank)+'] '+string+'\n', args)

def print_master(string, args):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        po.myprint('['+str(comm.rank)+'] '+string+'\n', args)

#-----------------------------------------
if __name__ == '__main__':
    properties, logbook = ap.Namespace(), ap.Namespace()
    parser.add_argument('--silent', dest='silent', action='store_true')
    parser.set_defaults(silent=False)
    parser.add_argument('--spec_smear', dest='spec_smear', action='store_true')
    parser.set_defaults(spec_smear=False)
    parser.add_argument('--toscreen', dest='toscreen', action='store_true')
    parser.set_defaults(toscreen=False)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--showplot', dest='showplot', action='store_true')
    parser.set_defaults(showplot=False)

    parser.add_argument('--vdel')
    parser.add_argument('--vdisp')
    parser.add_argument('--vres')
    parser.add_argument('--nhr')
    parser.add_argument('--nbin')
    parser.add_argument('--wmin')
    parser.add_argument('--wmax')
    parser.add_argument('--fitsname')
    parser.add_argument('--fittedcube')
    parser.add_argument('--fittederror')
    parser.add_argument('--outfile')
    args, leftovers = parser.parse_known_args()
    args.vdel = float(args.vdel)
    args.vdisp = float(args.vdisp)
    args.vres = float(args.vres)
    args.nhr = int(args.nhr)
    args.nbin = int(args.nbin)

    logbook.wlist, logbook.llist = po.readlist()
    logbook.wmin = float(args.wmin)
    logbook.wmax = float(args.wmax)
    logbook.llist = logbook.llist[np.where(np.logical_and(logbook.wlist > logbook.wmin, logbook.wlist < logbook.wmax))] #truncate linelist as per wavelength range
    logbook.wlist = logbook.wlist[np.where(np.logical_and(logbook.wlist > logbook.wmin, logbook.wlist < logbook.wmax))]
    logbook.fitsname = args.fitsname
    logbook.fittedcube = args.fittedcube
    logbook.fittederror = args.fittederror
    
    properties = po.get_disp_array(args, logbook, properties)
    properties.ppvcube = fits.open(logbook.fitsname)[0].data
    x = np.shape(properties.ppvcube)[0]
    y = np.shape(properties.ppvcube)[1]
    ncells = x*y
    properties.mapcube = np.zeros((x,y, len(logbook.wlist)))
    properties.errorcube = np.zeros((x,y, len(logbook.wlist)))
    logbook.resoln = po.c/args.vres
    
    
    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    if not args.silent: print_master('Total number of MPI ranks = '+str(ncores)+'. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)
    mapcube_local = np.zeros(np.shape(properties.mapcube))
    errorcube_local = np.zeros(np.shape(properties.errorcube))
    comm.Barrier()
    t_start = MPI.Wtime() ### Start stopwatch ###

    core_start = rank * (ncells/ncores)
    core_end   = (rank+1) * (ncells/ncores)       
    if (rank == ncores-1): core_end = ncells # last PE gets the rest
    for k in range(core_start, core_end):
        i, j = k/x, k%x
        mapcube_local[i,j,:], errorcube_local[i,j,:] = po.fit_all_lines(args, logbook, properties, properties.ppvcube[i,j,:], i, j, nres=5, z=0, z_err=0.0001)
        if not args.silent: print_mpi('Fitted cell '+str(k)+' i.e. cell '+str(i)+','+str(j)+' of '+str(ncells)+' cells', args)
    comm.Barrier()
    comm.Allreduce(mapcube_local, properties.mapcube, op=MPI.SUM)
    comm.Allreduce(errorcube_local, properties.errorcube, op=MPI.SUM)
    if rank ==0:
        po.write_fits(logbook.fittedcube, properties.mapcube, args)        
        po.write_fits(logbook.fittederror, properties.errorcube, args)    
            
    t_diff = MPI.Wtime()-t_start ### Stop stopwatch ###
    if not args.silent: print_master('Parallely: time taken for fitting of '+str(ncells)+' cells with '+str(ncores)+' cores= '+ str(t_diff/60.)+' min', args)
