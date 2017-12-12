#--python utility code which takes a final PPV cube as input,fits spectra of every spaxel in PARALLEL
#--and writes the resultant fitted line map cube and errror cube
#--by Ayam, May, 2017
import time
import datetime
import os
HOME = os.getenv('HOME')
import sys
sys.path.append(HOME+'/Work/astro/ayan_codes/mageproject/ayan/')
sys.path.append(HOME+'/Work/astro/ayan_codes/enzo_model_code/')
import splot_util as su
import plotobservables as po
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma as gf2s
from scipy.optimize import curve_fit
from mpi4py import MPI
from astropy.io import fits
import argparse as ap
parser = ap.ArgumentParser(description="parallel cube line-fitting tool")
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
    parser.add_argument('--addnoise', dest='addnoise', action='store_true')
    parser.set_defaults(addnoise=False)
    parser.add_argument('--contsub', dest='contsub', action='store_true')
    parser.set_defaults(contsub=False)

    parser.add_argument('--vdel')
    parser.add_argument('--vdisp')
    parser.add_argument('--vres')
    parser.add_argument('--nhr')
    parser.add_argument('--nbin')
    parser.add_argument('--nres')
    parser.add_argument('--wmin')
    parser.add_argument('--wmax')
    parser.add_argument('--fitsname')
    parser.add_argument('--fitsname_u')
    parser.add_argument('--no_noise_fitsname')
    parser.add_argument('--fittedcube')
    parser.add_argument('--fittederror')
    parser.add_argument('--outfile')
    parser.add_argument('--oneHII')
    parser.add_argument('--vmask')
    args, leftovers = parser.parse_known_args()
    args.vdel = float(args.vdel)
    args.vdisp = float(args.vdisp)
    args.vres = float(args.vres)
    args.nhr = int(args.nhr)
    args.nbin = int(args.nbin)
    args.nres = int(args.nres)
    args.vmask = float(args.vmask)
    if args.oneHII is not None: args.oneHII = int(args.oneHII)
    if args.debug: args.toscreen = True #debug mode forces toscreen
    
    logbook.wlist, logbook.llist = po.readlist()
    logbook.wmin = float(args.wmin)
    logbook.wmax = float(args.wmax)
    logbook.llist = logbook.llist[np.where(np.logical_and(logbook.wlist > logbook.wmin, logbook.wlist < logbook.wmax))] #truncate linelist as per wavelength range
    logbook.wlist = logbook.wlist[np.where(np.logical_and(logbook.wlist > logbook.wmin, logbook.wlist < logbook.wmax))]
    logbook.fitsname = args.fitsname
    logbook.fitsname_u = args.fitsname_u
    logbook.no_noise_fitsname = args.no_noise_fitsname
    logbook.fittedcube = args.fittedcube
    logbook.fittederror = args.fittederror
    
    properties = po.get_disp_array(args, logbook, properties)
    properties.ppvcube = fits.open(logbook.fitsname)[0].data
    #properties.ppvcube = fits.open(logbook.no_noise_fitsname)[0].data #
    properties.ppvcube_u = fits.open(logbook.fitsname_u)[0].data
    x = np.shape(properties.ppvcube)[0]
    y = np.shape(properties.ppvcube)[1]
    ncells = x*y
    properties.mapcube = np.zeros((x,y, len(logbook.wlist)))
    properties.errorcube = np.zeros((x,y, len(logbook.wlist)))
    logbook.resoln = po.c/args.vres if args.spec_smear else po.c/args.vdisp
    
    
    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = '+str(ncores)+'. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)
    mapcube_local = np.zeros(np.shape(properties.mapcube))
    errorcube_local = np.zeros(np.shape(properties.errorcube))
    comm.Barrier()
    t_start = MPI.Wtime() ### Start stopwatch ###

    core_start = rank * (ncells/ncores)
    core_end   = (rank+1) * (ncells/ncores)       
    if (rank == ncores-1): core_end = ncells # last PE gets the rest
    if args.debug: print_mpi('Operating on cell '+str(core_start)+' to '+str(core_end)+' out of '+str(ncells)+' cells', args)
    
    for k in range(core_start, core_end):
        i, j = k/x, k%x
        wave = np.array(properties.dispsol)  #converting to numpy arrays
        flam = np.array(properties.ppvcube[i,j,:]) #in units ergs/s/A/pc^2
        flam_u = np.array(properties.ppvcube_u[i,j,:]) #in units ergs/s/A/pc^2
        if (np.array(flam)>0.).any():
            lastgood, lastgood_u = flam[0], flam_u[0]
            for ind,ff in enumerate(flam):
                if not np.isnan(ff):
                    lastgood, lastgood_u = ff, flam_u[ind]
                    continue
                else:
                    flam[ind] = lastgood #to get rid of NANs in flam
                    flam_u[ind] = lastgood_u #to get rid of NANs in flam_u
        
            try:
                cont, cont_u = po.fitcont(wave, flam, flam_u, args, logbook) #cont in units ergs/s/A/pc^2
                if args.contsub:
                    flam_u = np.sqrt(flam_u**2 + cont_u**2)
                    flam -= cont
                else:
                    flam_u = np.sqrt((flam_u/cont)**2 + (flam*cont_u/(cont**2))**2) #flam_u = dimensionless
                    flam /= cont #flam = dimensionless
            except:
                print_mpi('parallel_fitting: line130: continuum fitting failed for pixel %d,%d: setting flam to zero.\n'%(i,j), args)
                cont = np.zeros(len(flam))
                cont_u = np.zeros(len(flam))
                flam_u = np.zeros(len(flam))
                flam = np.zeros(len(flam))
        else:
            cont = np.zeros(len(flam))
            cont_u = np.zeros(len(flam))
        
        if args.debug:
            print_mpi('Deb106: For pixel %d,%d\n'%(i,j), args)
            print_mpi('Deb110: flam in ergs/s/pc^2/A: Median, stdev, max, min= '+str(np.median(flam))+','+str(np.std(flam))+','+\
str(np.max(flam))+','+str(np.min(flam))+'\n', args)

        mapcube_local[i,j,:], errorcube_local[i,j,:] = po.fit_all_lines(args, logbook, wave, flam, flam_u, cont, i, j, z=0., z_err=0.0001)
        if not args.silent: print_mpi('Fitted cell '+str(k)+' i.e. cell '+str(i)+','+str(j)+' of '+str(core_end)+\
        ' cells at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)
    comm.Barrier()
    comm.Allreduce(mapcube_local, properties.mapcube, op=MPI.SUM)
    comm.Allreduce(errorcube_local, properties.errorcube, op=MPI.SUM)
   
    if rank ==0:
        if args.debug:
            po.myprint('Deb120: Trying to calculate some statistics on the cube of shape ('+str(np.shape(properties.mapcube)[0])+','+str(np.shape(properties.mapcube)[1])+','+str(np.shape(properties.mapcube)[2])+'), please wait...', args)
            po.mydiag('Deb121: in ergs/s/pc^2: for mapcube',properties.mapcube, args)
            po.myprint('Deb120: Trying to calculate some statistics on the cube of shape ('+str(np.shape(properties.errorcube)[0])+','+str(np.shape(properties.errorcube)[1])+','+str(np.shape(properties.errorcube)[2])+'), please wait...', args)
            po.mydiag('Deb121: in ergs/s/pc^2: for mapcube',properties.errorcube, args)
        po.write_fits(logbook.fittedcube, properties.mapcube, args)        
        po.write_fits(logbook.fittederror, properties.errorcube, args)    
            
    t_diff = MPI.Wtime()-t_start ### Stop stopwatch ###
    print_master('Parallely: time taken for fitting of '+str(ncells)+' cells with '+str(ncores)+' cores= '+ str(t_diff/60.)+' min', args)
