#--python utility code which takes a PPV cube as input, convolves its each spectral slice in PARALLEL
#--and writes the convolved cube
#--by Ayam, May, 2017
import time
import datetime
import astropy.convolution as con
from mpi4py import MPI
import numpy as np
import os
HOME = os.getenv('HOME')
import sys
sys.path.append(HOME+'/Work/astro/ayan_codes/enzo_model_code/')
import plotobservables_old as po
import argparse as ap
parser = ap.ArgumentParser(description="parallel convolution tool")
from astropy.io import fits
from matplotlib import pyplot as plt
#-------------------------------------------------------------------------------------------
def print_mpi(string, args):
    comm = MPI.COMM_WORLD
    po.myprint('['+str(comm.rank)+'] '+string+'\n', args)

def print_master(string, args):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        po.myprint('['+str(comm.rank)+'] '+string+'\n', args)
#-------------------------------------------------------------------------------------------
def show(map, title):
    map = np.ma.masked_where(map <0., map)
    plt.figure()
    plt.imshow(map)
    plt.title(title)
    plt.colorbar()
#-----------------------------parallelised--------------------------------------------------------------
if __name__ == '__main__':
    logbook = ap.Namespace()
    parser.add_argument('--silent', dest='silent', action='store_true')
    parser.set_defaults(silent=False)
    parser.add_argument('--toscreen', dest='toscreen', action='store_true')
    parser.set_defaults(toscreen=False)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--addnoise', dest='addnoise', action='store_true')
    parser.set_defaults(addnoise=False)
    parser.add_argument('--saveplot', dest='saveplot', action='store_true')
    parser.set_defaults(saveplot=False)
    parser.add_argument('--hide', dest='hide', action='store_true')
    parser.set_defaults(hide=False)

    parser.add_argument('--sig')
    parser.add_argument('--pow')
    parser.add_argument('--size')
    parser.add_argument('--ker')
    parser.add_argument('--convolved_filename')
    parser.add_argument('--outfile')
    parser.add_argument('--binned_cubename')
    parser.add_argument('--fitsname')
    parser.add_argument('--exptime')
    parser.add_argument('--final_pix_size')
    parser.add_argument('--fixed_SNR')
    parser.add_argument('--galsize')
    parser.add_argument('--cmin')
    parser.add_argument('--cmax')
    args, leftovers = parser.parse_known_args()
    args.sig = int(args.sig)
    args.pow = float(args.pow)
    args.size = int(args.size)
    args.galsize = float(args.galsize)
    if args.fixed_SNR is not None: args.fixed_SNR = float(args.fixed_SNR)
    if args.cmin is not None: args.cmin = float(args.cmin)
    if args.cmax is not None: args.cmax = float(args.cmax)
    if args.debug: args.toscreen = True #debug mode forces toscreen

    logbook.exptime = float(args.exptime)
    logbook.final_pix_size = float(args.final_pix_size)
    logbook.fitsname = args.fitsname

    cube = fits.open(args.binned_cubename)[0].data
    nslice = np.shape(cube)[2]
    convolved_cube = np.zeros(np.shape(cube))
    if args.ker == 'gauss': kernel = con.Gaussian2DKernel(args.sig, x_size = args.size, y_size = args.size)
    elif args.ker == 'moff': kernel = con.Moffat2DKernel(args.sig, args.pow, x_size = args.size, y_size = args.size)

    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = '+str(ncores)+'. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)
    convolved_cube_local = np.zeros(np.shape(cube))
    comm.Barrier()
    t_start = MPI.Wtime() ### Start stopwatch ###

    core_start = rank * (nslice/ncores)
    core_end   = (rank+1) * (nslice/ncores)       
    if (rank == ncores-1): core_end = nslice # last PE gets the rest
    if not args.silent: print_mpi('Operating on slice '+str(core_start)+' to '+str(core_end)+' out of '+str(nslice)+' cells', args)

    for k in range(core_start, core_end):
        slice_slab = 100
        if args.debug and k%slice_slab==0:
            dummy = po.plotmap(cube[:,:,k], 'slice '+str(k)+': before convolution', 'junk', 'ergs/s/A/pixel', args, logbook, islog=True)
            po.mydiag('['+str(comm.rank)+'] '+'Deb 77: in ergs/s/A/pixel: before convoluion', dummy, args)
        convolved_cube_local[:,:,k] = con.convolve_fft(cube[:,:,k], kernel, boundary = 'fill', fill_value = 0.0, normalize_kernel=True)        
        if args.debug and k%slice_slab==0:
            dummy = po.plotmap(convolved_cube_local[:,:,k], 'slice '+str(k)+': after convolution', 'junk', 'ergs/s/A/pixel', args, logbook, islog=True)
            po.mydiag('['+str(comm.rank)+'] '+'Deb 77: in ergs/s/A/pixel: after convoluion', dummy, args)
        if not args.silent: print_mpi('Convolved slice '+str(k)+' of '+str(core_end)+' slices at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)
    comm.Barrier()
    comm.Allreduce(convolved_cube_local, convolved_cube, op=MPI.SUM)
    if rank ==0:
        convolved_cube = np.ma.masked_where(convolved_cube <0., convolved_cube)    
        po.write_fits(args.convolved_filename, convolved_cube, args)
            
    t_diff = MPI.Wtime()-t_start ### Stop stopwatch ###
    print_master('Parallely: time taken for convolution of '+str(nslice)+' slices with '+str(ncores)+' cores= '+ str(t_diff/60.)+' min', args)
#-------------------------------------------------------------------------------------------
    '''
    # rank 0 needs buffer space to gather data
    if comm.rank == 0:
        gbuf = np.empty( (np.shape(convolved_cube)[0], np.shape(convolved_cube)[1], ncores) )
    else:
        gbuf = None               
    for i_base in range(0, nslice, ncores):
        k = i_base + comm.rank
        convolved_cube_local[:,:,k] = con.convolve_fft(cube[:,:,k], kernel, boundary = 'fill', fill_value = 0.0, normalize_kernel=True)
            
            # rank 0 gathers whitened images 
            comm.Gather(
                [map, MPI.DOUBLE],   # send buffer
                [gbuf, MPI.DOUBLE],  # receive buffer
                root=0               # rank 0 is root the root-porcess
                )
            print_mpi('Convolved slice '+str(i)+' of '+str(nslice)+' slices') #
    # rank 0 has to write into the master array
    if rank == 0: # Sequentially append each of the images
        for r in range(ncores):
        convolved_cube[:,:,k:k+ncores] = gbuf
    '''            
