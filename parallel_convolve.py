#--python utility code which takes a PPV cube as input, convolves its each spectral slice in PARALLEL
#--and returns the convolved cube
#--by Ayam, May, 2017
import time
import datetime
import astropy.convolution as con
from mpi4py import MPI
import numpy as np
import os
HOME = os.getenv('HOME')
import sys
sys.path.append(HOME+'/models/enzo_model_code/')
import plotobservables as po
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
    parser.add_argument('--silent', dest='silent', action='store_true')
    parser.set_defaults(silent=False)
    parser.add_argument('--toscreen', dest='toscreen', action='store_true')
    parser.set_defaults(toscreen=False)

    parser.add_argument('--sig')
    parser.add_argument('--pow')
    parser.add_argument('--size')
    parser.add_argument('--ker')
    parser.add_argument('--convolved_filename')
    parser.add_argument('--outfile')
    parser.add_argument('--binned_cubename')
    args, leftovers = parser.parse_known_args()
    sig = int(args.sig)
    pow = float(args.pow)
    size = int(args.size)
    ker = args.ker

    cube = fits.open(args.binned_cubename)[0].data
    nslice = np.shape(cube)[2]
    convolved_cube = np.zeros(np.shape(cube))
    if ker == 'gauss': kernel = con.Gaussian2DKernel(sig, x_size = size, y_size = size)
    elif ker == 'moff': kernel = con.Moffat2DKernel(sig, pow, x_size = size, y_size = size)

    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    if not args.silent: print_master('Total number of MPI ranks = '+str(ncores)+'. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)
    convolved_cube_local = np.zeros(np.shape(cube))
    comm.Barrier()
    t_start = MPI.Wtime() ### Start stopwatch ###

    core_start = rank * (nslice/ncores)
    core_end   = (rank+1) * (nslice/ncores)       
    if (rank == ncores-1): core_end = nslice # last PE gets the rest
    for k in range(core_start, core_end):
        convolved_cube_local[:,:,k] = con.convolve_fft(cube[:,:,k], kernel, boundary = 'fill', fill_value = 0.0, normalize_kernel=True)        
        if not args.silent: print_mpi('Convolved slice '+str(k)+' of '+str(nslice)+' slices at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)
    comm.Barrier()
    comm.Allreduce(convolved_cube_local, convolved_cube, op=MPI.SUM)
    if rank ==0:
        convolved_cube = np.ma.masked_where(convolved_cube <0., convolved_cube)    
        po.write_fits(args.convolved_filename, convolved_cube, args)
            
    t_diff = MPI.Wtime()-t_start ### Stop stopwatch ###
    if not args.silent: print_master('Parallely: time taken for convolution of '+str(nslice)+' slices with '+str(ncores)+' cores= '+ str(t_diff/60.)+' min', args)
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
