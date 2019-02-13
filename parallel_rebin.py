# --python utility code which takes a PPV cube as input, convolves its each spectral slice in PARALLEL
# --and writes the convolved cube
# --by Ayam, May, 2017
import time
import datetime
import astropy.convolution as con
from mpi4py import MPI
import numpy as np
import os

HOME = os.getenv('HOME')
import sys

sys.path.append(HOME + '/Work/astro/ayan_codes/enzo_model_code/')
import plotobservables as po
import argparse as ap

parser = ap.ArgumentParser(description="parallel rebinning tool")
from astropy.io import fits
from matplotlib import pyplot as plt
import subprocess

# -------------------------------------------------------------------------------------------
def print_mpi(string, args):
    comm = MPI.COMM_WORLD
    po.myprint('[' + str(comm.rank) + '] {'+subprocess.check_output(['uname -n'],shell=True)[:-1]+'} ' + string + '\n', args)


def print_master(string, args):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        po.myprint('[' + str(comm.rank) + '] ' + string + '\n', args)


# -------------------------------------------------------------------------------------------
def show(map, title):
    map = np.ma.masked_where(map < 0., map)
    plt.figure()
    plt.imshow(map)
    plt.title(title)
    plt.colorbar()


# -----------------------------parallelised--------------------------------------------------------------
if __name__ == '__main__':
    logbook = ap.Namespace()
    parser.add_argument('--silent', dest='silent', action='store_true')
    parser.set_defaults(silent=False)
    parser.add_argument('--toscreen', dest='toscreen', action='store_true')
    parser.set_defaults(toscreen=False)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--saveplot', dest='saveplot', action='store_true')
    parser.set_defaults(saveplot=False)
    parser.add_argument('--hide', dest='hide', action='store_true')
    parser.set_defaults(hide=False)

    parser.add_argument('--file')
    parser.add_argument('--outfile')
    parser.add_argument('--fitsname')
    parser.add_argument('--fitsname_u')
    parser.add_argument('--target_res')
    parser.add_argument('--galsize')
    parser.add_argument('--center')
    parser.add_argument('--cmin')
    parser.add_argument('--cmax')
    parser.add_argument('--xcenter_offset')
    parser.add_argument('--ycenter_offset')
    parser.add_argument('--rebinned_filename')
    parser.add_argument('--rebinned_u_filename')
    args, leftovers = parser.parse_known_args()
    args.galsize = float(args.galsize)
    args.target_res = float(args.target_res)
    args.center = float(args.center)
    args.xcenter_offset = float(args.xcenter_offset)
    args.ycenter_offset = float(args.ycenter_offset)
    if args.cmin is not None: args.cmin = float(args.cmin)
    if args.cmax is not None: args.cmax = float(args.cmax)
    if args.debug: args.toscreen = True  # debug mode forces toscreen

    logbook.fitsname = args.fitsname
    logbook.fitsname_u = args.fitsname_u

    cube = fits.open(logbook.fitsname)[0].data
    cube = np.nan_to_num(cube) * (args.galsize * 1e3/np.shape(cube)[0])**2 # to replace nan values with zeroes and to change cube units from ergs/s/A/pc^2 to ergs/s/A/pixel
    cube_u = fits.open(logbook.fitsname_u)[0].data
    cube_u = np.nan_to_num(cube_u) * (args.galsize * 1e3/np.shape(cube_u)[0])**2 # to replace nan values with zeroes and to change cube units from ergs/s/A/pc^2 to ergs/s/A/pixel
    nslice = np.shape(cube)[2]
    new_shape = int(np.round(args.galsize * 1e3 / args.target_res))
    rebinned_shape = (new_shape, new_shape)
    final_res = args.galsize * 1e3 / new_shape

    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = ' + str(ncores) + '. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)
    print_master('Rebinning cube serially from shape '+str(np.shape(cube)[0])+' to '+str(new_shape)+' cells on each side.', args)
    print_master('Resolution changes from '+str(args.galsize * 1e3/np.shape(cube)[0])+' pc/spaxel to '+str(final_res)+' pc/spaxel.', args)
    rebinned_cube_local = np.zeros((new_shape, new_shape, np.shape(cube)[2]))
    rebinned_cube_u_local = np.zeros((new_shape, new_shape, np.shape(cube)[2]))
    comm.Barrier()
    t_start = MPI.Wtime()  ### Start stopwatch ###

    # domain decomposition
    split_at_cpu = nslice - ncores*(nslice/ncores)
    nper_cpu1 = (nslice / ncores)
    nper_cpu2 = nper_cpu1 + 1
    if rank < split_at_cpu:
        core_start = rank * nper_cpu2
        core_end = (rank+1) * nper_cpu2 - 1
    else:
        core_start = split_at_cpu * nper_cpu2 + (rank - split_at_cpu) * nper_cpu1
        core_end = split_at_cpu * nper_cpu2 + (rank - split_at_cpu + 1) * nper_cpu1 - 1


    print_mpi('Operating on slice ' + str(core_start) + ' to ' + str(core_end) + ': '+str(core_end-core_start+1)+' out of ' + str(nslice) + ' slices', args)

    for k in range(core_start, core_end+1):
        slice_slab = nslice / 6
        if args.debug and k % slice_slab == 0:
            logbook.final_pix_size = args.galsize/np.shape(cube)[0]
            dummy = po.plotmap(cube[:, :, k], 'slice ' + str(k) + ': before rebinning', 'junk', 'ergs/s/A/pixel', args, logbook, makelog=True)
            print_mpi('Integrated flux in slice ' + str(k) + ' before rebinning = %.4E ergs/s/A' % np.sum(cube[:, :, k]), args)
        
        rebinned_slice = po.rebin(cube[:, :, k], rebinned_shape)  # re-bin 2d array before convolving to make things faster
        rebinned_cube_local[:, :, k] = rebinned_slice
        rebinned_u_slice = po.rebin(cube_u[:, :, k], rebinned_shape)  # re-bin 2d array before convolving to make things faster
        rebinned_cube_u_local[:, :, k] = rebinned_u_slice

        if args.debug and k % slice_slab == 0:
            logbook.final_pix_size = args.galsize/np.shape(rebinned_cube_local)[0]
            dummy = po.plotmap(rebinned_cube_local[:, :, k], 'slice ' + str(k) + ': after rebinning', 'junk', 'ergs/s/A/pixel', args, logbook, makelog=True)
            po.mydiag('[' + str(comm.rank) + '] ' + 'Deb 77: in ergs/s/A/pixel: after rebinning', dummy, args)
            print_mpi('Integrated flux in slice '+str(k)+' after convolution (and resampling) = %.4E ergs/s/A'%np.sum(result), args)
        if not args.silent: print_mpi('Rebinned slice ' + str(k) + ' of ' + str(core_end) + ' slices at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)

        if args.debug and slice is not None and k % slice_slab == 0:
            print_mpi('Pausing for 10...', args)
            plt.pause(10)

    comm.Barrier()
    comm.Allreduce(MPI.IN_PLACE, rebinned_cube_local, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, rebinned_cube_u_local, op=MPI.SUM)
    if rank == 0:
        rebinned_cube_local /= final_res ** 2  # to change units from ergs/s/A/pixel to ergs/s/A/pc^2
        rebinned_cube_local = np.ma.masked_where(rebinned_cube_local < 0., rebinned_cube_local)
        po.write_fits(args.rebinned_filename, rebinned_cube_local, args)
        rebinned_cube_u_local /= final_res ** 2  # to change units from ergs/s/A/pixel to ergs/s/A/pc^2
        rebinned_cube_u_local = np.ma.masked_where(rebinned_cube_u_local < 0., rebinned_cube_u_local)
        po.write_fits(args.rebinned_u_filename, rebinned_cube_u_local, args)

    t_diff = MPI.Wtime() - t_start  ### Stop stopwatch ###
    print_master(
        'Parallely: time taken for rebinning of ' + str(nslice) + ' slices with ' + str(ncores) + ' cores= ' + str(
            t_diff / 60.) + ' min', args)
