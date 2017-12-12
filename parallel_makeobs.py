# --python utility code which takes an intermediate PPV cube as input, adds noise and/or saturation limits etc. to every spectral slice in PARALLEL
# --and writes the resulting ppv cube
# --by Ayam, June, 2017
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

parser = ap.ArgumentParser(description="parallel makeobservable/addnoise tool")
from astropy.io import fits
from matplotlib import pyplot as plt


# -------------------------------------------------------------------------------------------
def print_mpi(string, args):
    comm = MPI.COMM_WORLD
    po.myprint('[' + str(comm.rank) + '] ' + string + '\n', args)


def print_master(string, args):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        po.myprint('[' + str(comm.rank) + '] ' + string + '\n', args)


# -----------------------------parallelised--------------------------------------------------------------
if __name__ == '__main__':
    properties, logbook = ap.Namespace(), ap.Namespace()
    parser.add_argument('--silent', dest='silent', action='store_true')
    parser.set_defaults(silent=False)
    parser.add_argument('--toscreen', dest='toscreen', action='store_true')
    parser.set_defaults(toscreen=False)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--addnoise', dest='addnoise', action='store_true')
    parser.set_defaults(addnoise=False)
    parser.add_argument('--noskynoise', dest='noskynoise', action='store_true')
    parser.set_defaults(noskynoise=False)
    parser.add_argument('--spec_smear', dest='spec_smear', action='store_true')
    parser.set_defaults(spec_smear=False)
    parser.add_argument('--parallel', dest='parallel', action='store_true')
    parser.set_defaults(parallel=False)
    parser.add_argument('--saveplot', dest='saveplot', action='store_true')
    parser.set_defaults(saveplot=False)
    parser.add_argument('--hide', dest='hide', action='store_true')
    parser.set_defaults(hide=False)

    parser.add_argument('--last_cubename')
    parser.add_argument('--vdel')
    parser.add_argument('--vdisp')
    parser.add_argument('--vres')
    parser.add_argument('--nhr')
    parser.add_argument('--nbin')
    parser.add_argument('--nres')
    parser.add_argument('--wmin')
    parser.add_argument('--wmax')
    parser.add_argument('--epp')
    parser.add_argument('--gain')
    parser.add_argument('--exptime')
    parser.add_argument('--final_pix_size')
    parser.add_argument('--outfile')
    parser.add_argument('--fitsname')
    parser.add_argument('--fitsname_u')
    parser.add_argument('--skynoise_cubename')
    parser.add_argument('--fixed_SNR')
    parser.add_argument('--fixed_noise')
    parser.add_argument('--scale_exp_SNR')
    parser.add_argument('--galsize')
    parser.add_argument('--cmin')
    parser.add_argument('--cmax')
    parser.add_argument('--snr_cmin')
    parser.add_argument('--snr_cmax')
    parser.add_argument('--rad')
    parser.add_argument('--dist')
    parser.add_argument('--scale_length')
    parser.add_argument('--flux_to_scale')
    parser.add_argument('--sky_norm')
    parser.add_argument('--multi_realisation')
    parser.add_argument('--xcenter_offset')
    parser.add_argument('--ycenter_offset')
    args, leftovers = parser.parse_known_args()
    args.vdel = float(args.vdel)
    args.vdisp = float(args.vdisp)
    args.vres = float(args.vres)
    args.nhr = int(args.nhr)
    args.nbin = int(args.nbin)
    args.nres = int(args.nres)
    args.el_per_phot = float(args.epp)
    args.gain = float(args.gain)
    args.galsize = float(args.galsize)
    args.rad = float(args.rad)
    args.multi_realisation = int(args.multi_realisation)
    args.xcenter_offset = float(args.xcenter_offset)
    args.ycenter_offset = float(args.ycenter_offset)
    if args.fixed_SNR is not None: args.fixed_SNR = float(args.fixed_SNR)
    if args.fixed_noise is not None:
        args.fixed_noise = float(args.fixed_noise)
        args.flux_to_scale = float(args.flux_to_scale)
        args.sky_norm = float(args.sky_norm)
    if args.scale_exp_SNR is not None: args.scale_exp_SNR = float(args.scale_exp_SNR)
    if args.cmin is not None: args.cmin = float(args.cmin)
    if args.cmax is not None: args.cmax = float(args.cmax)
    if args.snr_cmin is not None: args.snr_cmin = float(args.snr_cmin)
    if args.snr_cmax is not None: args.snr_cmax = float(args.snr_cmax)
    if args.scale_length is not None: args.scale_length = float(args.scale_length)
    if args.debug: args.toscreen = True  # debug mode forces toscreen

    logbook.wlist, logbook.llist = po.readlist()
    logbook.wmin = float(args.wmin)
    logbook.wmax = float(args.wmax)
    logbook.exptime = float(args.exptime)
    logbook.final_pix_size = float(args.final_pix_size)
    logbook.llist = logbook.llist[np.where(np.logical_and(logbook.wlist > logbook.wmin,
                                                          logbook.wlist < logbook.wmax))]  # truncate linelist as per wavelength range
    logbook.wlist = logbook.wlist[np.where(np.logical_and(logbook.wlist > logbook.wmin, logbook.wlist < logbook.wmax))]
    logbook.fitsname = args.fitsname
    logbook.fitsname_u = args.fitsname_u
    logbook.skynoise_cubename = args.skynoise_cubename

    properties = po.get_disp_array(args, logbook, properties)
    properties.dist = float(args.dist)
    if args.addnoise and not args.noskynoise:
        properties.skynoise = fits.open(logbook.skynoise_cubename)[0].data
        print_master('Reading existing skynoise cube from ' + logbook.skynoise_cubename + '\n', args)
    else:
        properties.skynoise = None

    cube = fits.open(args.last_cubename)[0].data
    nslice = np.shape(cube)[2]
    ppv = np.zeros(np.shape(cube))
    ppv_u = np.zeros(np.shape(cube))

    if args.parallel:
        comm = MPI.COMM_WORLD
        ncores = comm.size
        rank = comm.rank
        print_master('Making ' + str(nslice) + ' slices to observables, in parallel...', args)
        print_master('Total number of MPI ranks = ' + str(ncores) + '. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(
            datetime.datetime.now()), args)
        core_start = rank * (nslice / ncores)
        core_end = (rank + 1) * (nslice / ncores)
        if (rank == ncores - 1): core_end = nslice  # last PE gets the rest
        if args.debug: print_mpi(
            'Operating on slice ' + str(core_start) + ' to ' + str(core_end) + ' out of ' + str(nslice) + ' cells',
            args)
        prefix = '[' + str(rank) + '] '

        comm.Barrier()
        t_start = MPI.Wtime()  ### Start stopwatch ###

        ppv_local, ppv_u_local = po.makeobservable(cube, args, logbook, properties, core_start, core_end,
                                                   prefix)  # performing operation on cube

        comm.Barrier()
        comm.Allreduce(ppv_local, ppv, op=MPI.SUM)
        comm.Allreduce(ppv_u_local, ppv_u, op=MPI.SUM)
        if rank == 0:
            ppv = np.ma.masked_where(ppv < 0., ppv)
            po.write_fits(logbook.fitsname, ppv, args)
            po.write_fits(logbook.fitsname_u, ppv_u, args)

        t_diff = MPI.Wtime() - t_start  ### Stop stopwatch ###
        print_master('Parallely: time taken for make-observable of ' + str(nslice) + ' slices with ' + str(
            ncores) + ' cores= ' + str(t_diff / 60.) + ' min', args)
    else:
        po.myprint('Making slices to observables, in series...', args)
        time_temp = time.time()
        ppv, ppv_u = po.makeobservable(cube, args, logbook, properties, 0, nslice,
                                       '')  # performing operation on cube
        po.write_fits(logbook.fitsname, ppv, args)
        po.write_fits(logbook.fitsname_u, ppv_u, args)
        po.myprint('Serially: time taken for make-observable of ' + str(nslice) + ' slices= ' + str(
            (time.time() - time_temp) / 60.) + ' min', args)
