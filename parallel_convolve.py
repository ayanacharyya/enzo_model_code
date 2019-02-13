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

parser = ap.ArgumentParser(description="parallel convolution tool")
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

# ---------function for plotting individual kernels------------
def plot_kernel(kernel, kernel_name, fig=None, savefig=False, outfile=None, radius=None):
    if fig is None: fig = plt.figure()
    plt.imshow(kernel, interpolation='none', origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    cb.set_label('Intensity')
    plt.title(kernel_name)
    if False: #radius is not None:
        plt.xlim(-8 * radius, 8 * radius) # radius is in pixel units
        plt.ylim(-8 * radius, 8 * radius)  # radius is in pixel units
    if savefig:
        fig.savefig(outfile)
        print kernel_name, 'plot saved in', outfile
    plt.show(block=False)
    return fig

# -----------function to construct AO PSF----------------------
def get_AOPSF(args):
    # diameter =  telescope aperture in metres, for Airy kernel
    # arcsec_per_pixel =  arcseconds per pixel, for Moffat and Gaussian kernels
    # size = truncation size of all kernels in pixel
    mode = 'integral'  # if the kernels are normalised to peak intensity = 1 ('peak') or integrated area = 1 ('integral')
    arcsec_per_pixel = args.arcsec_per_pixel
    size = args.size
    diameter = 2 * args.rad
    # -----computing Airy Disk--------------
    wave = 6.5e-7 # Ha narrow-band wavelength in metres
    radius = 1.22 * wave / diameter # radius of first dark band, in radian
    radius_arcsec = radius * 180. * 3600. /np.pi # radius in arcseconds
    radius_pix = radius_arcsec / arcsec_per_pixel # radius in pixel units
    airy_kernel = con.AiryDisk2DKernel(radius_pix, x_size=size, y_size=size) # already normalised
    airy_kernel.normalize(mode=mode) # normalise such that peak = 1

    # -----computing Moffat profile---------
    moffat_kernel = con.Moffat2DKernel(gamma=args.sig, alpha=args.pow, x_size=size, y_size=size)
    moffat_kernel.normalize(mode=mode) # normalise such that peak = 1
    fwhm_arcsec = args.sig * (2 * np.sqrt(2 ** (1. / args.pow) - 1.)) * arcsec_per_pixel

    # --------computing Gaussian profile----------
    gaussian_blur_arcsec = 0.1 # Gaussian blur in arcseconds (FWHM)
    gaussian_blur_pix = (gaussian_blur_arcsec/2.355) / arcsec_per_pixel # Gaussian blur in pixel units (std dev, hence factor of 2.355)
    gaussian_kernel = con.Gaussian2DKernel(gaussian_blur_pix, x_size=size, y_size=size)
    gaussian_kernel.normalize(mode=mode) # normalise such that peak = 1, just for plotting

    # ------adding kernels----------
    added_kernel = args.strehl * airy_kernel + (1 - args.strehl) * moffat_kernel # both kernels had peak intensity = 1, which will now = 0.3, and 0.7, giving Strehl ratio=0.3

    # ------Gaussian smoothing for tip-tilt uncertainties----------
    final_kernel = con.convolve(added_kernel, gaussian_kernel, normalize_kernel=True)

    # -----computing fiducial Moffat profile---------
    fid_pow, fid_size = 4.7, 5
    sigma = args.sig * (2 * np.sqrt(2 ** (1. / args.pow) - 1.)) / (2 * np.sqrt(2 ** (1. / fid_pow) - 1.))
    fid_size = int((sigma * fid_size) // 2 * 2 + 1)
    fiducial_moffat_kernel = con.Moffat2DKernel(gamma=sigma, alpha=fid_pow, x_size=fid_size, y_size=fid_size)
    fiducial_moffat_kernel.normalize(mode=mode) # normalise such that peak = 1
    fiducial_fwhm_arcsec = args.sig * (2 * np.sqrt(2 ** (1. / args.pow) - 1.)) * arcsec_per_pixel
    #print_master('Areas final=%0.4F, gaussian=%0.4F, added=%0.4F, airy=%0.4F, moffat=%0.4F, fiducial_moffat=%0.4F'%(np.sum(final_kernel), np.sum(gaussian_kernel), np.sum(added_kernel), np.sum(airy_kernel), np.sum(moffat_kernel), np.sum(fiducial_moffat_kernel)), args)

    if args.plot_1dkernel and MPI.COMM_WORLD.rank == 0:
        # ------plotting 1D cross-sections of the 2D kernels----------
        fraction_to_plot = 0.7 # fraction of kernel to plot, before it becomes practically flat
        portion = int(fraction_to_plot * (size / 2)) + 1
        fig, ax = plt.subplots(figsize=(8, 4))  # figure for 1D plots
        fig.subplots_adjust(top=0.95, right=0.98, left=0.1, bottom=0.15)
        ax.plot(np.arange(portion), airy_kernel.array[size/2][size/2 : size/2 + portion], c='r', lw=1, label='Airy (%.2F")'%radius_arcsec)
        ax.plot(np.arange(portion), moffat_kernel.array[size/2][size/2 : size/2 + portion], c='brown', lw=1, label=r'Moffat (%.2F", $\alpha$=%.1F)'%(fwhm_arcsec,args.pow))
        ax.plot(np.arange(portion), added_kernel.array[size/2][size/2 : size/2 + portion], c='gray', lw=1, label='Airy + Moffat', linestyle='--')
        ax.plot(np.arange(portion), gaussian_kernel.array[size/2][size/2 : size/2 + portion], c='g', lw=1, label='Gaussian (%.2F")'%gaussian_blur_arcsec)
        ax.plot(np.arange(portion), final_kernel.array[size / 2][size / 2: size / 2 + portion], c='b', label='Convolved: Strehl %.1F'%args.strehl)
        ax.plot(np.arange(portion), fiducial_moffat_kernel.array[size / 2][size / 2: size / 2 + portion], c='goldenrod', label=r'Fiducial Moffat (%.2F", $\alpha$=%.1F))'%(fiducial_fwhm_arcsec, fid_pow))
        ax.legend()
        ax.set_xlim(0, portion - 1)
        ax.set_xticklabels(['%.2F' % (float(item) * arcsec_per_pixel) for item in ax.get_xticks()])
        ax.set_xlabel('Arcseconds')
        ax.set_ylabel('Intensity')
        ax.set_ylim(0,2 * np.max(fiducial_moffat_kernel.array[size / 2][size / 2: size / 2 + portion]))
        if args.saveplot:
            outfile = os.path.dirname(args.H2R_cubename)+'/PSF_comparison_res=%.1F"_Strehl=%.1F.eps'%(fwhm_arcsec, args.strehl)
            fig.savefig(outfile)
            print_master('PSF figure saved as ' + outfile, args)
        plt.show(block=True)

    return final_kernel

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
    parser.add_argument('--plot_1dkernel', dest='plot_1dkernel', action='store_true')
    parser.set_defaults(plot_1dkernel=False)

    parser.add_argument('--file')
    parser.add_argument('--sig')
    parser.add_argument('--pow')
    parser.add_argument('--size')
    parser.add_argument('--ker')
    parser.add_argument('--convolved_filename')
    parser.add_argument('--outfile')
    parser.add_argument('--H2R_cubename')
    parser.add_argument('--fitsname')
    parser.add_argument('--exptime')
    parser.add_argument('--final_pix_size')
    parser.add_argument('--galsize')
    parser.add_argument('--center')
    parser.add_argument('--intermediate_pix_size')
    parser.add_argument('--cmin')
    parser.add_argument('--cmax')
    parser.add_argument('--xcenter_offset')
    parser.add_argument('--ycenter_offset')
    parser.add_argument('--arcsec_per_pixel')
    parser.add_argument('--rad')
    parser.add_argument('--strehl')
    args, leftovers = parser.parse_known_args()
    args.sig = float(args.sig)
    args.pow = float(args.pow)
    args.size = int(args.size)
    args.galsize = float(args.galsize)
    args.center = float(args.center)
    args.xcenter_offset = float(args.xcenter_offset)
    args.ycenter_offset = float(args.ycenter_offset)
    args.arcsec_per_pixel = float(args.arcsec_per_pixel)
    args.rad = float(args.rad)
    args.strehl = float(args.strehl)
    if args.cmin is not None: args.cmin = float(args.cmin)
    if args.cmax is not None: args.cmax = float(args.cmax)
    if args.debug: args.toscreen = True  # debug mode forces toscreen

    logbook.exptime = float(args.exptime)
    logbook.final_pix_size = float(args.final_pix_size)
    logbook.fitsname = args.fitsname
    logbook.intermediate_pix_size = float(args.intermediate_pix_size)
    rebinned_shape = (int(args.galsize / logbook.intermediate_pix_size), int(args.galsize / logbook.intermediate_pix_size))

    cube = fits.open(args.H2R_cubename)[0].data
    nslice = np.shape(cube)[2]
    if args.ker == 'gauss':
        kernel = con.Gaussian2DKernel(args.sig, x_size=args.size, y_size=args.size)
    elif args.ker == 'moff':
        kernel = con.Moffat2DKernel(args.sig, args.pow, x_size=args.size, y_size=args.size)
    elif args.ker == 'AOPSF':
        kernel = get_AOPSF(args)
        #sys.exit() #

    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = ' + str(ncores) + '. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)
    convolved_cube_local = np.zeros((int(args.galsize / logbook.final_pix_size), int(args.galsize / logbook.final_pix_size),
                               np.shape(cube)[2]))  ###only if re-sampling convolved cube back to original resolution
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
        if not args.silent: print_mpi('Rebinning before convolution: slice ' + str(k) + ' of ' + str(core_end), args)
        if args.debug and k % slice_slab == 0:
            dummy = po.plotmap(cube[:, :, k], 'slice ' + str(k) + ': before rebinning before convolution', 'junk', 'ergs/s/A/pixel', args, logbook, makelog=True)
            print_mpi('Integrated flux in slice ' + str(k) + ' before rebinning = %.4E ergs/s/A' % np.sum(cube[:, :, k]), args)
        rebinned_slice = po.rebin(cube[:, :, k], rebinned_shape)  # re-bin 2d array before convolving to make things faster

        if args.debug and k % slice_slab == 0:
            dummy = po.plotmap(rebinned_slice, 'slice ' + str(k) + ': before convolution', 'junk', 'ergs/s/A/pixel', args, logbook, makelog=True)
            po.mydiag('[' + str(comm.rank) + '] ' + 'Deb 77: in ergs/s/A/pixel: before convoluion', dummy, args)
            print_mpi('Integrated flux in slice '+str(k)+' just before convolution = %.4E ergs/s/A'%np.sum(rebinned_slice), args)

        result = con.convolve_fft(rebinned_slice, kernel, normalize_kernel=True)
        median = np.median(np.ma.compressed(np.ma.masked_where(cube[:, :, k] <= 0, cube[:, :, k])))
        result[np.log10(np.abs(median)) - np.log10(np.abs(
            result)) > 10.] = 0.  # discarding any resulting pixel that is more than O(10) lower than input data, to avoid round off error pixels
        result = po.rebin(result, (np.shape(convolved_cube_local)[0], np.shape(convolved_cube_local)[
            1]))  ###this line actually matters only if re-sampling convolved cube back to original resolution
        convolved_cube_local[:, :, k] = result

        if args.debug and k % slice_slab == 0:
            dummy = po.plotmap(convolved_cube_local[:, :, k], 'slice ' + str(k) + ': after convolution', 'junk', 'ergs/s/A/pixel', args, logbook, makelog=True)
            po.mydiag('[' + str(comm.rank) + '] ' + 'Deb 77: in ergs/s/A/pixel: after convoluion', dummy, args)
            print_mpi('Integrated flux in slice '+str(k)+' after convolution (and resampling) = %.4E ergs/s/A'%np.sum(result), args)
        if not args.silent: print_mpi('Convolved slice ' + str(k) + ' of ' + str(core_end) + ' slices at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)

        if args.debug and slice is not None and k % slice_slab == 0:
            print_mpi('Pausing for 10...', args)
            plt.pause(10)

    comm.Barrier()
    comm.Allreduce(MPI.IN_PLACE, convolved_cube_local, op=MPI.SUM)
    if rank == 0:
        convolved_cube_local = np.ma.masked_where(convolved_cube_local < 0., convolved_cube_local)
        po.write_fits(args.convolved_filename, convolved_cube_local, args)

    t_diff = MPI.Wtime() - t_start  ### Stop stopwatch ###
    print_master(
        'Parallely: time taken for convolution of ' + str(nslice) + ' slices with ' + str(ncores) + ' cores= ' + str(
            t_diff / 60.) + ' min', args)
    # -------------------------------------------------------------------------------------------
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
