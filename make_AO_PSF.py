#python code to generate analytic PSF of AO assisted observation on Keck/OSIRIS, following K glazebrook's prescription
#by Ayan, October 2018
'''
Excerpt from KG email
Recipe:
e.g. Strehl = 0.3 in K band
- Put 30% of the light in a diffraction limited Airy disk
- Put the other 70% in seeing limited profile (Moffat beta=-2.5 ought to be reasonable with say a 0.5" FWHM)
- Add the two (same centre)
- Take the entire thing and blur it with a Gaussian of 0.1" to simulate 'tip tilt' error
That would be a rasonably fair generic representation, the main degree of freedom would be the Strehl.
'''

from matplotlib import pyplot as plt
import numpy as np
import astropy.convolution as con
import sys
import os
HOME = os.getenv('HOME') + '/'
from astropy.modeling import models, fitting

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

# -----values to edit------------
diameter = 10. # telescope aperture in metres, for Airy kernel
arcsec_per_pixel = 0.02 # arcseconds per pixel, for Moffat and Gaussian kernels
size_arcsec = 1. # truncation size of all kernels in arcseconds
mode = 'integral' # if the kernels are normalised to peak intensity = 1 ('peak') or integrated area = 1 ('integral')
strehl_arr =[0.3, 0.1] # [0.1, 0.3, 0.4, 1.]
fraction_to_plot = 1. # fraction of kernel to plot, before it becomes practically flat
band_arr = ['K', 'J'] # 'J' or 'K' band
plot2d = 0
plot_breakdown = 1
savefig = 0
outpath = HOME + 'Dropbox/papers/kinematics/AA_Working/'

# ------------------------------
size = int(size_arcsec / arcsec_per_pixel) # size of 2D array for all kernelsin pixel units
if size % 2 == 0: size += 1 # kernels need odd int size
if plot_breakdown:
    strehl_arr = [0.1] #[0.3]
    band_arr = ['optical'] #['K']
    outfile = 'PSF_construction.eps'
else:
    outfile = 'PSF_Strehl.eps'
#plt.close('all')
figure, ax = plt.subplots(figsize=(8,4)) # figure for 1D plots
figure.subplots_adjust(top=0.95, right=0.98, left=0.1, bottom=0.15)

# --------begin loop over different Strehl values----------------------
for (index, strehl) in enumerate(strehl_arr):
    # -----computing Airy Disk--------------
    band = band_arr[index]
    if band == 'K': wave = 2.2e-6 # K band wavelength in metres
    elif band == 'J': wave = 1.25e-6 # K band wavelength in metres
    elif band == 'optical': wave = 6.5e-7 # NII/Ha narrow-band wavelength in metres
    radius = 1.22 * wave / diameter # radius of first dark band, in radian
    radius_arcsec = radius * 180. * 3600. /np.pi # radius in arcseconds
    radius_pix = radius_arcsec / arcsec_per_pixel # radius in pixel units
    airy_kernel = con.AiryDisk2DKernel(radius_pix, x_size=size, y_size=size) # already normalised
    airy_kernel.normalize(mode=mode) # normalise such that peak = 1
    if plot2d: fig = plot_kernel(airy_kernel, 'Airy (%.2F")'%radius_arcsec, radius=radius_pix)

    # -----computing Moffat profile---------
    fwhm_arcsec =.5 # seeing in arcseconds
    fwhm_pix = fwhm_arcsec / arcsec_per_pixel # seeing in pixel units
    beta = 2.5 # power law index
    width = fwhm_pix/(2 * np.sqrt(np.power(2, 1./beta) - 1)) # core width in pixels
    moffat_kernel = con.Moffat2DKernel(gamma=width, alpha=beta, x_size=size, y_size=size)
    moffat_kernel.normalize(mode=mode) # normalise such that peak = 1
    if plot2d: fig = plot_kernel(moffat_kernel, 'Moffat (%.2F")'%fwhm_arcsec, radius=fwhm_pix)

    # --------computing Gaussian profile----------
    gaussian_blur_arcsec = 0.1 # Gaussian blur in arcseconds (FWHM)
    gaussian_blur_pix = (gaussian_blur_arcsec/2.355) / arcsec_per_pixel # Gaussian blur in pixel units (std dev, hence factor of 2.355)
    gaussian_kernel = con.Gaussian2DKernel(gaussian_blur_pix, x_size=size, y_size=size)
    gaussian_kernel.normalize(mode=mode) # normalise such that peak = 1, just for plotting
    if plot2d: fig = plot_kernel(gaussian_kernel, 'Gaussian (%.2F")'%gaussian_blur_arcsec, radius=gaussian_blur_pix)

    # ------adding kernels----------
    added_kernel = strehl * airy_kernel + (1 - strehl) * moffat_kernel # both kernels had peak intensity = 1, which will now = 0.3, and 0.7, giving Strehl ratio=0.3
    if plot2d: fig = plot_kernel(added_kernel, 'Added')

    # ------Gaussian smoothing for tip-tilt uncertainties----------
    final_kernel = con.convolve(added_kernel, gaussian_kernel, normalize_kernel=True)
    if not plot_breakdown: final_kernel.normalize(mode='integral')
    if plot2d: fig = plot_kernel(final_kernel, 'Convolved')

    # ------plotting 1D cross-sections of the 2D kernels----------
    portion = int(fraction_to_plot * (size/2)) + 1
    if plot_breakdown:
        ax.plot(np.arange(portion), airy_kernel.array[size/2][size/2 : size/2 + portion], c='r', lw=1, label='Airy (%.2F")'%radius_arcsec)
        ax.plot(np.arange(portion), moffat_kernel.array[size/2][size/2 : size/2 + portion], c='brown', lw=1, label='Moffat (%.2F")'%fwhm_arcsec)
        ax.plot(np.arange(portion), added_kernel.array[size/2][size/2 : size/2 + portion], c='gray', lw=1, label='Airy + Moffat', linestyle='--')
        ax.plot(np.arange(portion), gaussian_kernel.array[size/2][size/2 : size/2 + portion], c='g', lw=1, label='Gaussian (%.2F")'%gaussian_blur_arcsec)
        fwhm_guess = 0.1#fwhm_arcsec / 4.
        gamma_guess = (fwhm_guess/arcsec_per_pixel)/(2 * np.sqrt(np.power(2, 1./beta) - 1))
        fitted_moffat = fitting.LevMarLSQFitter()(models.Moffat1D(x_0=0, alpha=beta, gamma=gamma_guess, \
                                                                  fixed={'alpha':True, 'x_0':True, 'gamma':False, 'amplitude':False}), \
                                                  np.arange(portion), final_kernel.array[size / 2][size / 2: size / 2 + portion])
        print fitted_moffat #
        fitted_moffat_fwhm_pix = fitted_moffat.gamma * (2 * np.sqrt(np.power(2, 1./fitted_moffat.alpha) - 1))
        fitted_moffat_fwhm_arcsec = fitted_moffat_fwhm_pix * arcsec_per_pixel
        ax.plot(np.arange(portion), fitted_moffat(np.arange(portion)), c='goldenrod', lw=1, label='Fitted Moffat (%.2F")'%fitted_moffat_fwhm_arcsec)
        rms = np.sqrt(np.sum((fitted_moffat(np.arange(portion)) - final_kernel.array[size / 2][size / 2: size / 2 + portion])**2))
        print 'RMS = %.3E'%rms #
        label = 'Convolved: Strehl %.1F'%strehl
    else:
        label = '%s band: Strehl %.1F'%(band, strehl)
    ax.plot(np.arange(portion), final_kernel.array[size/2][size/2 : size/2 + portion], lw=2, label=label)

ax.legend()
ax.set_xlim(0, 0.2 / arcsec_per_pixel)
ax.set_xticklabels(['%.2F' % (float(item) * arcsec_per_pixel) for item in ax.get_xticks()])
ax.set_xlabel('Arcseconds')
is_normalised = '' if plot_breakdown else 'Normalised '
ax.set_ylabel(is_normalised + 'Intensity')
ax.set_ylim(0,0.005) #
if savefig:
    figure.savefig(outpath + outfile)
    print 'Figure saved as', outpath + outfile
plt.show(block=False)

print 'Finished!'