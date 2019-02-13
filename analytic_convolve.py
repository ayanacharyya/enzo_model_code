#python code to analytically test the effect of convolution on gradients
#by Ayan, August 2018

from matplotlib import pyplot as plt
import numpy as np
import astropy.convolution as con
from astropy.modeling.models import Moffat1D, Gaussian1D
import subprocess
import sys
import os
from scipy.special import erf

# -------------From Dopita 2016------------------------------------------------------------------------------
def D16(n2s2, n2ha):
    log_ratio = np.log10(n2s2) + 0.264 * np.log10(n2ha)
    logZ = log_ratio + 0.45 * (log_ratio + 0.3) ** 5
    return logZ

# -------------From Kewley & Dopita 2002------------------------------------------------------------------------------
def KD02(n2o2):
    log_ratio = np.log10(n2o2)
    logZ = np.log10(1.54020 + 1.26602 * log_ratio + 0.167977 * log_ratio ** 2)
    return logZ

# -------------To determine sigma from FWHM, depending on smoothing profile------------------------------------------------------------------------------
def get_sigma(fwhm, kernel_profile, beta=4.7):
    if kernel_profile == 'Gauss': sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    elif kernel_profile == 'Moffat': sigma = fwhm/(2*np.sqrt(2**(1/beta) - 1))
    else: sys.exit()
    return sigma


# -------------To apply analytic formula for smoothing instead if actual convolution: to avoid boundary issues---------------------------
def analytic_gauss(data, sigma, power):
    return 0.5 * np.exp(-power*(data - 0.5*power*sigma**2)) * [1. + erf((data - power*sigma**2)/(np.sqrt(2)*sigma))]
    # from equation 10 of http://www.np.ph.bham.ac.uk/research_resources/programs/halflife/gauss_exp_conv.pdf

# -------------------------------------------------------------------------------------------
for_paper = 1 # different formatting for paper
saveplot = False
nlines = 4
if len(sys.argv) > 1: which_profile = sys.argv[1]
else: which_profile = 'DD0600'

if which_profile == 'sami' or which_profile == 'SAMI':
    #-----------from SAMI data galaxy ID 288461------------
    slope_arr = np.array([-0.940, -0.823, 0.0, -0.972])*np.log(10) # sequence of NII, SII, OII, Ha # OII is mock for galaxy #288461
    yscale_arr = np.exp((np.array([-16.49, -16.28, -16.0, -15.49]) + 0.6)*np.log(10)) # sequence of NII, SII, OII, Ha # 0.6 to change from ergs/s/cm^2/spaxel to ergs/s/cm^2/arcsec^2, given each pixel is 0.5 x 0.5 arcsec
    xlim1, xlim2, nx = 0, 4, 500 # kpc
    ylim11, ylim21 = -20, -15.5 # for fluxes plot
    ylim12, ylim22 = -2.5, 2.5 # for ratios plot
    single_fwhm = 1 # kpc
elif which_profile == 'typhoon' or which_profile == 'TYPHOON':
    # -----------from SAMI data galaxy ID N5236------------
    slope_arr = np.array([-1.71, -1.54, -1.53, -1.87]) / 4. * np.log(10)  # sequence of NII, SII, OII, Ha # factor 4 is scale_length, since gradients were in units of dex/R_e
    yscale_arr = np.exp((np.array([-13.37, -13.77, -13.19, -12.99]) - 0.435) * np.log(10))  # sequence of NII, SII, OII, Ha # -0.435 to change from ergs/s/cm^2/spaxel to ergs/s/cm^2/arcsec^2, given each pixel is 1.65 x 1.65 arcsec
    xlim1, xlim2, nx = 0, 12, 500  # kpc
    ylim11, ylim21 = -19, -13  # for fluxes plot
    ylim12, ylim22 = -2.5, 2.5  # for ratios plot
    single_fwhm = 2  # kpc
else:
    #--------from DD0600 for Zgrad=-0.1------------------
    slope_arr = np.array([-0.1054, -0.0408, -0.0177, 0.0086])*np.log(10) # sequence of NII, SII, OII, Ha
    yscale_arr = np.exp((np.array([-13.36, -13.74, -12.95, -12.99]) - 1.86)*np.log(10)) # sequence of NII, SII, OII, Ha # -1.86 is to change from ergs/s/A/pc^2 to ergs/s/cm^2/arcsec^2
    xlim1, xlim2, nx = 0, 15, 500 # kpc
    ylim11, ylim21 = -17.5, -14.5 # for fluxes plot
    ylim12, ylim22 = -2.5, 2.5 # for ratios plot
    single_fwhm = 4 # kpc

if for_paper: fwhm_arr = [single_fwhm] # kpc
else: fwhm_arr = np.linspace(0.1,5,60) #kpc
fs = 15 # fontsize of labels
beta = 4.7 # for Moffat profile
linestyle_initial = 'solid'
linestyle_postconv = 'dashed'
line_thick = 3

col_arr = ['r', 'b', 'orange', 'darkgreen', 'brown']
label_arr = ['NII', 'SII', 'OII', 'Ha']
if for_paper: outpath = '/Users/acharyya/Dropbox/papers/enzo_paper/Figs/'
else: outpath = '/Users/acharyya/Desktop/bpt_contsub_contu_rms/analytic_convolution/'
kernel_profile = 'Moffat' # 'Gauss' # gauss or moffat
use_analytic = 0 # whether to use analytic formula instead of convolving
use_cheat = 1 # whether to divide out convolved map by a np.ones() convolved map to take out numerical effects

xarr = np.linspace(xlim1, xlim2, nx)
subprocess.call(['mkdir -p ' + outpath], shell=True)  # create output directory if it doesn't exist
use_analytic_flag = '_formula' if use_analytic else ''
use_cheat_flag = '_cheat' if use_cheat else ''
if not for_paper: plt.close('all')

for (index,fwhm) in enumerate(np.array(fwhm_arr, dtype=float)):
    fig = plt.figure(figsize=(14, 6) if for_paper else (12, 6))
    fig.subplots_adjust(hspace=0.8, top=0.98 if for_paper else 0.85, bottom=0.12 if for_paper else 0.15, left=0.07, right=0.98)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    # -----declaring arrays and computing kernel--------
    line_arr = np.zeros((nlines, nx))
    conv_arr = np.zeros((nlines, nx))
    dx = float(xlim2 - xlim1)/nx # kpc
    sigma = get_sigma(fwhm, kernel_profile) # both fwhm and sigma are in kpc
    sigma_npixel = sigma/dx # sigma_npixel is in pixel units

    xsize = int(np.round(8 * sigma_npixel))
    if xsize % 2 == 0: xsize += 1  # to make it odd integer
    if kernel_profile == 'Gauss':
        mykernel = Gaussian1D(0.5, 0, sigma_npixel) # amplitude is unimportant as kernel is normalised later
        plot_kernel = Gaussian1D(0.5, (xlim1 + xlim2)/2., sigma) # only for plotting purposes
    elif kernel_profile == 'Moffat':
        mykernel = Moffat1D(0.5, 0, sigma_npixel, beta) # amplitude is unimportant as kernel is normalised later
        plot_kernel = Moffat1D(0.5, (xlim1 + xlim2)/2., sigma, beta) # only for plotting purposes
    kernel = con.Model1DKernel(mykernel, x_size=xsize)

    #kernel = con.Gaussian1DKernel(stddev=sigma_npixel) # this is an alternative for manually defining Gaussian1D kernel with con.Model1DKernel
    print str(index+1) + ' out of '+str(len(fwhm_arr))+': For FWHM = %.2F kpc, smoothing %d pixel array with a sigma=%.2F \
pixels %s kernel..'%(fwhm, nx, sigma_npixel, kernel_profile + use_analytic_flag + use_cheat_flag)
    # -----plotting the kernel--------
    ax1.plot(xarr, ylim11 + 0.1 + plot_kernel(xarr), color='k', ls = linestyle_initial, lw=line_thick, label=\
        'PSF = %.2F kpc'%fwhm if for_paper else 'PSF (' + kernel_profile[0] + use_analytic_flag[:2] + use_cheat_flag[:2]+')')

    # -----loop over all emission lines--------
    for i in range(nlines):
        if for_paper and label_arr[i] == 'OII': continue
        line_arr[i] = yscale_arr[i] * np.exp(slope_arr[i]*xarr)
        if use_analytic and kernel_profile == 'Gauss':
            conv_arr[i] =  yscale_arr[i] * analytic_gauss(xarr, sigma, -slope_arr[i])
        else:
            conv_arr[i] = con.convolve_fft(line_arr[i], kernel, normalize_kernel=True)
            if use_cheat: conv_arr[i] /= con.convolve_fft(np.ones(np.shape(line_arr[i])), kernel, normalize_kernel = True) # to divide out convolved map by a np.ones() convolved map to take out numerical effects, esp at the edges
        # -----plotting individual lines--------
        ax1.plot(xarr, np.log10(line_arr[i]), color=col_arr[i], label=label_arr[i], ls=linestyle_initial, lw=line_thick)
        ax1.plot(xarr, np.log10(conv_arr[i]), color=col_arr[i], ls=linestyle_postconv, lw=line_thick)
    # -----calculating ratios and Z--------
    ratio1 = line_arr[0]/line_arr[1] # NII/SII
    if not for_paper: ratio2 = line_arr[0]/line_arr[2] # NII/OII
    ratio3 = line_arr[0]/line_arr[3] # NII/Ha
    logZ_D16 = D16(ratio1, ratio3)
    if not for_paper: logZ_KD02 = KD02(ratio2)
    # -----plotting individual ratios and Z--------
    ax2.plot(xarr, ratio1, color=col_arr[0], label=label_arr[0]+'/'+label_arr[1], ls=linestyle_initial, lw=line_thick)
    if not for_paper: ax2.plot(xarr, ratio2, color=col_arr[1], label=label_arr[0]+'/'+label_arr[2], ls=linestyle_initial, lw=line_thick)
    ax2.plot(xarr, ratio3, color=col_arr[2], label=label_arr[0]+'/'+label_arr[3], ls=linestyle_initial, lw=line_thick)
    ax2.plot(xarr, logZ_D16, color=col_arr[3], label='Log Z (D16)', ls=linestyle_initial, lw=line_thick)
    if not for_paper: ax2.plot(xarr, logZ_KD02, color=col_arr[4], label='Log Z (KD02)', ls=linestyle_initial, lw=line_thick)
    # -----calculating post-convolution ratios and Z--------
    conv_ratio1 = conv_arr[0]/conv_arr[1] # NII/SII
    if not for_paper: conv_ratio2 = conv_arr[0]/conv_arr[2] # NII/OII
    conv_ratio3 = conv_arr[0]/conv_arr[3] # NII/Ha
    conv_logZ_D16 = D16(conv_ratio1, conv_ratio3)
    if not for_paper: conv_logZ_KD02 = KD02(conv_ratio2)
    # -----plotting post-convolution ratios and Z--------
    ax2.plot(xarr, conv_ratio1, color=col_arr[0], ls=linestyle_postconv, lw=line_thick)
    if not for_paper: ax2.plot(xarr, conv_ratio2, color=col_arr[1], ls=linestyle_postconv, lw=line_thick)
    ax2.plot(xarr, conv_ratio3, color=col_arr[2], ls=linestyle_postconv, lw=line_thick)
    ax2.plot(xarr, conv_logZ_D16, color=col_arr[3], ls=linestyle_postconv, lw=line_thick)
    if not for_paper: ax2.plot(xarr, conv_logZ_KD02, color=col_arr[4], ls=linestyle_postconv, lw=line_thick)
    # -----plot labels of ax1--------
    ax1.set_xlim(xlim1-1, xlim2+1)
    ax1.set_ylabel(r'log(Flux in ergs/s/cm$^2$/arcsec$^2$)', fontsize=fs)
    ax1.set_ylim(ylim11, ylim21)
    ax1.legend(loc='lower left', fontsize=fs)
    # -----plot labels of ax2--------
    ax2.set_xlim(xlim1-1, xlim2+1)
    ax2.set_ylabel(r'Line ratio / log(Z/Z$_{\bigodot}$)', fontsize=fs)
    ax2.set_ylim(ylim12, ylim22)
    ax2.legend(loc='lower left', fontsize=fs)
    # -----decorate and save plots--------
    ax1.set_yticks(ax1.get_yticks()[1:-1])
    ax1.set_yticklabels(ax1.get_yticks(), fontsize=fs)
    ax1.set_xticklabels(['%.0F'%item for item in ax1.get_xticks()[:-1]], fontsize=fs)
    ax2.set_yticklabels(ax2.get_yticks(), fontsize=fs)
    ax2.set_xticklabels(['%.0F'%item for item in ax2.get_xticks()[:-1]], fontsize=fs)
    fig.text(0.5, 0.02, 'Radius (kpc)', color='k', va='bottom', ha='center', fontsize=fs)
    if not for_paper:
        fig.text(0.5, 0.98, 'PSF size = %.2F kpc'%fwhm, color='k', va='top', ha='center', fontsize=fs)
        fig.text(0.5, 0.92, '(Dashed lines denote after convolution with PSF)', color='k', va='top', ha='center', fontsize=fs)
    filename_root = 'analytic_convolve_'+kernel_profile + use_analytic_flag + use_cheat_flag
    if for_paper: filename = filename_root + '_fwhm%.2Fkpc.eps'%fwhm
    else: filename = filename_root + str(index+1)+'.png'
    if saveplot: fig.savefig(outpath+ filename)
    if len(fwhm_arr) > 10: plt.close()
    else: plt.show(block=False)

if len(fwhm_arr) > 10:
    movie_name = outpath + filename_root.replace('analytic_convolve_', 'convolution_anim_') + '_'+str(len(fwhm_arr))+'frames.mp4'
    if os.path.exists(movie_name): subprocess.call(['rm '+movie_name], shell=True)
    try:
        subprocess.check_output(['ffmpeg -f image2 -pattern_type sequence -start_number 1 -framerate 10 -i '+outpath+\
                                 filename_root + '%d.png -vcodec mpeg4 -loop 1 '+ movie_name], shell=True)
        print 'Created movie', movie_name
    except subprocess.CalledProcessError as e:
        print e.output
