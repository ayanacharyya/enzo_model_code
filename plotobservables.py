import time

start_time = time.time()
import datetime
import numpy as np
import subprocess
from matplotlib import pyplot as plt
from astropy.io import ascii, fits
import os

HOME = os.getenv('HOME')
WD = '/Work/astro/ayan_codes/enzo_model_code/'
import sys

sys.path.append(HOME + '/Work/astro/ayan_codes/mageproject/ayan/')
sys.path.append(HOME + WD)
import splot_util as su
from scipy.optimize import curve_fit, fminbound
from scipy.integrate import quad
from operator import itemgetter
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")
import multiprocessing as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse as ap

parser = ap.ArgumentParser(description="observables generating tool")
import astropy.convolution as con
import copy
from astropy.stats import gaussian_fwhm_to_sigma as gf2s
import re
from scipy.special import erf
import pandas as pd
import scipy.interpolate as si
from scipy import asarray as ar, exp
from copy import deepcopy
import scipy.ndimage as ndimage
import string


# ------------Reading pre-defined line list-----------------------------
def readlist():
    target = []
    finp = open(HOME + '/Mappings/lab/targetlines.txt', 'r')
    l = finp.readlines()
    for lin in l:
        if len(lin.split()) > 1 and lin.split()[0][0] != '#':
            target.append([float(lin.split()[0]), lin.split()[1]])
    finp.close()
    target = sorted(target, key=itemgetter(0))
    target = np.array(target)
    llist = np.array(target)[:, 1]
    wlist = np.asarray(target[:, 0], float)
    return wlist, llist


# -------------------------------------------------------------------------------------------
def Ke13b(x, z):
    return 0.61 / (x - 0.02 - 0.1833 * z) + 1.2 + 0.03 * z  # Kewley2013b


# -------------------------------------------------------------------------------------------
def Ke13a(x):
    return 0.61 / (x + 0.08) + 1.1  # Kewley2013a


# -------------------------------------------------------------------------------------------
def Ke01(x):
    return 0.61 / (x - 0.47) + 1.19  # Kewley2001


# -------------------------------------------------------------------------------------------
def Ka03(x):
    return 0.61 / (x - 0.05) + 1.3  # Kauffmann2003


# -------------------------------------------------------------------------------------------
def title(fn):
    if fn == 'DD0600_lgf':
        return 'Low gas fraction (0.1): after 600Myr\n'
    elif fn == 'DD0600':
        return 'Medium gas fraction (0.2): after 600Myr\n'
    elif fn == 'DD0300_hgf':
        return 'High gas fraction (0.4): after 300Myr\n'
    else:
        return ''


# --------------------to replace the group of digits after 'word's in a 'string' with 'repl'----------
def repl_wildcard(string, words, repl='*'):
    if not isinstance(words, list): words = [words]
    for word in words:
        string = string.split(word)[0] + word + re.sub(r'\d+', repl, string.split(word)[1], 1)
    return string


# -------------------------------------------------------------------------------------------
def plottheory():
    x = np.linspace(-1.3, -0.4, 1000)
    # plt.plot(x, Ke13a(x), linestyle='solid', label='Kewley 2013a')
    plt.plot(x, Ka03(x), linestyle='dashed', label='Kauffmann 2003')
    plt.plot(x, Ke01(x), linestyle='dotted', label='Kewley 2001')
    plt.legend(bbox_to_anchor=(0.45, 0.23), bbox_transform=plt.gcf().transFigure)


# -------------------------------------------------------------------------------------------------
def get_erf(lambda_array, height, centre, width, delta_lambda):
    return np.sqrt(np.pi / 2) * height * width * (
            erf((centre + delta_lambda / 2 - lambda_array) / (np.sqrt(2) * width)) - \
            erf((centre - delta_lambda / 2 - lambda_array) / (np.sqrt(
                2) * width))) / delta_lambda  # https://www.wolframalpha.com/input/?i=integrate+a*exp(-(x-b)%5E2%2F(2*c%5E2))*dx+from+(w-d%2F2)+to+(w%2Bd%2F2)


# -------------------------------------------------------------------------------------------------
def gauss(w, f, w0, f0, v, vz):
    w0 = w0 * (
            1 + vz / c)  # shift central wavelength wrt w0 due to LOS vel v_z of HII region as compared to systemic velocity
    sigma = w0 * v / c  # c=3e5 km/s
    A = f0 / np.sqrt(2 * np.pi * sigma ** 2)  # height of Gaussian, such that area = f0
    dw = w[np.where(w >= w0)[0][0]] - w[np.where(w >= w0)[0][0] - 1]
    g = get_erf(w, A, w0, sigma, dw)
    # g = A*np.exp(-((w-w0)**2)/(2*sigma**2))
    if args.debug and args.oneHII is not None: print 'Debugging76: input gaussian parm (ergs/s/A/pc^2) =', f[0] / (
            args.res * 1e3) ** 2, \
        (f0 / np.sqrt(2 * np.pi * sigma ** 2)) / (args.res * 1e3) ** 2, w0, sigma  #
    f += g
    return f


# -------------------------------------------------------------------------------------------
def fixcont_gaus(x, cont, n, *p):
    result = cont
    for xx in range(0, n):
        result += p[3 * xx + 0] * exp(-((x - p[3 * xx + 1]) ** 2) / (2 * p[3 * xx + 2] ** 2))
    return result


# -------------------------------------------------------------------------------------------
def fixcont_erf(x, cont, n, *p):
    result = cont
    for xx in range(0, n):
        dw = x[np.where(x >= p[3 * xx + 1])[0][0]] - x[np.where(x >= p[3 * xx + 1])[0][0] - 1]
        result += get_erf(x, p[3 * xx + 0], p[3 * xx + 1], p[3 * xx + 2], dw)
    return result


# -------------------------------------------------------------------------------------------
def bpt_pixelwise(args, logbook, properties):
    global info
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.1, right=0.95)
    ax = plt.subplot(111)
    if args.theory: plottheory()
    plt.ylabel('Log(O[III] 5007/Hb)')
    plt.xlabel('Log(N[II]6584/Ha)')
    if args.gridoverlay:
        plt.xlim(-3.5, 0.5)
        plt.ylim(-2.5, 1.5)
        gridoverlay(annotate=args.annotategrid)
    else:
        plt.xlim(-1.4, -0.2)
        plt.ylim(-1.5, 1.5)
    mapn2 = properties.mapcube[:, :, np.where(logbook.llist == 'NII6584')[0][0]]
    mapo3 = properties.mapcube[:, :, np.where(logbook.llist == 'OIII5007')[0][0]]
    mapha = properties.mapcube[:, :, np.where(logbook.llist == 'H6562')[0][0]]
    maphb = properties.mapcube[:, :, np.where(logbook.llist == 'Hbeta')[0][0]]

    if args.SNR_thresh is not None:
        # loading all the flux uncertainties
        mapn2_u = properties.errorcube[:, :, np.where(args.llist == 'NII6584')[0][0]]
        mapo3_u = properties.errorcube[:, :, np.where(args.llist == 'OIII5007')[0][0]]
        mapha_u = properties.errorcube[:, :, np.where(args.llist == 'H6562')[0][0]]
        maphb_u = properties.errorcube[:, :, np.where(args.llist == 'Hbeta')[0][0]]
        mapn2_u = np.ma.masked_where(mapn2_u <= 0., mapn2_u)
        mapo3_u = np.ma.masked_where(maps2a_u <= 0., mapo3_u)
        mapha_u = np.ma.masked_where(maps2b_u <= 0., mapha_u)
        maphb_u = np.ma.masked_where(mapha_u <= 0., maphb_u)
        # imposing SNR cut
        mapn2 = np.ma.masked_where(mapn2 / mapn2_u < SNR_thresh, mapn2)
        mapo3 = np.ma.masked_where(mapo3 / mapo3_u < SNR_thresh, mapo3)
        mapha = np.ma.masked_where(mapha / mapha_u < SNR_thresh, mapha)
        maphb = np.ma.masked_where(maphb / maphb_u < SNR_thresh, maphb)

    mapn2ha = np.divide(mapn2, mapha)
    mapo3hb = np.divide(mapo3, maphb)
    t = title(fn) + ' Galactrocentric-distance color-coded, BPT of model \n\
for Omega = ' + str(args.Om) + ', resolution = ' + str(args.res) + ' kpc' + info
    plt.scatter((np.log10(mapn2ha)).flatten(), (np.log10(mapo3hb)).flatten(), s=4, c=d.flatten(), lw=0, vmin=0,
                vmax=args.galsize/2)
    if not args.saveplot: plt.title(t)
    cb = plt.colorbar()
    # cb.ax.set_yticklabels(str(res*float(x.get_text())) for x in cb.ax.get_yticklabels())
    cb.set_label('Galactocentric distance (in kpc)')
    if args.saveplot:
        fig.savefig(args.path + t + '.eps')


# -------------------------------------------------------------------------------------------
def iterable(item):
    try:
        iter(item)
        return True
    except:
        return False


# -------------------------------------------------------------------------------------------
def meshplot2D(x, y):
    for i in range(0, np.shape(x)[0]):
        plt.plot(x[i, :], y[i, :], c='red')
    for i in range(0, np.shape(x)[1]):
        plt.plot(x[:, i], y[:, i], c='blue')
    plt.title(title(fn) + ' BPT of models')


# -------------------------------------------------------------------------------------------
def meshplot3D(x, y, args):
    age_arr = np.linspace(0., 5., 6)  # in Myr
    lnII = np.linspace(5., 12., 6)  # nII in particles/m^3
    lU = np.linspace(-4., -1., 4)  # dimensionless
    n = 3
    for k in range(0, np.shape(x)[2]):
        for i in range(0, np.shape(x)[0]):
            plt.plot(x[i, :, k], y[i, :, k], c='red', lw=0.5)
            if args.annotate and k == n: ax.annotate(str(age_arr[i]) + ' Myr', xy=(x[i, -1, k], y[i, -1, k]), \
                                                     xytext=(x[i, -1, k] - 0.4, y[i, -1, k] - 0), color='red',
                                                     fontsize=10,
                                                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3",
                                                                     color='red'))
            # plt.pause(1) #
        for i in range(0, np.shape(x)[1]):
            plt.plot(x[:, i, k], y[:, i, k], c='blue', lw=0.5)
            if args.annotate and k == n: ax.annotate('log nII= ' + str(lnII[i]), xy=(x[-2, i, k], y[-2, i, k]), \
                                                     xytext=(x[-2, i, k] + 0., y[-2, i, k] - 0.4), color='blue',
                                                     fontsize=10,
                                                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3",
                                                                     color='blue'))
            # plt.pause(1) #
        if args.annotate: ax.annotate('log U= ' + str(lU[k]), xy=(x[5, -1, k], y[5, -1, k]), \
                                      xytext=(x[5, -1, k] - 0.6, y[5, -1, k] - 1.), color='black', fontsize=10,
                                      arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='black'))

    for i in range(0, np.shape(x)[0]):
        for j in range(0, np.shape(x)[1]):
            plt.plot(x[i, j, :], y[i, j, :], c='black', lw=0.5)
            # plt.pause(1) #

    if not args.saveplot: plt.title('MAPPINGS grid of models')
    else: fig.savefig(args.path + t + '.eps')


# -------------------------------------------------------------------------------------------
def gridoverlay(args):
    s = ascii.read(HOME + '/Mappings/lab/totalspec.txt', comment='#', guess=False)
    y = np.reshape(np.log10(np.divide(s['OIII5007'], s['HBeta'])), (6, 6, 4))
    x = np.reshape(np.log10(np.divide(s['NII6584'], s['H6562'])), (6, 6, 4))
    x[[3, 4], :, :] = x[[4, 3], :, :]  # for clearer connecting lines between grid points
    y[[3, 4], :, :] = y[[4, 3], :, :]  #
    plt.scatter(x, y, c='black', lw=0, s=2)
    meshplot3D(x, y, args)
    if args.saveplot:
        fig.savefig(args.path + args.file + ':BPT overlay')


# ----------Function to measure scale length of disk---------------------------------------------------------------------------------
def knee(r, r_s, alpha):
    return r ** (1. / alpha) - np.exp(r / r_s)


def powerlaw(x, alpha):
    return x ** (1 / alpha)


def exponential(x, r_s):
    return np.exp(-x / r_s)


def func(r, r_s, alpha):
    from scipy.integrate import quad
    quad = np.vectorize(quad)
    r_knee = fminbound(knee, 0, 3, args=(r_s, alpha))  # 0.
    if args.toscreen: print 'deb166:r_s, alpha, r_knee=', r_s, alpha, r_knee, time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                            time.localtime())
    y = np.zeros(len(r))
    y += quad(powerlaw, 0., r, args=(alpha,))[0] * (r <= r_knee)
    y += (quad(powerlaw, 0., r_knee, args=(alpha,))[0] + quad(exponential, r_knee, r, args=(r_s,))[0]) * (r > r_knee)
    y /= (quad(powerlaw, 0., r_knee, args=(alpha,))[0] + quad(exponential, r_knee, np.inf, args=(r_s,))[0])
    return y


def get_scale_length(args, logbook):
    start = time.time()
    g, x, y = calcpos(logbook.s, args.center, args.galsize, args.res) # x and y range from (-galsize/2, galsize/2) kpc centering at 0 kpc

    d_list = np.sqrt(x** 2 + y** 2)  # kpc
    ha_list = [x for (y, x) in sorted(zip(d_list, logbook.s['H6562']), key=lambda pair: pair[0])]
    d_list = np.sort(d_list)
    ha_cum = np.cumsum(ha_list) / np.sum(ha_list)
    popt, pcov = curve_fit(func, d_list, ha_cum, p0=[4, 4], bounds=([1, 0.01], [7, 8]))
    scale_length, alpha = popt
    perc = .8
    light_rad = d_list[np.where(ha_cum >= perc)[0][0]]
    if not args.hide:
        fig = plt.figure()
        plt.scatter(d_list, ha_cum, s=10, lw=0, c='b', label='pixels')
        plt.plot(d_list, func(d_list, *popt), c='r',
                 label='Fit with scale length=%.2F kpc, pow=%.2F' % (scale_length, alpha))
        plt.axvline(light_rad, c='k', linestyle='--', label=str(perc) + ' percentile light radius')
        plt.axhline(perc, c='k', linestyle='--')
        ylab = 'Cumulative Halpha luminosity'
    '''
    ha_map = make2Dmap(logbook.s['H6562']/args.res**2, x, y, g, args.res)
    ha_map = np.ma.masked_where(ha_map<=0., ha_map)
    b = np.linspace(-g/2 + 1,g/2,g)*(args.galsize)/g #in kpc
    d = np.sqrt(b[:,None]**2+b**2)
    ha_list = np.log(ha_map.flatten())
    d_list = d.flatten()
    x_arr = np.arange(0,args.galsize/2,0.1)
    digitized = np.digitize(d_list, x_arr)    
    ha_list = [ha_list[digitized == i].mean() for i in range(1, len(x_arr))]
    d_list = [d_list[digitized == i].mean() for i in range(1, len(x_arr))]
    ha_list = np.ma.masked_where(np.isnan(ha_list), ha_list)
    d_list = np.ma.masked_array(d_list, ha_list.mask)
    ha_list = np.ma.compressed(ha_list)
    d_list = np.ma.compressed(d_list)            
    linefit = np.polyfit(d_list, ha_list, 1)
    scale_length = -1./linefit[0] #kpc
    if not args.hide:
        fig = plt.figure()
        plt.scatter(d_list,ha_list, s=10, lw=0, c='b', label='pixels')
        plt.plot(x_arr, np.poly1d(linefit)(x_arr), c='r',label='Inferred Ha gradient')
        plt.axhline(-1+linefit[1], linestyle='--', c='k')
        plt.axvline(scale_length, linestyle='--', c='k', label='scale_length=%.2F kpc'%scale_length)
        ylab = 'log Halpha surface brightness'
    '''
    if not args.hide:
        plt.xlabel('Galactocentric distance (kpc)')
        plt.ylabel(ylab)
        plt.xlim(0, args.galsize/2)
        plt.legend(bbox_to_anchor=(0.9, 0.42), bbox_transform=plt.gcf().transFigure)
        if not args.saveplot: plt.title('Measuring star formation scale length')
        else: fig.savefig(args.path + args.file+'_scale_length.eps')
        plt.show(block=False)

    output = 'Scale length from Halpha map = ' + str(scale_length) + ' kpc, and ' + str(
        perc * 100) + ' percentile light radius = ' + str(light_rad) + ', in %.2F min\n' % ((time.time() - start) / 60.)
    myprint(output, args)
    return scale_length


# -------------From Dopita 2016------------------------------------------------------------------------------
def get_D16_met(map_num_series, map_den_series, num_err=None, den_err=None):
    mapn2 = map_num_series[0]
    maps2, mapha = map_den_series[0], map_den_series[1]
    logOHsol = 8.77  # log(O/H)+12 value used for Solar metallicity in MAPPINGS-V, Dopita 2016
    log_ratio = np.log10(np.divide(mapn2, maps2)) + 0.264 * np.log10(np.divide(mapn2, mapha))
    logOHobj_map = log_ratio + 0.45 * (log_ratio + 0.3) ** 5  # + 8.77
    if num_err is not None:
        mapn2_u = num_err[0]
        maps2_u, mapha_u = den_err[0], den_err[1]
        n2s2_u = np.sqrt((mapn2_u / maps2) ** 2 + (mapn2 * maps2_u / maps2 ** 2) ** 2)
        n2ha_u = np.sqrt((mapn2_u / mapha) ** 2 + (mapn2 * mapha_u / mapha ** 2) ** 2)
        log_ratio_u = 0.434 * (n2s2_u / np.divide(mapn2, maps2) + 0.264 * n2ha_u / np.divide(mapn2, mapha))
        logOHobj_map_u = log_ratio_u + 0.45 * 5 * log_ratio_u * (log_ratio + 0.3) ** 4
        return logOHsol, logOHobj_map, logOHobj_map_u
    else:
        return logOHsol, logOHobj_map


# -------------From Kewley 2002------------------------------------------------------------------------------
def get_KD02_met(map_num_series, map_den_series, num_err=None, den_err=None):
    mapn2 = map_num_series[0]
    mapo2 = map_den_series[0]
    #logOHsol = 8.93  # log(O/H)+12 value used for Solar metallicity in earlier MAPPINGS in 2001
    logOHsol = 8.77 #hack to make sure both D16 and KD02 diagnostics have same zero point, does not affect the gradient
    log_ratio = np.log10(np.divide(mapn2, mapo2))
    logOHobj_map = np.log10(1.54020 + 1.26602 * log_ratio + 0.167977 * log_ratio ** 2)  # + 8.93
    if num_err is not None:
        mapn2_u = num_err[0]
        mapo2_u = den_err[0]
        n2o2_u = np.sqrt((mapn2_u / mapo2) ** 2 + (mapn2 * mapo2_u / mapo2 ** 2) ** 2)
        log_ratio_u = 0.434 * n2o2_u / np.divide(mapn2, mapo2)
        logOHobj_map_u = 0.434 * (1.26602 * log_ratio_u + 0.167977 * 2 * log_ratio * log_ratio_u) / (
                1.54020 + 1.26602 * log_ratio + 0.167977 * log_ratio ** 2)
        return logOHsol, logOHobj_map, logOHobj_map_u
    else:
        return logOHsol, logOHobj_map


# ------------Function to emulate np.ma.sum(array,axis=0) but propagating masks------------
def mysum(array, iserror=False):
    pow = 2. if iserror else 1.
    r = np.zeros(np.shape(array[0]))
    for ind in range(np.shape(array)[0]):
        r = r + array[ind] ** pow  # uncertainties are added in quadrature
    r = r ** (1. / pow)
    return r


# ------------Function to measure SNR in a line map within certain annulus *dr* of scale length------------
def get_snr_annulus(line, args, logbook, properties, dr=0.1):
    map = properties.mapcube[:, :, np.where(logbook.llist == line)[0][0]]
    map_u = properties.errorcube[:, :, np.where(logbook.llist == line)[0][0]]
    # map = np.ma.masked_where((map_u <= 0) | (map <= 0), map)
    # map_u = np.ma.masked_where((map_u <= 0) | (map <= 0), map_u)

    snr_map = map / map_u

    map_list = get_pixels_within(map, args, logbook, annulus=True, dr=dr)
    snr_list = get_pixels_within(snr_map, args, logbook, annulus=True, dr=dr)
    map_list = np.ma.compressed(np.ma.masked_where(~np.isfinite(snr_list), map_list))
    snr_list = np.ma.compressed(np.ma.masked_where(~np.isfinite(snr_list), snr_list))
    snr_list = np.ma.compressed(np.ma.masked_where(~np.isfinite(map_list), snr_list))
    map_list = np.ma.compressed(np.ma.masked_where(~np.isfinite(map_list), map_list))
    snr_list = np.array([x for (y, x) in sorted(zip(map_list, snr_list), key=lambda pair: pair[0])])
    map_list = np.sort(map_list)

    cumsum = np.cumsum(map_list) / np.sum(map_list)
    ind = np.where(map_list >= np.percentile(map_list, 50.))[0][0]
    ind2 = np.where(cumsum >= 0.5)[0][0]
    '''
    print np.shape(map_list), np.shape(snr_list), map_list, snr_list, cumsum #
    print np.percentile(map_list, 50.), ind, map_list[ind], cumsum[ind], snr_list[ind] #
    print map_list[ind2], ind2, map_list[ind2], cumsum[ind2], snr_list[ind2] #
    '''
    result = [np.mean(snr_list)]
    '''
    result += [np.mean(map_list)/np.mean(get_pixels_within(map_u, args, logbook, annulus=True, dr=dr))]
    result += [snr_list[ind]]
    result += [snr_list[ind2]]
    result += [np.mean([snr_list[np.where(map_list >= np.percentile(map_list, item))[0][0]] for item in np.linspace(50*0.9,50*1.1,10)])]
    result += [np.percentile(get_pixels_within(snr_map, args, logbook), 100 - 90.)]
    result += [10**(np.mean(np.log10(snr_list)))]
    '''
    if args.show_annulus:
        plt.figure()
        snr_list_hist = plt.hist(snr_list, histtype='step', bins=50, color='k', label='annulus')
        hist, bins = np.histogram(snr_map[np.isfinite(snr_map)].flatten(), bins=np.linspace(-50, 200, 500))
        bins = bins[:-1] + np.diff(bins)  # to get bin centers
        hist = np.array(hist) * np.max(snr_list_hist[0]) / np.max(hist)
        plt.step(bins, hist, color='r', label='entire FoV, scaled', alpha=0.5)
        plt.axvline(linestyle='dotted', lw=1, c='b')
        plt.xlim(np.min(snr_list_hist[1]) - 10, np.max(snr_list_hist[1]) + 10)
        plt.title(str(line) + ' map: SNR in +/- ' + str(dr * 100) + '% annulus of ' + str(args.scale_length) + ' kpc')
        plt.xlabel('SNR')
        plt.ylabel('Frequency')
        plt.legend(loc='best')
        plt.show(block=False)
        bins = snr_list_hist[1][:-1] + np.diff(snr_list_hist[1])
        result += [bins[np.argmax(snr_list_hist[0])]]
        '''
        plt.figure()
        plt.plot(np.arange(len(cumsum)), cumsum, c='r')
        plt.scatter(np.arange(len(cumsum)), cumsum, c='k', lw=0, s=5)
        plt.title(str(line)+' map: cumuluative distrib of flux in +/- '+str(dr*100)+'% annulus of '+str(args.scale_length)+' kpc')
        plt.xlabel('no. of pixels')
        plt.ylabel('Sum')
        plt.show(block=False)
        '''
    return result[:1]


# ------------metallicity profile (in log scale) to fit to (if using curve_fit)-------------------------------------------------------------------------------
def met_profile(x, *p):
    return p[0] * x + p[1]


# ---------returns weighted mean of 1d array, given 1sigma uncertainties--------------------------------------
def wt_mean(data, uncert):
    data, uncert = np.array(data), np.array(uncert)
    pow = 2.
    return np.sum(data / uncert ** pow) / np.sum(1. / uncert ** pow), (1. / np.sum(1. / uncert ** pow)) ** (1. / pow)


# ------------function to bin data based on another array, in a weighted average way---------------------------
def bin_data(array, data, bins, err=None):
    bins_cen = bins[:-1] + np.diff(bins) / 2.
    indices = np.digitize(array, bins)
    binned_data, binned_err = [], []
    for i in range(1, len(
            bins)):  # assuming all values in array are within limits of bin, hence last cell of binned_data=nan
        thisdata = data[indices == i]
        if err is not None:
            thiserr = err[indices == i]
            mean_data, mean_err = wt_mean(thisdata, thiserr)
        else:
            mean_data, mean_err = np.mean(thisdata), np.std(thisdata)
        binned_data.append(mean_data)
        binned_err.append(mean_err)
    binned_data = np.array(binned_data)
    binned_err = np.array(binned_err)
    return bins_cen, binned_data, binned_err


# ------------Function to measure metallicity-------------------------------------------------------------------------------
def metallicity(args, logbook, properties):
    global info
    g = np.shape(properties.mapcube)[0]
    b = np.linspace(-g / 2 + 1, g / 2, g) * (args.galsize) / g  # in kpc
    d = np.sqrt((b[:, None] - args.xcenter_offset) ** 2 + (b - args.ycenter_offset) ** 2)
    t = args.file + '_Met_Om' + str(args.Om)
    if args.smooth: t += '_arc' + str(args.res_arcsec)
    if args.spec_smear: t += '_vres=' + str(args.vres) + 'kmps'
    t += info + args.gradtext
    if args.SNR_thresh is not None: t += '_snr' + str(args.SNR_thresh)
    flux_limit = 1e-15 # in ergs/s; discard fluxes below this as machine precision errors
    nnum, nden = len(np.unique(args.num_arr)), len(np.unique(args.den_arr))
    nplots = nnum + nden + len(args.num_arr) + 1
    marker_arr = ['s', '^']
    met_minlim, met_maxlim = -2, 0.5
    fs = 20 # fontsize of labels

    if not args.inspect:
        nrow, ncol, figsize = 1, 1, (8, 8)
    else:
        nrow, ncol, figsize = int(np.ceil(nplots / 3.)), min(nplots, 3), (12, 8)

    if not args.hide:
        fig, axes = plt.subplots(nrow, ncol, sharex=True, figsize=figsize)
        fig.subplots_adjust(hspace=0.1, wspace=0.25, top=0.9, bottom=0.15, left=0.14, right=0.98)
    else:
        fig, axes = 0, 0 #dummy variables for function to return to caller

    map_num, map_num_u, map_den, map_den_u = [], [], [], []
    plot_index, already_plotted = 0, []

    for num_grp in args.num_arr:
        log_fluxmin, log_fluxmax = 0, -100
        map_num_grp, map_num_grp_u = [], []
        if not iterable(num_grp): num_grp = [num_grp]
        if not args.hide:
            if args.inspect: ax = axes[plot_index / ncol][plot_index % ncol]
            if num_grp not in already_plotted:
                plot = True
                already_plotted.append(num_grp)
            else:
                plot = False
        
        for (jj, num) in enumerate(num_grp):

            temp_u = properties.errorcube[:, :, np.where(logbook.llist == num)[0][0]] * (
                    logbook.final_pix_size * 1e3) ** 2
            temp_u = np.ma.masked_where(temp_u <= 0., temp_u)
            map_num_grp_u.append(temp_u)

            temp = properties.mapcube[:, :, np.where(logbook.llist == num)[0][0]] * (logbook.final_pix_size * 1e3) ** 2
            temp = np.ma.masked_where(temp <= flux_limit, temp) # discard too low fluxes that are machine precision errors
            if args.SNR_thresh is not None: temp = np.ma.masked_where(temp / temp_u < args.SNR_thresh, temp)
            if not args.hide and plot and args.inspect:
                myprint('fitted: ' + num + ' max= ' + str(np.max(temp)) + ', min= ' + str(
                    np.min(temp)) + ', integrated= ' + str(np.sum(temp / properties.flux_ratio)), args)
                myprint('error: ' + num + ' max= ' + str(np.max(temp_u)) + ', min= ' + str(
                    np.min(temp_u)) + ', integrated= ' + str(np.sum(temp_u / properties.flux_ratio)), args)
                if not args.nomap: dummy = plotmap(temp, num + ' map after fitting', 'Metallicity', 'log flux(ergs/s)',
                                                   args, logbook, makelog=True)
                ax.scatter(d.flatten(), np.log10(temp.flatten()), s=5, lw=0, marker=marker_arr[jj], c='b',
                           label='pixel' if plot_index == 1 else None)
                if args.showerr: ax.errorbar(d.flatten(), np.log10(temp.flatten()),
                                             yerr=0.434 * temp_u.flatten() / temp.flatten(), ls='None', c=col, fmt='',
                                             capsize=0, alpha=0.1)  # for z=log(y), \delta z = 0.434*\delta y/y
                if np.min(np.log10(np.ma.masked_where(temp <= 0, temp))) < log_fluxmin: log_fluxmin = np.min(
                    np.log10(np.ma.masked_where(temp <= 0, temp)))  # to set plot limits
                if np.max(np.log10(temp)) > log_fluxmax: log_fluxmax = np.max(np.log10(temp))  # to set plot limits
            map_num_grp.append(temp)
        map_num.append(map_num_grp)
        map_num_u.append(map_num_grp_u)
        if not args.hide and plot and args.inspect:
            ax.set_ylabel('log(' + ','.join(num_grp) + ')')
            if not args.inspect: ax.set_ylim(log_fluxmin, log_fluxmax)
            plot_index += 1

    for den_grp in args.den_arr:
        log_fluxmin, log_fluxmax = 0, -100
        map_den_grp, map_den_grp_u = [], []
        if not iterable(den_grp): den_grp = [den_grp]
        if not args.hide:
            if args.inspect: ax = axes[plot_index / ncol][plot_index % ncol]
            if den_grp not in already_plotted:
                plot = True
                already_plotted.append(den_grp)
            else:
                plot = False
        for (jj, den) in enumerate(den_grp):

            temp_u = properties.errorcube[:, :, np.where(logbook.llist == den)[0][0]] * (
                    logbook.final_pix_size * 1e3) ** 2
            temp_u = np.ma.masked_where(temp_u <= 0., temp_u)
            map_den_grp_u.append(temp_u)

            temp = properties.mapcube[:, :, np.where(logbook.llist == den)[0][0]] * (logbook.final_pix_size * 1e3) ** 2
            temp = np.ma.masked_where(temp <= flux_limit, temp) # discard too low fluxes that are machine precision errors
            if args.SNR_thresh is not None: temp = np.ma.masked_where(temp / temp_u < args.SNR_thresh, temp)
            if not args.hide and plot and args.inspect:
                myprint('fitted: ' + den + ' max= ' + str(np.max(temp)) + ', min= ' + str(
                    np.min(temp)) + ', integrated= ' + str(np.sum(temp / properties.flux_ratio)), args)
                myprint('error: ' + den + ' max= ' + str(np.max(temp_u)) + ', min= ' + str(
                    np.min(temp_u)) + ', integrated= ' + str(np.sum(temp_u / properties.flux_ratio)), args)
                if not args.nomap: dummy = plotmap(temp, den + ' map after fitting', 'Metallicity', 'log flux(ergs/s)',
                                                   args, logbook, makelog=True)
                ax.scatter(d.flatten(), np.log10(temp.flatten()), s=5, lw=0, marker=marker_arr[jj], c='b',
                           label='pixel' if plot_index == 1 else None)
                if args.showerr: ax.errorbar(d.flatten(), np.log10(temp.flatten()),
                                             yerr=0.434 * temp_u.flatten() / temp.flatten(), ls='None', c=col, fmt='',
                                             capsize=0, alpha=0.1)  # for z=log(y), \delta z = 0.434*\delta y/y
                if np.min(np.log10(np.ma.masked_where(temp <= 0, temp))) < log_fluxmin: log_fluxmin = np.min(
                    np.log10(np.ma.masked_where(temp <= 0, temp)))  # to set plot limits
                if np.max(np.log10(temp)) > log_fluxmax: log_fluxmax = np.max(np.log10(temp))  # to set plot limits
            map_den_grp.append(temp)
        map_den.append(map_den_grp)
        map_den_u.append(map_den_grp_u)
        if not args.hide and plot and args.inspect:
            ax.set_ylabel('log(' + ','.join(den_grp) + ')')
            ax.set_ylim(log_fluxmin, log_fluxmax)
            plot_index += 1

    map_num_series = [mysum(map_num[ind]) for ind in range(len(args.num_arr))]
    map_num_u_series = [mysum(map_num_u[ind], iserror=True) for ind in range(len(args.num_arr))]
    map_den_series = [mysum(map_den[ind]) for ind in range(len(args.den_arr))]
    map_den_u_series = [mysum(map_den_u[ind], iserror=True) for ind in range(len(args.den_arr))]

    if args.useKD:
        logOHsol, logOHobj_map, logOHobj_map_u = get_KD02_met(map_num_series, map_den_series, num_err=map_num_u_series,
                                                              den_err=map_den_u_series)  # =log(O/H)_obj - log(O/H)_sun; in log units
        t += '_KD02'
    else:
        logOHsol, logOHobj_map, logOHobj_map_u = get_D16_met(map_num_series, map_den_series, num_err=map_num_u_series,
                                                             den_err=map_den_u_series)  # =log(O/H)_obj - log(O/H)_sun; in log units
        t += '_D16'
    myprint('logOHobj_map before conversion med, min ' + str(np.median(logOHobj_map)) + ' ' + str(
        np.min(logOHobj_map)) + '\n', args)
    # ---------------------------------------------------
    Z_list = logOHobj_map.flatten()
    myprint('Deb474: No. of masked pixels in Z_list = ' + str(np.ma.count_masked(Z_list)) + '\n', args)  #
    myprint('Z_list after conversion med, mean, min ' + str(np.median(Z_list)) + ' ' + str(np.mean(Z_list)) + ' ' + str(
        np.min(Z_list)) + '\n', args)
    if args.noweight:
        Z_u_list = np.zeros(len(Z_list)) + 1e-5  # artificially small unfiform uncertainty
        myprint('NOT weighting metallicity in pixels by uncertainties. If you want uncertainty weighted values do not use --noweight option.', args)
    else:
        Z_u_list = logOHobj_map_u.flatten()
        myprint('Deb474: No. of masked pixels in Z_u_list = ' + str(np.ma.count_masked(Z_u_list)) + '\n', args)  #
        myprint('Z_u_list after conversion med, mean, min ' + str(np.median(Z_u_list)) + ' ' + str(
            np.mean(Z_u_list)) + ' ' + str(np.min(Z_u_list)) + '\n', args)

    if not args.inspect and not args.hide:
        choice = int(args.choice) if args.choice is not None else 0

        if choice == 0:
            col = 'b' if not args.showbin else 'gray'
        if choice == 1:
            col_map = np.log10(np.inner(properties.ppvcube, properties.delta_lambda))
            cbtitle = 'Summed up surface brightness (log ergs/s/pc^2)'
        elif choice == 2:
            col_map = np.log10(np.sum(properties.ppvcube, axis=2))
            cbtitle = 'Summed up surface brightness (log ergs/s/A/pc^2)'
        elif choice == 3:
            col_map = np.log10(map_den_series[1] / (logbook.final_pix_size * 1e3) ** 2)  # Halpha
            cbtitle = 'Ha surface brightness (log ergs/s/pc^2)'
        elif choice == 4:
            dummy, x, y = calcpos(logbook.s, args.center, args.galsize, args.res)
            col_map = np.log10(make2Dmap(logbook.s['nII'], x, y, g, args.galsize / g, weights=10 ** logbook.s['logQ0']))
            cbtitle = 'Luminosity weighted average H2R density of pixel (log cm^-3)'
        elif choice == 5:
            dummy, x, y = calcpos(logbook.s, args.center, args.galsize, args.res)
            col_map = make2Dmap(logbook.s['age(MYr)'], x, y, g, args.galsize / g, weights=10 ** logbook.s['logQ0'])
            cbtitle = 'Luminosity weighted average H2R age of pixel (Myr)'
        elif choice == 6:
            dummy, x, y = calcpos(logbook.s, args.center, args.galsize, args.res)
            col_map = np.log10(make2Dmap(10 ** logbook.s['logQ0'], x, y, g, args.galsize / g))
            cbtitle = 'Bolometric luminosity (log ergs/s)'
        elif choice == 7:
            dummy, x, y = calcpos(logbook.s, args.center, args.galsize, args.res)
            fit = mapha  # np.add(maps2a,maps2b) #
            lines = logbook.s['H6562']  # np.add(logbook.s['SII6717'],logbook.s['SII6730']) #
            true = make2Dmap(lines, x, y, g, args.galsize / g) / (logbook.final_pix_size * 1e3) ** 2
            mydiag('fitted SII map', fit, args)
            mydiag('true SII map', true, args)
            col_map = np.log10(np.abs(fit / true))
            cbtitle = 'Inferred / True Ha SB (log ergs/s/pc^2)'

        if choice: col = col_map.flatten()

    else:
        col = 'b' if not args.showbin else 'gray'

    d = np.ma.masked_array(d, logOHobj_map.mask | logOHobj_map_u.mask)
    if args.showcoord: coord_list = np.transpose(np.where(~d.mask))
    d_list = d.flatten()
    myprint('Deb474: No. of masked pixels in either Z_list or Z_u_list = ' + str(
        np.ma.count_masked(d_list)) + ' out of total ' + str(len(d_list)) + ' pixels.\n', args)  #
    d_list = np.ma.compressed(d_list)
    Z_u_list = np.ma.masked_array(Z_u_list, Z_list.mask)
    Z_list = np.ma.masked_array(Z_list, Z_u_list.mask)
    if not isinstance(col, basestring):
        col = np.ma.masked_array(col, Z_list.mask | Z_u_list.mask)
        col = np.ma.compressed(col)
    Z_u_list = np.ma.compressed(Z_u_list)
    Z_list = np.ma.compressed(Z_list)
    Z_list = np.array([x for (y, x) in sorted(zip(d_list, Z_list), key=lambda pair: pair[0])])  # sorting by distance
    Z_u_list = np.array([x for (y, x) in sorted(zip(d_list, Z_u_list), key=lambda pair: pair[0])])
    if args.showcoord: coord_list = np.array(
        [x for (y, x) in sorted(zip(d_list, coord_list), key=lambda pair: pair[0])])
    if not isinstance(col, basestring): col = np.array(
        [x for (y, x) in sorted(zip(d_list, col), key=lambda pair: pair[0])])
    d_list = np.sort(d_list)
    if args.fitupto is not None:  # to fit metallicity gradient only upto 'args.fitupto' x args.scale_length
        Z_list = Z_list[d_list <= float(args.fitupto) * args.scale_length]
        Z_u_list = Z_u_list[d_list <= float(args.fitupto) * args.scale_length]
        if args.showcoord: coord_list = coord_list[d_list <= float(args.fitupto) * args.scale_length]
        if not isinstance(col, basestring): col = col[d_list <= float(args.fitupto) * args.scale_length]
        d_list = d_list[d_list <= float(args.fitupto) * args.scale_length]

    if not args.hide:
        if args.inspect:
            for ii in range(len(map_num_series)):
                ax = axes[plot_index / ncol][plot_index % ncol]
                ax.scatter(d.flatten(), np.divide(map_num_series[ii], map_den_series[ii]).flatten(), s=5, lw=0,
                           marker=marker_arr[jj], c='b')
                if args.showerr: ax.errorbar(d.flatten(), np.divide(map_num_series[ii], map_den_series[ii]).flatten(),
                                             yerr=np.sqrt((map_num_u_series[ii] / map_den_series[ii]) ** 2 + (
                                                     map_num_series[ii] * map_den_u_series[ii] / map_den_series[
                                                 ii] ** 2) ** 2).flatten(), ls='None', c=col, fmt='', capsize=0,
                                             alpha=0.1)  # for z=log(y), \delta z = 0.434*\delta y/y
                ax.set_ylabel('+'.join(args.num_arr[ii]) + '/' + '+'.join(args.den_arr[ii]))
                ax.set_ylim(args.ratio_lim[ii])
                plot_index += 1

        ax = axes[plot_index / ncol][plot_index % ncol] if args.inspect else axes
        plot = ax.scatter(d_list, Z_list, s=5, lw=0, c=col, alpha=0.1 if args.showbin else 1)
        if not isinstance(col, basestring): plt.colorbar(plot).set_label(cbtitle)

        if args.showbin:
            limit = args.galsize/2. if args.fitupto is None else float(args.fitupto) * args.scale_length
            d_bin, Z_bin, Z_u_bin = bin_data(d_list, Z_list, np.linspace(0, limit, int(limit) + 1), err=Z_u_list)
            if args.fitbin:
                linefit, linecov = np.polyfit(d_bin, Z_bin, 1, cov=True, w=1. / Z_u_bin)
                x_arr = np.arange(args.galsize/2)
                ax.plot(x_arr, np.poly1d(linefit)(x_arr), c='b', label='Fit to bins')
                plt.legend()

            ax.errorbar(d_bin, Z_bin, yerr=Z_u_bin, ls='None', c='b', fmt='o')

        if args.showcoord:
            for ind in range(len(d_list)):
                ax.annotate('(' + str(coord_list[ind][0]) + ',' + str(coord_list[ind][1]) + ')',
                            (d_list[ind] + 0.1, Z_list[ind] - 0.02), va='top', ha='center', color='k', fontsize=4,
                            alpha=0.5)
        if args.showerr: ax.errorbar(d_list, Z_list, yerr=Z_u_list, ls='None', c=col, capsize=0, fmt='', alpha=0.5)

        ax.set_ylabel(r'$\log{(Z/Z_{\bigodot})}$', fontsize=fs if not args.inspect else None)
        if not args.inspect:
            ax.set_xlim(0, args.galsize/2)
            minor_ticks = np.arange(0, args.galsize/2, 0.2)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_xticks(np.arange(0, args.galsize/2, 2))
            ax.set_xticklabels(list(ax.get_xticks()), fontsize=fs)
            ax.set_xlabel('Galactocentric distance (kpc)', fontsize=fs)
            if not args.saveplot: fig.text(0.5, 0.95, t, ha='center')
            ax.text(args.galsize/2-0.1, met_minlim+0.7 - 0.02-.015*fs, 'True slope= %.2F' % (args.logOHgrad), va='top', ha='right', color='r', fontsize=fs)
            #ax.text(0.08, met_minlim+0.7 - 0.02-0.01*fs, 'True intercept= %.4F' % (args.logOHcen - logOHsol), va='top', color='r',fontsize=fs)

        ax.set_ylim(met_minlim, met_maxlim)
        minor_ticks = np.arange(-2, met_maxlim, 0.1)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_yticklabels(list(ax.get_yticks()), fontsize=fs if not args.inspect else None)
        ax.plot(np.arange(args.galsize/2),
                np.poly1d((args.logOHgrad, args.logOHcen))(np.arange(args.galsize/2)) - logOHsol,
                c='k' if args.inspect else 'r', label='True gradient slope=' + str('%.4F' % args.logOHgrad))
        ax.axhline(0, c='k', linestyle='--', label='Zsol')  # line for solar metallicity
        if args.getmap: dummy = plotmap(logOHobj_map, t + '_fitted', logbook.fitsname[:-5]+'_Metallicity', 'log(Z/Z_sol)', args, logbook,
                                        makelog=False, addcircle_radius=args.scale_length)
# ---to compute apparent gradient and scatter---
    if args.calcgradient:
        myprint('Fitting gradient..' + '\n', args)
        myprint('Deb513: No. of pixels being considered for gradient fit = ' + str(len(Z_list)) + '\n', args)
        if len(
                Z_list) <= 4:  # no .of data points must exceed order+2 for covariance estimation; in our case order = deg+1 = 2
            myprint('Not enough data points for vres= ' + str(args.vres) + ' above given SNR_thresh of ' + str(
                args.SNR_thresh) + '\n.', args)
        else:
            '''
            from lts_fits.lts_linefit import lts_linefit
            p = lts_linefit(d_list, Z_list, np.zeros(len(d_list)), Z_u_list, pivot=np.median(d_list), text=True, plot=True)
            linefit, linecov = [p.ab[1], p.ab[0]], [[p.ab_err[1]**2, 0.], [0., p.ab_err[0]**2]]
            '''
            linefit, linecov = np.polyfit(d_list, Z_list, 1, cov=True, w=1. / Z_u_list)
            # linefit, linecov =  curve_fit(met_profile, d_list, Z_list, p0=[-0.05, 0.], sigma=Z_u_list, absolute_sigma=True) #gives same result as np.polyfit
            myprint('Fit paramters: ' + str(linefit) + '\n', args)
            myprint('Fit errors: ' + str(linecov) + '\n', args)
            properties.logOHgrad, properties.logOHcen = linefit
            line, dr = 'NII6584', 0.1
            try:
                snr_measured = get_snr_annulus(line, args, logbook, properties, dr=dr)
                myprint(
                    'Measured mean SNR in ' + line + ' map within annulus of +/- ' + str(dr * 100) + '% of scale length (' \
                    + str(args.scale_length) + 'kpc) with different methods = ' + string.join(
                        [str('%.2F' % item) for item in snr_measured], ', ') + '\n', args)
                snr_measured = snr_measured[0]
            except:
                pass
            if not args.nowrite:
                gradfile = 'met_grad_Om='+str(args.Om)
                if args.useKD: gradfile += '_KD02'
                else: gradfile += '_D16'
                if args.scale_exptime: gradfile += '_exp_scaled'
                if args.fixed_SNR is not None: gradfile += '_fixed_SNR'
                if args.fixed_noise is not None: gradfile += '_fixed_noise'
                if args.binto is not None: gradfile += '_binto' + str(args.binto)
                if args.fitupto is not None: gradfile += '_fitupto' + str(args.fitupto)
                if args.mergespec is not None: gradfile += '_branches_merged'
                gradfile += '.txt'
                if not os.path.exists(gradfile):
                    head = '#File to store metallicity gradient information for different telescope parameters, as pandas dataframe\n\
#Columns are:\n\
#simulation : name of simulation from which star particles were extracted\n\
#res_arcsec : spatial resoltn of telescope\n\
#vres : spectral resoltn of telescope\n\
#pow, size : smoothing parameters, assuming Moffat profile\n\
#fixed_SNR : SNR applied before painting noise, if fixed_SNR mode (-99 if not fixed_SNR mode)\n\
#fixed_noise : SNR applied before painting fixed noise, if fixed_noise mode (-99 if not fixed_noise mode)\n\
#slope, intercept : fitted parameters\n\
#slope_u, intercept_u : uncertainty in above quantities\n\
#exptime: exposure time (s); not relevant for fixed_SNR mode\n\
#realisation : which random realisation\n\
#snr_cut : snr_cut applied, if any (-99 if no cut applied)\n\
#snr_measured : mean SNR of NII6584 line map measured at 0.9 - 1.1 x scale_radius\n\
#by Ayan\n\
simulation  res_arcsec   res_phys   vres        power   size    logOHcen        logOHgrad       fixed_SNR      fixed_noise      snr_measured     slope       slope_u     \
intercept       intercept_u       exptime       snr_cut         realisation\n'
                    open(gradfile, 'w').write(head)
                snr_cut_text = str(args.SNR_thresh) if args.SNR_thresh is not None else '-99'
                fixed_SNR_text = str(args.fixed_SNR) if args.fixed_SNR is not None else '-99'
                fixed_noise_text = str(args.fixed_noise) if args.fixed_noise is not None else '-99'
                with open(gradfile, 'a') as fout:
                    output = '\n' + args.file + '\t\t' + str(args.res_arcsec) + '\t\t' + str(logbook.fwhm * \
                                                                                             logbook.intermediate_pix_size) + '\t\t' + str(
                        args.vres) + '\t\t' + str(args.pow) + '\t\t' + \
                             str(args.size) + '\t\t' + str(args.logOHcen) + '\t\t' + '\t\t' + str(
                        args.logOHgrad) + '\t\t' + '\t\t' + \
                             fixed_SNR_text + '\t\t' + '\t\t' + fixed_noise_text + '\t\t' + '\t\t' + str(
                        '%.2F' % snr_measured) + '\t\t' + '\t\t' + str('%0.4F' % properties.logOHgrad) + '\t\t' + str(
                        '%0.4F' % np.sqrt(linecov[0][0])) + \
                             '\t\t' + str('%0.4F' % properties.logOHcen) + '\t\t' + '\t\t' + str(
                        '%0.4F' % np.sqrt(linecov[1][1])) + '\t\t' + \
                             '\t\t' + str(
                        float(logbook.exptime)) + '\t\t' + '\t\t' + snr_cut_text + '\t\t' + '\t\t' + str(
                        args.multi_realisation)
                    fout.write(output)

            x_arr = np.arange(args.galsize/2)
            if not args.hide:
                ax.plot(x_arr, np.poly1d(linefit)(x_arr), c='b',
                        label='Inferred gradient slope=' + str('%.4F' % properties.logOHgrad))
                if not args.inspect:
                    ax.text(args.galsize/2-0.1, met_minlim+0.7 - 0.02-.025*fs,
                            'Inferred slope= %.4F +/- %.4F' % (properties.logOHgrad, np.sqrt(linecov[0][0])), va='top',
                            ha='right', color='k', fontsize=fs)
                    #ax.text(0.08, met_minlim+0.7 - 0.02-.035*fs, 'Inferred intercept= %.4F +/- %.4F' % (properties.logOHcen, np.sqrt(linecov[1][1])),va='top', color='k', fontsize=fs)

    if args.saveplot and not args.hide and not args.inspect:
        fig.savefig(args.path + t + '.eps')
        myprint('Saved file ' + args.path + t + '.eps', args)

    return properties, axes


# -------------Function to undo continuum normalisation (OR subtraction)------------
def undo_contnorm(flux, cont, contsub=False):
    if contsub:
        return flux + cont
    else:
        return flux * cont


# -------------Fucntion for fitting multiple lines----------------------------
def fit_all_lines(args, logbook, wave, flam, flam_u, cont, pix_i, pix_j, z=0, z_err=0.0001):
    scaling = 1e-19 if args.contsub else 1.  # to make "good" numbers that python can handle
    flam /= scaling
    flam_u /= scaling
    cont /= scaling
    kk, count, flux_array, flux_error_array = 1, 0, [], []
    ndlambda_left, ndlambda_right = [
                                        args.nres] * 2  # how many delta-lambda wide will the window (for line fitting) be on either side of the central wavelength, default 5
    try:
        count = 1
        first, last = [logbook.wlist[0]] * 2
    except IndexError:
        pass
    while kk <= len(logbook.llist):
        center1 = last
        if kk == len(logbook.llist):
            center2 = 1e10  # insanely high number, required to plot last line
        else:
            center2 = logbook.wlist[kk]
        if center2 * (1. - ndlambda_left / logbook.resoln) > center1 * (1. + ndlambda_right / logbook.resoln):
            leftlim = first * (1. - ndlambda_left / logbook.resoln)
            rightlim = last * (1. + ndlambda_right / logbook.resoln)
            wave_short = wave[(leftlim < wave) & (wave < rightlim)]
            flam_short = flam[(leftlim < wave) & (wave < rightlim)]
            flam_u_short = flam_u[(leftlim < wave) & (wave < rightlim)]
            cont_short = cont[(leftlim < wave) & (wave < rightlim)]
            if args.debug: myprint(
                'Trying to fit ' + str(logbook.llist[kk - count:kk]) + ' line/s at once. Total ' + str(count) + '\n',
                args)
            try:
                popt, pcov = fitline(wave_short, flam_short, flam_u_short, logbook.wlist[kk - count:kk], logbook.resoln,
                                     z=z, z_err=z_err, contsub=args.contsub)
                popt, pcov = np.array(popt), np.array(pcov)
                level = 0. if args.contsub else 1.
                popt = np.concatenate(([level],
                                       popt))  # for fitting after continuum normalised (OR subtracted), so continuum is fixed=1 (OR 0) and has to be inserted to popt[] by hand after fitting
                pcov = np.hstack((np.zeros((np.shape(pcov)[0] + 1, 1)), np.vstack((np.zeros(np.shape(pcov)[1]),
                                                                                   pcov))))  # for fitting after continuum normalised (OR subtracted), so error in continuum is fixed=0 and has to be inserted to pcov[] by hand after fitting
                if args.showfit:  #
                    plt.axvline(leftlim, linestyle='--', c='g')
                    plt.axvline(rightlim, linestyle='--', c='g')

                ndlambda_left, ndlambda_right = [args.nres] * 2
                if args.debug: myprint('Done this fitting!' + '\n', args)

            except TypeError, er:
                if args.debug: myprint('Trying to re-do this fit with broadened wavelength window..\n', args)
                ndlambda_left += 1
                ndlambda_right += 1
                continue
            except (RuntimeError, ValueError), e:
                level = 0. if args.contsub else 1.
                popt = np.concatenate(([level], np.zeros(
                    count * 3)))  # if could not fit the line/s fill popt with zeros so flux_array gets zeros
                pcov = np.zeros((count * 3 + 1,
                                 count * 3 + 1))  # if could not fit the line/s fill popt with zeros so flux_array gets zeros
                if args.debug: myprint('Could not fit lines ' + str(logbook.llist[kk - count:kk]) + ' for pixel ' + str(
                    pix_i) + ', ' + str(pix_j) + '\n', args)
                pass

            for xx in range(0, count):
                # in popt for every bunch of lines,
                # elements (0,1,2) or (3,4,5) etc. are the height(b), mean(c) and width(d)
                # so, for each line the elements (cont=a,0,1,2) or (cont=a,3,4,5) etc. make the full suite of (a,b,c,d) gaussian parameters
                # so, for each line, flux f (area under gaussian) = sqrt(2pi)*(b-a)*d
                # also the full covariance matrix pcov looks like:
                # |00 01 02 03 04 05 06 .....|
                # |10 11 12 13 14 15 16 .....|
                # |20 21 22 23 24 25 26 .....|
                # |30 31 32 33 34 35 36 .....|
                # |40 41 42 43 44 45 46 .....|
                # |50 51 52 53 54 55 56 .....|
                # |60 61 62 63 64 65 66 .....|
                # |.. .. .. .. .. .. .. .....|
                # |.. .. .. .. .. .. .. .....|
                #
                # where, 00 = var_00, 01 = var_01 and so on.. (var = sigma^2)
                # let var_aa = vaa (00), var_bb = vbb(11), var_ab = vab(01) = var_ba = vba(10) and so on..
                # for a single gaussian, f = const * (b-a)*d
                # i.e. sigma_f^2 = d^2*(saa^2 + sbb^2) + (b-a)^2*sdd^2 (dropping the constant for clarity of explanation)
                # i.e. var_f = d^2*(vaa + vbb) + (b-a)^2*vdd
                # the above holds if we assume covariance matrix to be diagonal (off diagonal terms=0) but thats not the case here
                # so we also need to include off diagnoal covariance terms while propagating flux errors
                # so now, for each line, var_f = d^2*(vaa + vbb) + (b-a)^2*vdd + 2d^2*vab + 2d*(b-a)*(vbd - vad)
                # i.e. in terms of element indices,
                # var_f = 3^2(00 + 11) + (1-0)^2*33 - (2)*3^2*10 + (2)*3*(1-0)*(13-03),
                # var_f = 6^2(00 + 44) + (4-0)^2*66 - (2)*6^2*40 + (2)*6*(4-0)*(46-06),
                # var_f = 9^2(00 + 77) + (1-0)^2*99 - (2)*9^2*70 + (2)*9*(7-0)*(79-09), etc.
                #
                popt_single = np.concatenate(([popt[0]], popt[3 * xx + 1:3 * (xx + 1) + 1]))
                cont_at_line = cont[np.where(wave >= logbook.wlist[kk + xx - count])[0][0]]
                if args.debug and args.oneHII is not None: print 'Debugging534: linefit param at (', pix_i, ',', pix_j, ') for ', \
                    logbook.llist[kk + xx - count], '(ergs/s/pc^2/A) =', popt_single  #
                flux = np.sqrt(2 * np.pi) * (popt_single[1] - popt_single[0]) * popt_single[
                    3] * scaling  # total flux = integral of guassian fit ; resulting flux in ergs/s/pc^2 units
                if args.debug and not args.contsub: flux *= cont_at_line  # if continuum is normalised (and NOT subtracted) then need to change back to physical units by multiplying continuum at that wavelength
                if args.oneHII is not None: print 'Debugging536: lineflux at (', pix_i, ',', pix_j, ') for ', \
                    logbook.llist[kk + xx - count], '(ergs/s/pc^2/A) =', flux  #
                flux_array.append(flux)
                flux_error = np.sqrt(2 * np.pi * (popt_single[3] ** 2 * (pcov[0][0] + pcov[3 * xx + 1][3 * xx + 1]) \
                                                  + (popt_single[1] - popt_single[0]) ** 2 * pcov[3 * (xx + 1)][
                                                      3 * (xx + 1)] \
                                                  - 2 * popt_single[3] ** 2 * pcov[3 * xx + 1][0] \
                                                  + 2 * (popt_single[1] - popt_single[0]) * popt_single[3] * (
                                                          pcov[3 * xx + 1][3 * (xx + 1)] - pcov[0][3 * (xx + 1)]) \
                                                  )) * scaling  # var_f = 3^2(00 + 11) + (1-0)^2*33 - (2)*3^2*10 + (2)*3*(1-0)*(13-03)
                if not args.contsub: flux_error *= cont_at_line  # if continuum is normalised (and NOT subtracted) then need to change back to physical units by multiplying continuum at that wavelength
                flux_error_array.append(flux_error)
                if args.showfit:
                    leftlim = popt_single[2] * (1. - args.nres / logbook.resoln)
                    rightlim = popt_single[2] * (1. + args.nres / logbook.resoln)
                    wave_short_single = wave[(leftlim < wave) & (wave < rightlim)]
                    cont_short_single = cont[(leftlim < wave) & (wave < rightlim)]
                    if args.contsub: plt.plot(wave_short_single, (su.gaus(wave_short_single,1, *popt_single) + cont_short_single)*scaling,lw=1, c='r') # adding back the continuum just for plotting purpose
                    else: plt.plot(wave_short_single, su.gaus(wave_short_single,1, *popt_single)*cont_short_single,lw=1, c='r')
                    count = 1
            if args.showfit:
                if count > 1:
                    plt.plot(wave_short, undo_contnorm(su.gaus(wave_short, count, *popt), cont_short,
                                                       contsub=args.contsub), lw=2, c='brown')
                plt.draw()

            first, last = [center2] * 2
        else:
            last = center2
            count += 1
        kk += 1
    # -------------------------------------------------------------------------------------------
    flux_array = np.array(flux_array)
    flux_error_array = np.array(flux_error_array)
    return flux_array, flux_error_array


# -------------------------------------------------------------------------------------------
def fitline(wave, flam, flam_u, wtofit, resoln, z=0, z_err=0.0001, contsub=False):
    v_maxwidth = 10 * c / resoln  # 10*vres in km/s
    z_allow = 3 * z_err  # wavelengths are at restframe; assumed error in redshift
    p_init, lbound, ubound = [], [], []
    for xx in range(0, len(wtofit)):
        fl = np.max(flam)  # flam[(np.abs(wave - wtofit[xx])).argmin()] #flam[np.where(wave <= wtofit[xx])[0][0]]
        p_init = np.append(p_init, [fl - flam[0], wtofit[xx], wtofit[xx] * 2. * gf2s / resoln])
        lbound = np.append(lbound, [0., wtofit[xx] * (1. - z_allow / (1. + z)), wtofit[xx] * 1. * gf2s / resoln])
        ubound = np.append(ubound, [np.inf, wtofit[xx] * (1. + z_allow / (1. + z)), wtofit[xx] * v_maxwidth * gf2s / c])
    level = 0. if contsub else 1.
    if flam_u.any():
        # popt, pcov = curve_fit(lambda x, *p: fixcont_erf(x, level, len(wtofit), *p),wave,flam,p0= p_init, max_nfev=10000, bounds = (lbound, ubound))
        popt, pcov = curve_fit(lambda x, *p: fixcont_erf(x, level, len(wtofit), *p), wave, flam, p0=p_init,
                               maxfev=10000, bounds=(lbound, ubound), sigma=flam_u, absolute_sigma=True)
    else:
        popt, pcov = curve_fit(lambda x, *p: fixcont_erf(x, level, len(wtofit), *p), wave, flam, p0=p_init,
                               max_nfev=10000, bounds=(lbound, ubound))
    '''         
    plt.figure() #
    plt.plot(wave, flam, c='k', label='flam') #
    plt.plot(wave, flam_u, c='gray', label='flam_u') #
    #plt.plot(wave, cont_short, c='g', label='cont') #
    plt.plot(wave, fixcont_erf(wave, level, len(wtofit), *popt), c='r', label='fit') #
    plt.plot(wave, fixcont_erf(wave, level, len(wtofit), *p_init), c='b', label='init guess') #
    plt.ylim(-0.2e-18, 1.2e-18) #
    plt.legend() #
    '''
    return popt, pcov


# -------------------------------------------------------------------------------------------
def emissionmap(args, logbook, properties, additional_text=''):
    map = properties.mapcube[:, :, np.where(logbook.llist == args.line)[0][0]]
    map_u = properties.errorcube[:, :, np.where(logbook.llist == args.line)[0][0]]
    if args.SNR_thresh is not None: map = np.ma.masked_where(map / map_u < args.SNR_thresh, map)
    if not args.hide:
        t = additional_text + '_'+ args.line + '_map:\n' + logbook.fitsname
        linename_dict = {'OII3727':r'\mathrm{O}\mathtt{II}\;\lambda 3727', 'OII3729':r'\mathrm{O}\mathtt{II}\;\lambda 3729', 'H6562':r'\mathrm{H}\alpha', 'NII6584':r'\mathrm{N}\mathtt{II}\;\lambda 6584', 'SII6717':r'\mathrm{S}\mathtt{II}\;\lambda 6717', 'SII6730':r'\mathrm{S}\mathtt{II}\;\lambda 6730'}
        linename = linename_dict[args.line]
        if args.add_text_to_plot == 'line':
            text_on_plot = r'$%s$'%linename
            linename = '\mathrm{f}_\lambda'
        elif args.add_text_to_plot == 'res':
            text_on_plot = 'PSF %.1F"'%(args.res_arcsec)
        elif args.add_text_to_plot == 'snr':
            if args.fixed_noise is not None and args.addnoise:
                text_on_plot = 'SNR=%.0F' % (args.fixed_noise * args.scale_SNR)
            elif args.fixed_SNR is not None and args.addnoise:
                text_on_plot = 'SNR=%.0F' % (args.fixed_SNR * args.scale_SNR)
            else:
                text_on_plot = 'No noise'
        else:
            text_on_plot = None


        dummy = plotmap(map, t, logbook.fitsname[:-5]+'_'+args.line + '_map', r'$\log{(' + linename + '\:\mathrm{ergs\,s}^{-1}\mathrm{cm}^{-2}\mathrm{arcsec}^{-2})}$', args, logbook,
                        addcircle_radius=args.scale_length, text_on_plot=text_on_plot)

        if args.snrmap:
            snr_map = map / map_u
            mydiag('SNR map', snr_map, args)
            percentile, dr = 90, 0.1
            snr_frac = np.percentile(get_pixels_within(snr_map, args, logbook), 100 - percentile)
            myprint(str(percentile) + '% of pixels within ' + str(args.scale_length) + 'kpc have SNR >= ' + str(snr_frac),
                    args)
            snr_measured = np.mean(get_pixels_within(snr_map, args, logbook, annulus=True, dr=dr))
            myprint(
                'Measured mean SNR in ' + args.line + ' map within annulus of +/- ' + str(dr * 100) + '% of scale length (' \
                + str(args.scale_length) + 'kpc) = ' + str('%.2F' % snr_measured) + '\n', args)
            if args.snrhist:
                plt.figure()
                if args.cmin is not None: snr_map = np.ma.masked_where(np.log10(map) < args.cmin, snr_map)
                if args.cmax is not None: snr_map = np.ma.masked_where(np.log10(map) > args.cmax, snr_map)
                plt.hist(np.ma.compressed(snr_map.flatten()), bins=100, range=(0, 1000))
            # dummy = plotmap(snr_map, args.line+' SNR map', 'junk', 'log(SNR)', args, logbook, makelog=True, addcircle_radius=args.scale_length, issnrmap=True)
            dummy = plotmap(snr_map, args.line + '_SNRmap', logbook.fitsname[:-5]+'_'+args.line + '_SNRmap', 'SNR', args, logbook, makelog=False,\
                            addcircle_radius=args.scale_length, issnrmap=True, text_on_plot=text_on_plot)

        if additional_text != 'onlyDIG':
            dr = 0.1
            snr_measured = get_snr_annulus(args.line, args, logbook, properties, dr=dr)
            myprint('Measured mean SNR in ' + args.line + ' map within annulus of +/- ' + str(dr * 100) + '% of scale length (' \
                    + str(args.scale_length) + 'kpc) with different methods = ' + string.join(
                [str('%.2F' % item) for item in snr_measured], ', ') + '\n', args)

    return map


# -------------------------------------------------------------------------------------------
def SFRmaps(args, logbook, properties):
    global info
    ages = logbook.s['age(MYr)']
    masses = logbook.s['mass(Msun)']
    # ----------to get correct conversion rate for lum to SFR----------------------------------------------
    SBmodel = 'starburst08'  # this has continuous 1Msun/yr SFR
    input_quanta = HOME + '/SB99-v8-02/output/' + SBmodel + '/' + SBmodel + '.quanta'
    SB99_age = np.array([float(x.split()[0]) for x in open(input_quanta).readlines()[6:]])
    SB99_logQ = np.array([float(x.split()[1]) for x in open(input_quanta).readlines()[6:]])
    const = 1. / np.power(10, SB99_logQ)
    # -------------------------------------------------------------------------------------------
    # const = 7.9e-42*1.37e-12 #factor to convert Q0 to SFR, value from literature
    const = 1.0 * const[-1]  # factor to convert Q0 to SFR, value from corresponding SB99 file (07)
    # d = np.sqrt(b[:,None]**2+b**2)
    SFRmapHa = properties.mapcube[:, :, np.where(logbook.llist == 'H6562')[0][0]]
    g, x, y = calcpos(logbook.s, args.center, args.galsize, args.res)
    SFRmap_real = make2Dmap(masses, x, y, g, args.res) / ((res * 1e3) ** 2)
    agemap = 1e6 * make2Dmap(ages, x, y, g, args.res, domean=True)
    # SFRmap_real /= agemap #dividing by mean age in the box
    SFRmap_real /= 5e6  # dividing by straight 5M years
    SFRmap_real[np.isnan(SFRmap_real)] = 0
    SFRmap_real = rebin(SFRmap_real, np.shape(SFRmapHa))
    SFRmap_real = np.ma.masked_where(SFRmap_real <= 0., SFRmap_real)
    SFRmapHa *= (const / 1.37e-12)  # Msun/yr/pc^2

    t = title(args.file) + 'SFR map for Omega = ' + str(args.Om) + ', resolution = ' + str(
        logbook.final_pix_size) + ' kpc' + info
    if args.getmap:
        SFRmap_real = plotmap(SFRmap_real, t, 'SFRmap_real', 'Log SFR(real) density in Msun/yr/pc^2', args, logbook)
        SFRmapHa = plotmap(SFRmapHa, t, 'SFRmapHa', 'Log SFR(Ha) density in Msun/yr/pc^2', galsize, args, logbook)
    else:
        fig = plt.figure(figsize=(8, 6))
        fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.1, left=0.1, right=0.95)
        ax = plt.subplot(111)
        ax.scatter((np.log10(SFRmap_real)).flatten(), (np.log10(SFRmapHa)).flatten(), s=4, c='b', lw=0, label='SFR(Ha)')
        # ax.scatter((np.log10(SFRmap_real)).flatten(),(np.log10(SFRmapHa)).flatten(), s=4, c=d.flatten(), lw=0, label='SFR(Ha)')
        # -----to plot x=y line----------#
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label='x=y line')
        # -------------------------------#
        t = 'SFR comparison for ' + args.file + ', res ' + str(logbook.final_pix_size) + ' kpc' + info
        plt.ylabel('Log (Predicted SFR density) in Msun/yr/pc^2')
        plt.xlabel('Log (Actual SFR density) in Msun/yr/pc^2')
        if not args.saveplot: plt.title(t)
        # plt.colorbar().set_label('Galactocentric distance (in pix)')
        plt.legend(bbox_to_anchor=(0.35, 0.88), bbox_transform=plt.gcf().transFigure)
        if saveplot:
            fig.savefig(args.path + title(args.file)[:-2] + '_' + t + '.eps')
        '''
        ax.scatter(reso,mean,s=4)
        plt.xlabel('res(kpc)')
        plt.ylabel('log(mean_SFR_density) in Msun/yr/kpc^2')
        plt.title('Resolution dependence of SFR density for '+fn)
        '''
    return SFRmap_real, SFRmapHa


# -------------------------------------------------------------------------------------------------
def readSB(wmin, wmax):
    inpSB = open(HOME + '/SB99-v8-02/output/starburst08/starburst08.spectrum', 'r')  # sb08 has cont SF
    speclines = inpSB.readlines()[5:]
    age = [0., 1., 2., 3., 4., 5.]  # in Myr
    funcar = []
    for a in age:
        cw, cf = [], []
        for line in speclines:
            if float(line.split()[0]) / 1e6 == a:
                if wmin - 150. <= float(line.split()[1]) <= wmax + 150.:
                    cw.append(float(line.split()[1]))  # column 1 is wavelength in A
                    cf.append(10 ** float(line.split()[3]))  # column 3 is stellar continuum in ergs/s/A
        funcar.append(interp1d(cw, cf, kind='cubic'))
        # plt.plot(cw, np.divide(cf,5e34),lw=0.5, linestyle='--') #
    return funcar


# -------------------------------------------------------------------------------------------
def spec_at_point(args, logbook, properties):
    global info
    fig = plt.figure(figsize=(14, 6))
    fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.05, right=0.95)
    for i in logbook.wlist:
        plt.axvline(i, ymin=0.9, c='black')
    cbarlab = 'Log surface brightness in erg/s/pc^2'  # label of color bar
    plt.plot(properties.dispsol, np.log10(properties.ppvcube[args.X][args.Y][:]), lw=1, c='b')
    t = 'Spectrum at pp ' + str(args.X) + ',' + str(args.Y) + ' for ' + logbook.fitsname
    if not args.saveplot: plt.title(t)
    plt.ylabel(cbarlab)
    plt.xlabel('Wavelength (A)')
    plt.ylim(30, 37)
    plt.xlim(logbook.wmin, logbook.wmax)
    if not args.hide:
        plt.show(block=False)
    if args.saveplot:
        fig.savefig(path + t + '.eps')


# -------------------------------------------------------------------------------------------
def plotintegmap(args, logbook, properties):
    ppv = properties.ppvcube[:, :, (properties.dispsol >= logbook.wmin) & (properties.dispsol <= logbook.wmax)]
    cbarlab = 'Log surface brightness in erg/s/pc^2'  # label of color bar
    line = 'lambda-integrated wmin=' + str(logbook.wmin) + ', wmax=' + str(logbook.wmax) + '\n'
    map = np.sum(ppv, axis=2)
    t = title(args.file) + line + ' map for Omega = ' + str(args.Om) + ', res = ' + str(logbook.final_pix_size) + ' kpc'
    dummy = plotmap(map, t, line, cbarlab, args, logbook)
    return dummy


# -------------------------------------------------------------------------------------------
def spec_total(w, ppv, thistitle, args, logbook):
    cbarlab = 'Surface brightness in erg/s/A'  # label of color bar
    fig = plt.figure(figsize=(14, 6))
    fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.1, right=0.95)
    ax = plt.subplot(111)
    for i in logbook.wlist:
        plt.axvline(i, ymin=0.9, c='black')

        # -------------------------------------------------------------------------------------------
    y = np.sum(ppv, axis=(0, 1))
    plt.plot(w, y, lw=1)
    t = thistitle + ', for ' + title(args.file) + ' Nebular+ stellar for Om = ' + str(args.Om) + ', res = ' + str(
        logbook.final_pix_size) + ' kpc' + info
    # -------------------------------------------------------------------------------------------
    if not args.saveplot: plt.title(t)
    plt.ylabel(cbarlab)
    plt.xlabel('Wavelength (A)')
    plt.ylim(np.min(y) * 0.9, np.max(y) * 1.1)
    plt.xlim(logbook.wmin, logbook.wmax)
    plt.show(block=False)
    if args.saveplot:
        fig.savefig(args.path + t + '.eps')
    if args.debug: plt.show(block=False)


# -----------to get the sky normalization from skynoise provided by Rob--------------------------------------------------------------------------------
def get_sky_norm(properties, logbook, args):
    if args.branch == 'blue':
        args_dummy = deepcopy(args)
        logbook_dummy = deepcopy(logbook)
        properties_dummy = deepcopy(properties)
        args_dummy.wmin, args_dummy.wmax = 6400, None
        args_dummy.branch = None
        args_dummy, logbook_dummy = getfitsname(args_dummy, properties_dummy)  # name of fits file to be written into
        properties_dummy = get_disp_array(args_dummy, logbook_dummy, properties_dummy)
        llist = logbook_dummy.llist
        wlist = logbook_dummy.wlist
        wave = properties_dummy.dispsol
        properties_dummy.skynoise = getskynoise(args_dummy, logbook_dummy, properties_dummy)
        skynoise = properties_dummy.skynoise
        delta_lambda = properties_dummy.delta_lambda

    else:
        llist = logbook.llist
        wlist = logbook.wlist
        wave = properties.dispsol
        skynoise = properties.skynoise
        delta_lambda = properties.delta_lambda
    
    lambda_NII = wlist[llist == 'NII6584'][0]
    index_NII = np.where(wave >= lambda_NII)[0][0]
    sky_norm = skynoise[index_NII] # properties.skynoise in units of ergs/s/cm^2/A/arcsec^2;
    '''
    R_band_range = [5500., 8000.] #Angstrom
    skynoise_Rband = skynoise[(wave>=R_band_range[0]) & (wave<=R_band_range[1])]
    sky_norm = np.sum(skynoise_Rband) #since it is already in units of noise*dlambda, just adding = integration
    sky_norm /= np.diff(R_band_range)[0] #to divide by wavelength range, to bring sky_norm to units of el/s/spaxel/A
    '''
    return sky_norm


# -------------to get the dispersion solution------------------------------------------------------------------------------
def get_disp_array(args, logbook, properties):
    sig = 5 * args.vdel / c
    w = np.linspace(logbook.wmin, logbook.wmax, args.nbin)
    for ii in logbook.wlist:
        w1 = ii * (1 - sig)
        w2 = ii * (1 + sig)
        highres = np.linspace(w1, w2, args.nhr)
        w = np.hstack((w[:np.where(w < w1)[0][-1] + 1], highres, w[np.where(w > w2)[0][0]:]))
    properties.w = w
    # -------------------------------------------------------------------------------------------
    if args.spec_smear:
        properties.new_w = [properties.w[0]]
        while properties.new_w[-1] <= properties.w[-1]:
            properties.new_w.append(properties.new_w[-1] * (1 + args.vres / c))
        properties.bin_index = np.digitize(properties.w, properties.new_w)

    properties.dispsol = np.array(properties.new_w[1:]) if args.spec_smear else np.array(properties.w)
    properties.nwbin = len(properties.dispsol)
    properties.delta_lambda = np.array(
        [properties.dispsol[1] - properties.dispsol[0]] + [(properties.dispsol[i + 1] - properties.dispsol[i - 1]) / 2
                                                           for i in range(1, len(properties.dispsol) - 1)] + [
            properties.dispsol[-1] - properties.dispsol[-2]])  # wavelength spacing for each wavecell; in Angstrom

    if args.debug: myprint(
        'Deb663: for vres= ' + str(args.vres) + ', length of dispsol= ' + str(len(properties.dispsol)) + '\n', args)
    return properties


# -------------------------------------------------------------------------------------------
def spec(args, logbook, properties):
    global info
    properties = get_disp_array(args, logbook, properties)
    if os.path.exists(logbook.H2R_filename) and args.oneHII is None:
        ppv = fits.open(logbook.H2R_filename)[0].data
        myprint('Reading existing H2R file cube from ' + logbook.H2R_filename + '\n', args)
    else:
        # -------------------------------------------------------------------------------------------
        g, x, y = calcpos(logbook.s, args.center, args.galsize, args.res)
        ppv = np.zeros((g, g, properties.nwbin))
        funcar = readSB(logbook.wmin, logbook.wmax)
        # -------------------------------------------------------------------------------------------
        if args.debug and args.oneHII is not None:
            print 'Debugging750: Mappings fluxes (ergs/s/pc^2) for H2R #', args.oneHII, '=', np.array(
                [logbook.s[line][args.oneHII] for line in ['H6562', 'NII6584', 'SII6717', 'SII6730']]) / (
                                                                                                     args.res * 1e3) ** 2
            startHII, endHII = args.oneHII, args.oneHII + 1
        else:
            startHII, endHII = 0, len(logbook.s)

        for count, j in enumerate(range(startHII, endHII)):
            myprint('Particle ' + str(count + 1) + ' of ' + str(endHII - startHII) + '\n', args)
            vz = float(logbook.s['vz'][j])
            a = int(round(logbook.s['age(MYr)'][j]))
            f = np.multiply(funcar[a](properties.w), (
                    300. / 1e6))  # to scale the continuum by 300Msun, as the ones produced by SB99 was for 1M Msun
            # the continuum is in ergs/s/A

            if args.debug and count == 0:
                fig = plt.figure(figsize=(14, 6))
                plt.plot(properties.w, f, label='cont')
                plt.xlim(logbook.wmin, logbook.wmax)
                plt.xlabel('Wavelength (A)')
                plt.ylabel('flam (ergs/s/A)')

            flist = []
            for l in logbook.llist:
                try:
                    flist.append(logbook.s[l][j])  # ergs/s
                except:
                    continue

            for i, fli in enumerate(flist):
                f = gauss(properties.w, f, logbook.wlist[i], fli, args.vdisp,
                          vz)  # adding every line flux on top of continuum; gaussians are in ergs/s/A

            if args.debug and count == 0: plt.plot(properties.w, f, label='cont + lines')

            if args.spec_smear:
                f = np.array([f[properties.bin_index == ii].mean() for ii in range(1,properties.nwbin + 1)])  # spectral smearing i.e. rebinning of spectrum                                                                                                                             #mean() is used here to conserve flux; as f is in units of ergs/s/A, we want integral of f*dlambda to be preserved (same before and after resampling)
                # this can be checked as np.sum(f[1:]*np.diff(wavelength_array))

            if args.debug and count == 0:
                if args.spec_smear: plt.plot(properties.dispsol, f, label='cont+lines+smeared:vres= ' + str(args.vres))
                plt.title('For just one H2R #%d'%startHII)
                plt.legend()
                if args.oneHII is None: plt.ylim(0.4e34, 1e34)
                plt.xlim(6500,6650) #
                plt.show(block=False)

            #f *= properties.flux_ratio  # converting from emitted flux to flux actually entering each pixel, after multiplication f is in ergs/s/A/cm^2/arcsec^2
            ppv[int((x[j]+args.galsize/2.) / args.res)][int((y[j]+args.galsize/2.) / args.res)][:] += f  # f is ergs/s/A, ppv becomes ergs/s/A/pixel
                                                                                                        # x[j], y[j] used to range from (-galsize/2, galsize/2) kpc, which is changed here to (0, galsize) kpc
        # -------------------------------------------------------------------------------------------
        myprint('Done reading in ' + str(endHII - startHII) + ' HII regions in ' + str(
            (time.time() - start_time) / 60) + ' minutes.\n', args)
        write_fits(logbook.H2R_filename, ppv, args)
    if args.oneHII is not None: myexit(args, text='Testing for only one H2R case.')
    # ------------------------------------------------------------------------#
    if args.debug:
        print 'Deb1322: in ergs/s/A/pixel: for H2R cube: shape=', np.shape(ppv) ##
        '''
        spec_total(properties.dispsol, ppv, 'Spectrum for only H2R after spec smear', args, logbook)
        myprint(
            'Deb705: Trying to calculate some statistics on the cube of shape (' + str(np.shape(ppv)[0]) + ',' + str(
                np.shape(ppv)[1]) + \
            ',' + str(np.shape(ppv)[2]) + '), please wait for ~5 minutes. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(
                datetime.datetime.now()), args)
        time_temp = time.time()
        mydiag('Deb706: in ergs/s/A/pixel: for H2R cube', ppv, args)
        myprint('This done in %s minutes\n' % ((time.time() - time_temp) / 60), args)
        '''
    # ----to find a pixel which has non zero data and plot its spectrum--------#
    '''
    if args.debug and not args.hide:
        XX, YY = np.shape(ppv)[0], np.shape(ppv)[1]
        x, y = 0, 0
        dx = 0
        dy = -1
        for i in range(max(XX, YY) ** 2):
            if (-XX / 2 < x <= XX / 2) and (-YY / 2 < y <= YY / 2):
                if np.array(ppv[x + XX / 2, y + YY / 2, :]).any(): break
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
                dx, dy = -dy, dx
            x, y = x + dx, y + dy
        xx, yy = x + XX / 2, y + YY / 2

        fig = plt.figure(figsize=(15, 5))
        for ii in logbook.wlist:
            plt.axvline(ii, ymin=0.9, c='black')
        plt.ylabel('flam in erg/s/A')
        plt.xlabel('Wavelength (A)')
        plt.ylim(-0.2 * np.max(ppv[xx, yy, :]), 1.2 * np.max(ppv[xx, yy, :]))
        plt.xlim(logbook.wmin, logbook.wmax)
        plt.title('Spectrum along 1 LOS')
        plt.scatter(properties.dispsol, ppv[xx, yy, :], c='k', s=5,
                    label='H2R spec pix (' + str(xx) + ',' + str(yy) + ')')
    '''
    sizex, sizey = np.shape(ppv)[0], np.shape(ppv)[1]
    # -------------------------Now ideal PPV is ready: do whatever with it------------------------------------------------------------------
    if args.smooth:
        if os.path.exists(logbook.convolved_filename):  # read it in if the convolved cube already exists
            myprint('Reading existing convolved cube from ' + logbook.convolved_filename + '\n', args)
        else:  # make the convolved cube and save it if it doesn't exist already
            if args.debug: print 'Deb718: shape before rebinning = ', np.shape(ppv)
            myprint('Trying to parallely rebin AND convolve with ' + str(args.ncores) + ' core...\n', args)  #
            myprint(
                'Using ' + args.ker + ' kernel.\nUsing parameter set: sigma= ' + str(logbook.sig) + ', size= ' + str(
                    logbook.size, ) + '\n', args)

            funcname = HOME + WD + 'parallel_convolve.py'
            if args.silent:
                silent = ' --silent'
            else:
                silent = ''
            if args.toscreen:
                toscreen = ' --toscreen'
            else:
                toscreen = ''
            if args.debug:
                debug = ' --debug'
            else:
                debug = ''
            if args.saveplot:
                saveplot = ' --saveplot'
            else:
                saveplot = ''
            if args.hide:
                hide = ' --hide'
            else:
                hide = ''
            if args.plot_1dkernel:
                plot_1dkernel = ' --plot_1dkernel'
            else:
                plot_1dkernel = ''
            if args.cmin is not None:
                cmin = ' --cmin ' + str(args.cmin)
            else:
                cmin = ''
            if args.cmax is not None:
                cmax = ' --cmax ' + str(args.cmax)
            else:
                cmax = ''
            if args.strehl is not None:
                strehl = ' --strehl ' + str(args.strehl)
            else:
                strehl = ''
            arcsec_per_pixel = args.res_arcsec/logbook.fwhm

            command = args.which_mpirun+' -np ' + str(args.ncores) + ' python ' + funcname + ' --fitsname ' + logbook.fitsname + \
                      ' --file ' + args.file + ' --sig ' + str(logbook.sig) + ' --pow ' + str(args.pow) + ' --size ' + str(
                logbook.size) + ' --ker ' + args.ker + ' --convolved_filename ' + \
                      logbook.convolved_filename + ' --outfile ' + args.outfile + ' --H2R_cubename ' + logbook.H2R_filename + \
                      ' --exptime ' + str(logbook.exptime) + ' --final_pix_size ' + str(
                logbook.final_pix_size) + ' --xcenter_offset ' + \
                      str(args.xcenter_offset) + ' --ycenter_offset ' + str(args.ycenter_offset) + ' --galsize ' + str(
                args.galsize) + ' --center '+str(args.center)+ ' --res '+str(args.res)+\
                      ' --intermediate_pix_size ' + str(logbook.intermediate_pix_size) + ' --arcsec_per_pixel '+str(arcsec_per_pixel)+\
                      ' --rad ' +str(args.rad) + strehl + silent + toscreen + debug + cmin + cmax + saveplot + hide + plot_1dkernel
            myprint(command+'\n', args)
            #status = os.system.(command)
            subprocess.call([command], shell=True)

        ppv = fits.open(logbook.convolved_filename)[0].data  # reading in convolved cube from file
        #print 'Deb1429: in ergs/s/A: for convolved cube', 'shape=', np.shape(ppv), 'mean=', np.mean(masked_data(ppv)) ##
        '''
        if args.debug:
            myprint('Trying to calculate some statistics on the cube, please wait...', args)
            mydiag('Deb737: in ergs/s/A: for convolved cube', ppv, args)
            if not args.hide:
                xx, yy = xx * np.shape(ppv)[0] / sizex, yy * np.shape(ppv)[1] / sizey
                plt.plot(properties.dispsol, ppv[xx, yy, :] * (logbook.final_pix_size * 1e3) ** 2, c='g',label='convolved spec pix (' + str(xx) + ',' + str(yy) + ')')
        '''
    if not args.maketheory:
        last_cubename = logbook.convolved_filename if args.smooth else logbook.H2R_filename
        funcname = HOME + WD + 'parallel_makeobs.py'
        if args.silent:
            silent = ' --silent'
        else:
            silent = ''
        if args.toscreen:
            toscreen = ' --toscreen'
        else:
            toscreen = ''
        if args.debug:
            debug = ' --debug'
        else:
            debug = ''
        if args.addnoise:
            noise = ' --addnoise'
        else:
            noise = ''
        if args.noskynoise:
            skynoise = ' --noskynoise'
        else:
            skynoise = ''
        if args.spec_smear:
            smear = ' --spec_smear '
        else:
            smear = ''
        if args.saveplot:
            saveplot = ' --saveplot'
        else:
            saveplot = ''
        if args.hide:
            hide = ' --hide'
        else:
            hide = ''
        if args.fixed_SNR is not None and args.addnoise:
            fixed_SNR = ' --fixed_SNR ' + str(args.fixed_SNR)
        else:
            fixed_SNR = ''
        if args.addnoise and not os.path.exists(logbook.skynoise_cubename):
            myprint('Computing skynoise cube..\n', args)
            properties.skynoise = getskynoise(args, logbook, properties)
            write_fits(logbook.skynoise_cubename, properties.skynoise, args, fill_val=np.nan)
        if args.fixed_noise is not None and args.addnoise:
            myprint('Preparing for implementing fixed noise model..\n', args)
            fixed_noise = ' --fixed_noise ' + str(args.fixed_noise)
            line, dr = 'NII6584', 0.1
            nonoise_line_map = get_no_noise_map(logbook, args, line)  # to scale noise to the mean NII6584 flux at r_scale
            nonoise_line_map_annulus = get_pixels_within(nonoise_line_map, args, logbook, annulus=True, dr=dr)
            flux_rscale = np.mean(nonoise_line_map_annulus)
            # flux_rscale = np.mean([np.percentile(nonoise_line_map_annulus, ind) for ind in np.linspace(50*0.9,50*1.1,10)])
            myprint(
                'Mean ' + line + ' flux for no noise case, at ' + str(dr * 100) + '% annuli of scale_length = ' + str(
                    flux_rscale) + ' ergs/s/cm^2/arcsec^2\n', args)
            flux_to_scale = ' --flux_to_scale ' + str(flux_rscale)
            if os.path.exists(logbook.skynoise_cubename):
                properties.skynoise = fits.open(logbook.skynoise_cubename)[0].data
                myprint('Reading existing skynoise cube from ' + logbook.skynoise_cubename + '\n', args)
            else:
                myprint('Computing skynoise cube..\n', args)
                properties.skynoise = getskynoise(args, logbook, properties)
                write_fits(logbook.skynoise_cubename, properties.skynoise, args, fill_val=np.nan)
            N_sky = get_sky_norm(properties, logbook, args)
            myprint('Sky noise normalisation in R-band = ' + str(N_sky) + '\n', args)
            sky_norm = ' --sky_norm ' + str(N_sky)
        else:
            fixed_noise = ''
            flux_to_scale = ''
            sky_norm = ''
        if args.cmin is not None:
            cmin = ' --cmin ' + str(args.cmin)
        else:
            cmin = ''
        if args.cmax is not None:
            cmax = ' --cmax ' + str(args.cmax)
        else:
            cmax = ''
        if args.snr_cmin is not None:
            snr_cmin = ' --cmin ' + str(args.snr_cmin)
        else:
            snr_cmin = ''
        if args.snr_cmax is not None:
            snr_cmax = ' --cmax ' + str(args.snr_cmax)
        else:
            snr_cmax = ''
        if args.scale_length is not None:
            scale_length = ' --scale_length ' + str(args.scale_length)
        else:
            scale_length = ''
        parallel = args.which_mpirun + ' -np ' + str(args.ncores) + ' python ' + funcname + ' --parallel'
        series = 'python ' + funcname
        series_or_parallel = parallel  # USE parallel OR series

        command = series_or_parallel + ' --fitsname ' + logbook.fitsname + ' --fitsname_u ' + logbook.fitsname_u + ' --outfile ' + args.outfile + ' --last_cubename ' + \
                  last_cubename + ' --nbin ' + str(args.nbin) + ' --nres ' + str(args.nres) + ' --vdel ' + str(
            args.vdel) + ' --vdisp ' + str(args.vdisp) + ' --vres ' + str(args.vres) + \
                  ' --nhr ' + str(args.nhr) + ' --wmin ' + str(logbook.wmin) + ' --wmax ' + str(
            logbook.wmax) + ' --epp ' + str(args.el_per_phot) + ' --center '+str(args.center) + \
                  ' --gain ' + str(args.gain) + ' --exptime ' + str(logbook.exptime) + ' --final_pix_size ' + \
                  str(logbook.final_pix_size) + ' --flux_ratio ' + str(properties.flux_ratio) + ' --dist ' + str(
            properties.dist) + ' --skynoise_cubename ' + logbook.skynoise_cubename + ' --galsize ' + str(args.galsize) + \
                  ' --rad ' + str(args.rad) + ' --multi_realisation ' + str(
            args.multi_realisation) + ' --xcenter_offset ' + \
                  str(args.xcenter_offset) + ' --ycenter_offset ' + str(
            args.ycenter_offset) + fixed_SNR + fixed_noise + smear + noise + skynoise + silent + toscreen + debug + \
                  cmin + cmax + saveplot + hide + scale_length + flux_to_scale + sky_norm + snr_cmin + snr_cmax
        myprint(command + '\n', args)
        subprocess.call([command], shell=True)
    else:
        write_fits(logbook.fitsname, ppv, args,
                   fill_val=np.nan)  # writing the last cube itself as the ppv cube, if asked to make theoretical cube

    ppv = fits.open(logbook.fitsname)[0].data  # reading in ppv cube from file
    #print 'Deb1548: ergs/s/pc^2/A: for final PPV cube', 'shape=', np.shape(ppv), 'mean=', np.mean(masked_data(ppv)) ##

    if args.debug and not args.hide:
        xx, yy = xx * np.shape(ppv)[0] / sizex, yy * np.shape(ppv)[1] / sizey
        plt.plot(properties.dispsol, ppv[xx, yy, :] * (logbook.final_pix_size * 1e3) ** 2, c='r',
                 label='obs spec pix (' + str(xx) + ',' + str(yy) + ')')
        plt.legend()
        plt.show(block=False)

    if args.debug:
        myprint(
            'Trying to calculate some statistics on the cube, please wait for ~10 minutes. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(
                datetime.datetime.now()), args)
        time_temp = time.time()
        mydiag('Deb739: in ergs/s/pc^2/A: for final PPV cube', ppv, args)
        myprint('This done in %s minutes\n' % ((time.time() - time_temp) / 60), args)

    myprint('Final pixel size on target frame = ' + str(logbook.final_pix_size) + ' kpc' + ' and shape = (' + str(
        np.shape(ppv)[0]) + ',' + str(np.shape(ppv)[1]) + ',' + str(np.shape(ppv)[2]) + ') \n', args)
    # -------------------------Now realistic (smoothed, noisy) PPV is ready------------------------------------------------------------------
    # if not args.hide: spec_total(properties.dispsol, ppv, 'Spectrum for total', args, logbook)
    myprint('Returning PPV as variable "ppvcube"' + '\n', args)
    return np.array(ppv)


# -------------------------------------------------------------------------------------------
def makeobservable(cube, args, logbook, properties, core_start, core_end, prefix):
    nslice = np.shape(cube)[2]
    new_cube = np.zeros(np.shape(cube))
    new_cube_u = np.zeros(np.shape(cube))

    for k in range(core_start, core_end+1):
        map = cube[:, :, k]

        map_u = new_cube_u[:, :, k]
        factor = logbook.exptime * args.el_per_phot * properties.delta_lambda[k] / (args.gain * planck * (c * 1e3) / (
                properties.dispsol[k] * 1e-10)) * (args.rad / (2 * properties.dist * 3.08e21))**2  # to bring ergs/s/A/pixel to units of counts/pixel (ADUs)
        # factor = 1. #
        if args.debug:
            mydiag(prefix + 'Deb 754: in ergs/s/A/pixel: before factor', map, args)
            myprint(
                prefix + 'Deb1434: Factor = exptime= %.4E * el_per_phot= %.1F / (gain= %.2F * planck= %.4E * c= %.4E / lambda=%.2E)' % (
                logbook.exptime, \
                args.el_per_phot, args.gain, planck, c * 1e3, properties.dispsol[k] * 1e-10), args)
            myprint(prefix + 'Deb1436: Factor= %.4E\n' % factor, args)
        map *= factor  # to get in counts/pixel from ergs/s/A/pixel
        if args.debug:
            mydiag(prefix + 'Deb 756: in counts/pixel: after multiplying by factor', map, args)
        if args.addnoise:
            skynoiseslice = properties.skynoise[k] if properties.skynoise is not None else None
            logbook.resoln = c / args.vres if args.spec_smear else c / args.vdisp
            map, map_u = makenoisy(map, args, logbook, properties, skynoiseslice=skynoiseslice, factor=args.gain,
                                   slice=k,
                                   prefix=prefix)  # factor=gain as it is already in counts (ADU), need to convert to electrons for Poisson statistics
            # map, map_u = makenoisy(map, args, logbook, properties, skynoiseslice=skynoiseslice, factor=1., slice=k, prefix=prefix) #factor=gain as it is already in counts (ADU), need to convert to electrons for Poisson statistics
        if args.debug: myprint(
            prefix + 'Deb898: # of non zero elements in map before clipping= ' + str(len(map.nonzero()[0])), args)
        #map = np.ma.masked_where(map < 1, map) #clip all that have less than 1 count
        #map = np.ma.masked_where(map > 1e5, map)  # clip all that have more than 100,000 count i.e. saturating
        if args.debug: myprint(
            prefix + 'Deb898: # of non zero elements in map after clipping= ' + str(len(map.nonzero()[0])), args)
        map /= factor  # convert back to ergs/s/A/pixel from counts/pixel
        map_u /= factor  # convert back to ergs/s/A/pixel from counts/pixel
        if args.debug: mydiag(prefix + 'Deb 762: in ergs/s/A/pixel: after dividing by factor', map, args)
        map *= properties.flux_ratio # convert to ergs/s/A/cm^2/arcsec^2 from ergs/s/pix/A
        map_u *= properties.flux_ratio # convert to ergs/s/A/cm^2/arcsec^2 from ergs/s/pix/A
        if args.debug: mydiag(prefix + 'Deb 764: in ergs/s/A/cm^2/arcsec^2: after dividing by pixel area', map, args)
        new_cube[:, :, k] = map
        new_cube_u[:, :, k] = map_u
    return new_cube, new_cube_u


# ----------------to add noise---------------------------------------------------------------------------
def makenoisy(data, args, logbook, properties, skynoiseslice=None, factor=1., slice=None, prefix=None):
    data_copy = copy.copy(data)
    slice_slab = properties.nwbin / 5
    percentile = 90.  # to check 90% of the SNR map within 4kpc of galactic center
    np.random.seed(args.multi_realisation)
    if args.debug:
        if slice is not None and slice % slice_slab == 0 and not args.hide: dummy = plotmap(data_copy, 'slice ' + str(
            slice) + ': before adding any noise', 'junk', 'counts', args, logbook, makelog=False)
        mydiag(prefix + 'Deb 767: in counts/pixel: before adding any noise', data_copy, args)
    size = args.galsize / np.shape(data)[0]
    data *= factor  # to transform into electrons/pixel from physical units/pixel
    if args.debug: mydiag(prefix + 'Deb 771: in el/pixel: after mutiplying gain factor, before adding noise', data,
                          args)
    if args.fixed_SNR is not None:  # adding only fixed amount of SNR to ALL spaxels
        epsilon = 1e-3
        noise = np.random.normal(loc=0., scale=np.abs((data + epsilon) / args.fixed_SNR), size=np.shape(
            data))  # drawing from normal distribution about a mean value of 0 and width =counts/SNR
        # adding epsilon otherwise cant add errors with zero width (=scale), for cells that have no counts
        noisydata = data + noise
        if args.debug:
            mydiag(prefix + 'Deb 775: in el/pixel: after fixed_SNR ' + str(args.fixed_SNR) + ' noise', noisydata, args)
            snr_map = data / noise
            snr_map = np.ma.masked_where(data <= 0, snr_map)
            if slice is not None and slice % slice_slab == 0 and not args.hide:
                dummy = plotmap(noisydata,
                                'slice ' + str(slice) + ': after fixed_SNR ' + str(args.fixed_SNR) + ' noise', 'junk',
                                'counts', args, logbook, makelog=False)
                dummy = plotmap(snr_map, 'slice ' + str(slice) + ': SNR map', 'junk', 'ratio', args, logbook,
                                makelog=False)
    elif args.fixed_noise is not None:  # adding same (but lambda dependent) noise to all spatial pixels
        skynoiseslice = np.array(skynoiseslice) # already in units of ergs/s/A/cm^2/arcsec^2
        skynoiseslice[skynoiseslice <= 0.] = 1e-9  # extremely small to avoid 0
        #skynoiseslice /= properties.delta_lambda[slice]  # to bring skynoise from units of el/s/spaxel to el/s/spaxel/A
        skycontrib = skynoiseslice / args.sky_norm  # sky_norm is already in units of ergs/s/A/cm^2/arcsec^2; so skycontrib is now dimensionless

        try:
            lambda_NII = logbook.wlist[logbook.llist == 'NII6584'][0]
            delta_lambda_NII = properties.delta_lambda[np.where(properties.dispsol >= lambda_NII)[0][0]]
        except:
            lambda_NII = 6585.273
            delta_lambda_NII = 0.6585343937049402
        npix_NII = 4. #number of points the NII line is resolved with (during line fitting)
        fixednoise = skycontrib * (args.flux_to_scale / (args.fixed_noise *
            npix_NII * delta_lambda_NII))  # args.flux_to_scale is in ergs/s/cm^2/arcsec^2 units; now fixednoise becomes in ergs/s/A/cm^2/arcsec^2 units
        if args.debug: myprint(prefix + 'Deb1704: fixednoise (in ergs/s/A/cm^2/arcsec^2) = skycontrib= %.2F * flux_to_scale (ergs/s/cm^2/arcsec^2) = %.3E / (SNR= %.1F * sqrt(npix_NII= %.2F) * delta_lambda_NII (in A)= %.2F) = %.3E'
                %(skycontrib, args.flux_to_scale, args.fixed_noise, npix_NII, delta_lambda_NII, fixednoise), args)
        fixednoise_dummy = fixednoise # for printing  purposes
        fixednoise *= logbook.exptime * args.el_per_phot * properties.delta_lambda[slice] * np.pi * (args.rad * 1e2 * logbook.final_pix_size*180*3600/(np.pi * properties.dist * 1e3))**2 / (planck * (c * 1e3) / (properties.dispsol[slice] * 1e-10))  # convert from ergs/s/cm^2/A/arcsec^2 units to el/pixel; properties.flux_ratio has units of cm^-2 arcsec^-2
        if args.debug:
            myprint(prefix + 'Deb1704: fixednoise (in el/pix) = fixednoise (in ergs/s/A/cm^2/arcsec^2) = %.3E * exptime (in s) = %.4E * \
        el_per_phot= %.1F * delta_lambda (in A) = %.2F * pi * (telescope diameter (in cm) = %.2F)^2 * (arcsec = %.2F)^2/ (planck (in ergs.sec) = %.4E * c (in m/s) = %.4E / lambda (in m) =%.2E) = %.3E'
                %(fixednoise_dummy, logbook.exptime, args.el_per_phot, properties.delta_lambda[slice], args.rad*1e2, logbook.final_pix_size*180*3600/(np.pi * properties.dist * 1e3), planck, c * 1e3, \
                  properties.dispsol[slice] * 1e-10, fixednoise), args)
            myprint(prefix + 'Deb 1707: in el/pixel: the noise value to be added (to be drawn from) = %.3F'%fixednoise, args)
        sigma_map = np.ones(np.shape(map)) * fixednoise  # 1 sigma uncertainty at every pixel, will be spatially uniform in this case
        noise = np.random.poisson(lam=fixednoise ** 2, size=np.shape(data)) - fixednoise ** 2
        noisydata = data + noise  # data and noise both in el/pixel units

        if args.debug:
            if slice is not None and slice % slice_slab == 0 and not args.hide:
                dummy = plotmap(data, 'slice ' + str(slice) + ': before fixed noise', 'junk', 'el/pixel', args,
                                logbook, makelog=False)
                dummy = plotmap(noisydata, 'slice ' + str(slice) + ': after fixed noise', 'junk', 'el/pixel', args,
                                logbook, makelog=False)
            mydiag(prefix + 'Deb 781: in el/pixel: with skycontrib = ' + str(
                skycontrib) + ' and only fixed noise of ' + str(fixednoise), noise, args)
            mydiag(prefix + 'Deb 782: in el/pixel: after adding fixed noise', noisydata, args)
    else:
        # ------adding poisson noise-----------
        noisydata = np.random.poisson(lam=data, size=None)  # adding poisson noise to counts (electrons)
        noisydata = noisydata.astype(float)
        poissonnoise = noisydata - data_copy
        if args.debug:
            if slice is not None and slice % slice_slab == 0 and not args.hide: dummy = plotmap(noisydata,
                                                                              'slice ' + str(slice) + ': after poisson',
                                                                              'junk', 'counts', args, logbook,
                                                                              makelog=False)
            mydiag(prefix + 'Deb 783: in el/pixel: after adding poisson noise', noisydata, args)
        # ------adding readnoise-----------
        '''
        readnoise = np.random.normal(loc=0., scale=3.5, size=np.shape(noisydata)) #to draw gaussian random variables from distribution N(0,3.5) where 3.5 is width in electrons per pixel
                                    #multiply sqrt(14) to account for the fact that for SAMI each spectral fibre is 2 pix wide and there are 7 CCD frames for each obsv
        if args.debug: mydiag(prefix+'Deb 781: in el/pixel: only RDNoise', readnoise, args)
        noisydata += readnoise #adding readnoise
        if args.debug:
            if slice is not None and slice%slice_slab == 0: dummy = plotmap(noisydata, 'slice '+str(slice)+': after readnoise', 'junk', 'counts', args, logbook, makelog=False)
            mydiag(prefix+'Deb 783: in el/pixel: after adding readnoise', noisydata, args)
        '''
        readnoise = np.zeros(np.shape(noisydata))  # not adding readnoise bcz skynoise includes it already
        # ------adding sky noise-----------
        if skynoiseslice is not None and skynoiseslice != 0:
            skynoise = np.random.normal(loc=0., scale=np.abs(skynoiseslice), size=np.shape(
                noisydata))  # drawing from normal distribution about a sky noise value at that particular wavelength
            skynoise *= logbook.exptime * (
                    ((3600 * 180. / np.pi) * logbook.final_pix_size / (properties.dist * 1e3)) ** 2 / (
            0.5) ** 2)  # converting skynoise from el/s/spaxel to el/pixel; each spaxel = 0.5"x0.5" FoV (of sky) for EACH pixel for SAMI noise cube
            if args.debug: mydiag(prefix + 'Deb 790: in el/pixel: only skynoise+RDnoise', skynoise, args)
            noisydata += skynoise  # adding sky noise
            if slice is not None and slice % slice_slab == 0 and not args.hide:
                dummy = plotmap(noisydata, 'slice ' + str(slice) + ': after skynoise', 'junk', 'counts', args, logbook,
                                makelog=False)
        else:
            skynoise = np.zeros(np.shape(noisydata))
        # ------finished adding noise-----------

    noisydata /= factor  # converting back to physical units from electrons/pixel
    totalnoise = noisydata - data_copy
    if args.debug:
        snr_map = data_copy / totalnoise
        snr_map = np.ma.masked_where((data_copy <= 0) | (totalnoise <= 0), snr_map)
        #try:
        snr_frac = np.percentile(get_pixels_within(snr_map, args, logbook), 100 - percentile)
        snr_list = get_pixels_within(snr_map, args, logbook, annulus=True, dr=0.1)
        snr_list = np.ma.masked_where(~np.isfinite(snr_list), snr_list)
        mydiag(prefix + 'Deb 797: in ratio-units: SNR map', snr_map, args)
        myprint(prefix + 'Deb797.5: in ratio-units: mean SNR in annulus = %.2F'%np.mean(snr_list), args)
        myprint(prefix + 'Deb798: makenoisy: Net effect of adding noise:\n', args)
        mydiag(prefix + 'Deb 800: in counts/pixel: total noise', totalnoise, args)
        mydiag(prefix + 'Deb 803: in counts/pixel: after adding noise dividing gain factor', noisydata, args)
        '''
        except:
            snr_frac = 0.
        '''
        myprint(prefix + 'Deb 796: ' + str(percentile) + '% pixels within ' + str(
            args.scale_length) + 'kpc are above SNR=' + str(snr_frac), args)
        if args.scale_exp_SNR:
            goodfrac = len(snr_map > float(args.scale_exp_SNR)) / np.ma.count(snr_map)
            myprint(prefix + ' fraction of pixels above SNR ' + str(args.scale_exp_SNR) + ' = ' + str(goodfrac), args)
        if slice is not None and slice % slice_slab == 0 and not args.hide:
            dummy = plotmap(snr_map, 'slice ' + str(slice) + ': SNR map', 'junk', 'ratio', args, logbook, makelog=False,
                            addcircle_radius=args.scale_length, issnrmap=True)
            myprint('Pausing for 1...', args)
            plt.pause(1)
    return noisydata, sigma_map


# ----to return pixels within a certain readius or annulus--------
def get_pixels_within(map, args, logbook, radius=None, annulus=False, dr=0.1):
    if radius is None: radius = args.scale_length
    rows, cols = np.shape(map)
    d_list = [np.sqrt((x * logbook.final_pix_size - args.galsize/2. - args.xcenter_offset) ** 2 + (
            y * logbook.final_pix_size - args.galsize/2. - args.ycenter_offset) ** 2) for x in xrange(rows) for y in
              xrange(cols)]  # kpc
    list = np.array([x for (y, x) in sorted(zip(d_list, map.flatten()), key=lambda pair: pair[0])])
    d_list = np.sort(d_list)
    if annulus:
        list = list[
            np.abs(d_list - radius) / radius <= dr]  # consider only within annulus of "dr" fraction of scale length
    else:
        list = list[d_list <= radius]  # consider only within scale length of galaxy; default 4 kpc
    return list


# -----------------to inspect a given slice after adding noise--------------------------------------------------------------------------
def inspect_noiseslice(slice, args, logbook, nres=20):
    test = ap.Namespace()
    test.slice = slice
    test.nres = nres

    nonoise_cube = fits.open(logbook.no_noise_fitsname)[0].data
    wnoise_cube = fits.open(logbook.fitsname)[0].data
    test.nonoise_cube = np.nan_to_num(nonoise_cube)
    test.wnoise_cube = np.nan_to_num(wnoise_cube)
    test.noise_cube = test.wnoise_cube - test.nonoise_cube

    test.nonoise_slice = test.nonoise_cube[:, :, test.slice]
    test.wnoise_slice = test.wnoise_cube[:, :, test.slice]
    test.noise_slice = test.noise_cube[:, :, test.slice]

    args.cmin, args.cmax = None, None  # -25, -17
    args.hide = False
    args.met = True  # to prevent next plotmap() call from masking negative values
    dummy = plotmap(test.nonoise_slice, 'slice:' + str(test.slice) + ' nonoise', 'junk', 'Log (erg/s/A/cm^2/arcsec^2)', args,
                    logbook)
    dummy = plotmap(test.wnoise_slice, 'slice:' + str(test.slice) + ' wnoise', 'junk', 'Log (erg/s/A/cm^2/arcsec^2)', args,
                    logbook)
    dummy = plotmap(test.noise_slice, 'slice:' + str(test.slice) + ' noise', 'junk', 'Log (erg/s/A/cm^2/arcsec^2)', args,
                    logbook)
    dummy = plotmap(test.nonoise_slice / test.noise_slice, 'slice:' + str(test.slice) + ' no-noise/noise', 'junk',
                    'Ratio', \
                    args, logbook, makelog=False, addcircle_radius=args.scale_length, issnrmap=True)

    test.nonoise_list = get_pixels_within(test.nonoise_slice, args, logbook, annulus=True)
    test.wnoise_list = get_pixels_within(test.wnoise_slice, args, logbook, annulus=True)
    test.noise_list = get_pixels_within(test.noise_slice, args, logbook, annulus=True)

    plt.figure()
    test.noise_hist = plt.hist(test.noise_list, histtype='step', bins=100, color='k', label='noise')
    test.nonoise_hist = plt.hist(test.nonoise_list, histtype='step', bins=100, color='r', label='nonoise')
    test.wnoise_hist = plt.hist(test.wnoise_list, histtype='step', bins=100, color='b', label='wnoise')
    plt.title('test: slice' + str(test.slice))
    plt.xlabel('flam (erg/s/A/pc^2)')
    plt.ylabel('Frequency')
    plt.legend(loc='best')
    plt.show(block=False)

    test.nonoise_map = np.sum((test.nonoise_cube * properties.delta_lambda)[:, :,
                              test.slice - int(test.nres) / 2: test.slice + int(test.nres) / 2], axis=2)
    test.wnoise_map = np.sum((test.wnoise_cube * properties.delta_lambda)[:, :,
                             test.slice - int(test.nres) / 2: test.slice + int(test.nres) / 2], axis=2)
    test.noise_map = test.wnoise_map - test.nonoise_map

    dummy = plotmap(test.nonoise_map, 'integrated map:' + str(slice) + ' nonoise', 'junk', 'Log (erg/s/cm^2/arcsec^2)', args,
                    logbook)
    dummy = plotmap(test.wnoise_map, 'integrated map:' + str(slice) + ' wnoise', 'junk', 'Log (erg/s/cm^2/arcsec^2)', args,
                    logbook)
    dummy = plotmap(test.noise_map, 'integrated map:' + str(slice) + ' noise', 'junk', 'Log (erg/s/cm^2/arcsec^2)', args,
                    logbook)

    test.nonoise_map_list = get_pixels_within(test.nonoise_map, args, logbook, annulus=True)
    test.wnoise_map_list = get_pixels_within(test.wnoise_map, args, logbook, annulus=True)
    test.noise_map_list = get_pixels_within(test.noise_map, args, logbook, annulus=True)

    print 'wnoise/noise maps', np.mean(test.wnoise_map_list / test.noise_map_list), 'nonoise/noise maps', np.mean(
        test.nonoise_map_list / test.noise_map_list)

    return test


# -----------------to get corresponding no-noise line map, used for scaling while adding noise--------------------------------------------------------------------------
def get_no_noise_map(logbook, args, line):
    if os.path.exists(logbook.no_noise_fittedcube):
        myprint('Reading existing ' + logbook.no_noise_fittedcube + '\n', args)
    else:
        myprint(
            'Could not find corresponding no-noise version of mapcube at: ' + logbook.no_noise_fittedcube + '. So creating one...\n',
            args)
        args2 = deepcopy(args)
        args2.addnoise = False
        args2.fixed_noise = None
        args2.debug = False
        args2.multi_realisation = 1
        args2.wmin = 6400.0
        args2.wmax = None
        args2.branch = None

        properties2 = deepcopy(properties)
        args2, logbook2 = getfitsname(args2, properties2)
        logbook2.s = ascii.read(getfn(args2), comment='#', guess=False)
        args2.outfile = logbook2.fitsname.replace('PPV', 'output_PPV')[:-5] + '.txt'

        if logbook2.fittedcube != logbook.no_noise_fittedcube: myexit(args, text='Line 1876 in \
plotobservables_old.get_no_noise_map(), "no_noise_mapcube_path" and "logbook2.fittedcube" do not match.\n \
logbook2.fittedcube = ' + logbook2.fittedcube + '\n\
logbook.no_noise_fittedcube = ' + logbook.no_noise_fittedcube + '\nAborting.')

        if not args2.silent: myprint('Will be using/creating ' + logbook2.fitsname + ' file.' + '\n', args2)
        if os.path.exists(logbook2.fitsname):
            if not args2.silent: myprint('Reading existing ppvcube from ' + logbook2.fitsname + '\n', args2)
            properties2.ppvcube = fits.open(logbook2.fitsname)[0].data
        else:
            if not args2.silent: myprint('PPV file does not exist. Creating ppvcube..' + '\n', args2)
            properties2.ppvcube = spec(args2, logbook2, properties2)

        if args2.spec_smear:
            smear = ' --spec_smear'
        else:
            smear = ''
        if args2.silent:
            silent = ' --silent'
        else:
            silent = ''
        if args2.toscreen:
            toscreen = ' --toscreen'
        else:
            toscreen = ''
        if args2.debug:
            debug = ' --debug'
        else:
            debug = ''
        if args2.showfit:
            showfit = ' --showfit'
        else:
            showfit = ''
        if args2.oneHII is not None:
            oneHII = ' --oneHII ' + str(args2.oneHII)
        else:
            oneHII = ''
        if args2.addnoise:
            addnoise = ' --addnoise'
        else:
            addnoise = ''
        if args2.contsub:
            contsub = ' --contsub'
        else:
            contsub = ''

        funcname = HOME + WD + 'parallel_fitting.py'
        command = args2.which_mpirun + ' -np ' + str(
            args2.ncores) + ' python ' + funcname + ' --fitsname ' + logbook2.fitsname + \
                  ' --no_noise_fitsname ' + logbook2.no_noise_fitsname + ' --fitsname_u ' + logbook2.fitsname_u + ' --nbin ' + str(
            args2.nbin) + \
                  ' --vdel ' + str(args2.vdel) + ' --vdisp ' + str(args2.vdisp) + ' --vres ' + str(
            args2.vres) + ' --nhr ' + str(args2.nhr) + ' --wmin ' + \
                  str(logbook2.wmin) + ' --wmax ' + str(
            logbook2.wmax) + ' --fittedcube ' + logbook2.fittedcube + ' --fittederror ' + logbook2.fittederror + \
                  ' --outfile ' + args2.outfile + ' --nres ' + str(args2.nres) + ' --vmask ' + str(
            args2.vmask) + smear + silent + toscreen \
                  + debug + showfit + oneHII + addnoise + contsub
        subprocess.call([command], shell=True)

    no_noise_mapcube = fits.open(logbook.no_noise_fittedcube)[0].data
    if args.branch == 'blue':
        wlist_red, llist_red = readlist()
        wmin_red = 6400.
        wmax_red = wlist_red[-1] + 50.
        llist_red = llist_red[np.where(np.logical_and(wlist_red > wmin_red, wlist_red < wmax_red))]  # truncate linelist as per wavelength range
        #wlist_red = wlist_red[np.where(np.logical_and(wlist_red > wmin_red, wlist_red < wmax_red))]
        line_list = llist_red
    else:
        line_list = logbook.llist
    no_noise_line_map = np.array(no_noise_mapcube)[:, :, np.where(line_list == line)[0][0]]
    return no_noise_line_map


# -------------------------------------------------------------------------------------------
def inspectmap(args, logbook, properties, axes=None, additional_text=''):
    g, x, y = calcpos(logbook.s, args.center, args.galsize, args.res) # x and y range from (-galsize/2, galsize/2) kpc centering at 0 kpc
    t = args.file + ':Met_Om' + str(args.Om)
    if args.smooth: t += '_arc' + str(args.res_arcsec) + '"'
    if args.spec_smear: t += '_vres=' + str(args.vres) + 'kmps_'
    t += info + args.gradtext
    if args.SNR_thresh is not None: t += '_snr' + str(args.SNR_thresh)
    if args.useKD:
        t += '_KD02'
    else:
        t += '_D16'
    if args.emulate_henry:
        t += '_target_res'+str(args.target_res)+'pc_'+ additional_text

    nnum, nden = len(np.unique(args.num_arr)), len(np.unique(args.den_arr))
    nplots = nnum + nden + len(args.num_arr) + 1
    nrow, ncol = int(np.ceil(float(nplots / 3))), min(nplots, 3)
    marker_arr = ['s', '^']
    met_maxlim, alpha = 0.5, 0.1

    if axes is None:
        fig, axes = plt.subplots(nrow, ncol, sharex=True, figsize=(14, 8))
        fig.subplots_adjust(hspace=0.1, wspace=0.25, top=0.9, bottom=0.1, left=0.07, right=0.98)
    else:
        fig = plt.gcf()
    # ----------plotting individual H2R---------------------------
    d = np.sqrt(
        (x - args.xcenter_offset) ** 2 + (y - args.ycenter_offset) ** 2)  # kpc
    plot_index, col, already_plotted = 0, 'r', []

    indiv_map_num, indiv_map_den = [], []
    for num_grp in args.num_arr:
        log_fluxmin, log_fluxmax = 0, -100
        ax = axes[plot_index / ncol][plot_index % ncol]
        indiv_map_num_grp = []
        if not iterable(num_grp): num_grp = [num_grp]
        if num_grp not in already_plotted:
            plot = True
            already_plotted.append(num_grp)
        else:
            plot = False
        for (jj, num) in enumerate(num_grp):
            temp = logbook.s[num] * properties.flux_ratio
            indiv_map_num_grp.append(temp)
            if plot:
                ax.scatter(d.flatten(), np.log10(temp.flatten()), s=5, lw=0, marker=marker_arr[jj], c=col, alpha=alpha,
                           label='pixel' if plot_index == 1 else None)
                if np.min(np.log10(np.ma.masked_where(temp <= 0, temp))) < log_fluxmin: log_fluxmin = np.min(
                    np.log10(np.ma.masked_where(temp <= 0, temp)))
                if np.max(np.log10(temp)) > log_fluxmax: log_fluxmax = np.max(np.log10(temp))
                myprint('all HIIR: ' + num + ' median= ' + str(np.median(temp)) + ', integrated= ' + str(np.sum(temp)),
                        args)
        # -----to fit radial profile of individual H2R fluxes of the last numerator-------
        if plot:
            linefit, linecov = np.polyfit(d.flatten(), np.log10(temp.flatten()), 1, cov=True)
            x_arr = np.arange(args.galsize/2)
            ax.plot(x_arr, np.poly1d(linefit)(x_arr), c='k')
            ax.text(args.galsize/2 -0.1, ax.get_ylim()[-1]-0.1, 'H2R '+num+' slope=%.4F, int=%.4F'%(linefit[0], linefit[1]), ha='right', va='top', color='k')

        indiv_map_num.append(indiv_map_num_grp)
        #ax.set_ylim(log_fluxmin, np.log10(30 * 10 ** log_fluxmax))
        ax.set_ylabel('log(' + ','.join(num_grp) + ')')
        if plot: plot_index += 1

    for den_grp in args.den_arr:
        log_fluxmin, log_fluxmax = 0, -100
        indiv_map_den_grp = []
        ax = axes[plot_index / ncol][plot_index % ncol]
        if not iterable(den_grp): den_grp = [den_grp]
        if den_grp not in already_plotted:
            plot = True
            already_plotted.append(den_grp)
        else:
            plot = False
        for (jj, den) in enumerate(den_grp):
            temp = logbook.s[den] * properties.flux_ratio
            indiv_map_den_grp.append(temp)
            if plot:
                ax.scatter(d.flatten(), np.log10(temp.flatten()), s=5, lw=0, marker=marker_arr[jj], c=col, alpha=alpha)
                if np.min(np.log10(np.ma.masked_where(temp <= 0, temp))) < log_fluxmin: log_fluxmin = np.min(
                    np.log10(np.ma.masked_where(temp <= 0, temp)))
                if np.max(np.log10(temp)) > log_fluxmax: log_fluxmax = np.max(np.log10(temp))
                myprint('all HIIR: ' + den + ' median= ' + str(np.median(temp)) + ', integrated= ' + str(np.sum(temp)),
                        args)
        # -----to fit radial profile of individual H2R fluxes of the last denominator-------
        if plot:
            linefit, linecov = np.polyfit(d.flatten(), np.log10(temp.flatten()), 1, cov=True)
            x_arr = np.arange(args.galsize / 2)
            ax.plot(x_arr, np.poly1d(linefit)(x_arr), c='k')
            ax.text(args.galsize/2 - 0.1, ax.get_ylim()[-1]-0.1, 'H2R '+den+' slope=%.4F, int=%.4F'%(linefit[0], linefit[1]), ha='right', va='top', color='k')

        indiv_map_den.append(indiv_map_den_grp)
        #ax.set_ylim(log_fluxmin, np.log10(30 * 10 ** log_fluxmax))
        ax.set_ylabel('log(' + ','.join(den_grp) + ')')
        if plot: plot_index += 1

    indiv_map_num_series = [np.ma.sum(indiv_map_num[ind], axis=0) for ind in range(len(args.num_arr))]
    indiv_map_den_series = [np.ma.sum(indiv_map_den[ind], axis=0) for ind in range(len(args.den_arr))]

    if args.useKD:
        logOHsol, logOHobj_indiv_map = get_KD02_met(indiv_map_num_series, indiv_map_den_series)
    else:
        logOHsol, logOHobj_indiv_map = get_D16_met(indiv_map_num_series, indiv_map_den_series)
    myprint('all HIIR: median max min Z/Zsol= ' + str(np.median(10 ** logOHobj_indiv_map)) + ' ' + str(
        np.max(10 ** logOHobj_indiv_map)) + \
            ' ' + str(np.min(10 ** logOHobj_indiv_map)), args)

    for ii in range(len(indiv_map_num_series)):
        ax = axes[plot_index / ncol][plot_index % ncol]
        ax.scatter(d.flatten(), np.divide(indiv_map_num_series[ii], indiv_map_den_series[ii]).flatten(), s=5, lw=0,
                   marker=marker_arr[jj], c=col, alpha=alpha)
        ax.set_ylabel('+'.join(args.num_arr[ii]) + '/' + '+'.join(args.den_arr[ii]))
        plot_index += 1

    ax = axes[plot_index / ncol][plot_index % ncol]
    ax.scatter(d.flatten(), logOHobj_indiv_map.flatten(), s=5, lw=0, c=col, alpha=alpha)
    linefit, linecov = np.polyfit(d.flatten(), logOHobj_indiv_map.flatten(), 1, cov=True)
    x_arr = np.arange(args.galsize/2)
    ax.plot(x_arr, np.poly1d(linefit)(x_arr), c=col, label='Indiv H2R gradient slope=' + str('%.4F' % linefit[0]))

    # -------------plotting binned maps------------------------------------
    g = np.shape(properties.ppvcube)[0]
    b = np.linspace(-g / 2 + 1, g / 2, g) * (args.galsize) / g  # in kpc
    d = np.sqrt((b[:, None] - args.xcenter_offset) ** 2 + (b - args.ycenter_offset) ** 2)  # kpc
    plot_index, col, already_plotted = 0, 'g', []

    binned_map_num, binned_map_den = [], []
    for num_grp in args.num_arr:
        ax = axes[plot_index / ncol][plot_index % ncol]
        binned_map_num_grp = []
        if not iterable(num_grp): num_grp = [num_grp]
        if num_grp not in already_plotted:
            plot = True
            already_plotted.append(num_grp)
        else:
            plot = False
        for (jj, num) in enumerate(num_grp):
            temp = make2Dmap(logbook.s[num], x, y, g, args.galsize / g) * properties.flux_ratio
            binned_map_num_grp.append(temp)
            if plot:
                if not args.nomap: plotmap(temp, num + ': H2R binned', 'binned', 'log flux(ergs/s)', args, logbook,
                                           makelog=True)
                ax.scatter(d.flatten(), np.log10(temp.flatten()), s=5, lw=0, marker=marker_arr[jj], c=col, alpha=alpha,
                           label='pixel' if plot_index == 1 else None)
                myprint('binned HIIR: ' + num + ' median= ' + str(np.median(temp)) + ', integrated= ' + str(np.sum(temp)),
                        args)
        binned_map_num.append(binned_map_num_grp)
        if plot:
            plot_index += 1

    for den_grp in args.den_arr:
        binned_map_den_grp = []
        ax = axes[plot_index / ncol][plot_index % ncol]
        if not iterable(den_grp): den_grp = [den_grp]
        if den_grp not in already_plotted:
            plot = True
            already_plotted.append(den_grp)
        else:
            plot = False
        for (jj, den) in enumerate(den_grp):
            temp = make2Dmap(logbook.s[den], x, y, g, args.galsize / g) * properties.flux_ratio
            binned_map_den_grp.append(temp)
            if plot:
                if not args.nomap: plotmap(temp, den + ': H2R binned', 'binned', 'log flux(ergs/s)', args, logbook,
                                           makelog=True)
                ax.scatter(d.flatten(), np.log10(temp.flatten()), s=5, lw=0, marker=marker_arr[jj], c=col, alpha=alpha)
                myprint(
                    'binned HIIR: ' + den + ' median= ' + str(np.median(temp)) + ', integrated= ' + str(np.sum(temp)),
                    args)
        binned_map_den.append(binned_map_den_grp)
        if plot:
            plot_index += 1

    binned_map_num_series = [np.ma.sum(binned_map_num[ind], axis=0) for ind in range(len(args.num_arr))]
    binned_map_den_series = [np.ma.sum(binned_map_den[ind], axis=0) for ind in range(len(args.den_arr))]

    if args.useKD:
        logOHsol, logOHobj_binned_map = get_KD02_met(binned_map_num_series, binned_map_den_series)
    else:
        logOHsol, logOHobj_binned_map = get_D16_met(binned_map_num_series, binned_map_den_series)
    myprint('binned HIIR: median Z/Zsol= ' + str(np.median(10 ** logOHobj_binned_map)), args)

    for ii in range(len(binned_map_num_series)):
        ax = axes[plot_index / ncol][plot_index % ncol]
        ax.scatter(d.flatten(), np.divide(binned_map_num_series[ii], binned_map_den_series[ii]).flatten(), s=5, lw=0,
                   marker=marker_arr[jj], c=col, alpha=alpha)
        ax.set_ylim(args.ratio_lim[ii])
        plot_index += 1

    ax = axes[plot_index / ncol][plot_index % ncol]
    ax.scatter(d.flatten(), logOHobj_binned_map.flatten(), s=5, lw=0, c=col, alpha=alpha)
    logOH_arr = logOHobj_binned_map.flatten()
    d_arr = d.flatten()
    d_arr = np.ma.masked_where(np.isnan(logOH_arr), d_arr)
    logOH_arr = np.ma.masked_where(np.isnan(logOH_arr), logOH_arr)
    logOH_arr = np.ma.compressed(logOH_arr)
    d_arr = np.ma.compressed(d_arr)
    linefit, linecov = np.polyfit(d_arr, logOH_arr, 1, cov=True)
    x_arr = np.arange(args.galsize/2)
    ax.plot(x_arr, np.poly1d(linefit)(x_arr), c=col, label='Binned H2R gradient slope=' + str('%.4F' % linefit[0]))

    if not args.met:
        ax.set_ylabel(r'$\log{(Z/Z_{\bigodot})}$')
        ax.set_ylim(-2, met_maxlim)
        ax.set_xlim(0, args.galsize/2)
    ax.legend(fontsize=8, loc='lower left')
    fig.text(0.5, 0.03, 'Galactocentric distance (kpc)', ha='center')
    fig.text(0.5, 0.95, t, ha='center')
    if args.saveplot:
        fig.savefig(args.path + 'inspectmap_' + t + '.eps')
        myprint('Saved file ' + args.path + 'inspectmap_' + t + '.eps', args)
    if args.getmap: dummy = plotmap(logOHobj_binned_map, t + '_binned', 'Metallicity', 'log(Z/Z_sol)', args, logbook,
                                    makelog=False)

    return


# -------------------------------------------------------------------------------------------
def fixfit(args, logbook, properties):
    fs = 15 # label fontsize
    if args.mergespec is None:
        mergespec = [args.branch if args.branch is not None else '', str(logbook.wmin), str(logbook.wmax)]
    else:
        mergespec = args.mergespec.split(',')
    nbranches = len(mergespec) / 3 
    for kk in range(nbranches):
        args_dummy = deepcopy(args)
        args_dummy.branch = mergespec[3 * kk]
        args_dummy.wmin = float(mergespec[3 * kk + 1])
        args_dummy.wmax = float(mergespec[3 * kk + 2])
        #if args_dummy.branch == 'blue': args_dummy.nbin = 300 # for blue arm
        args_dummy, logbook_dummy = getfitsname(args_dummy, properties)
        properties_dummy = ap.Namespace()
        properties_dummy.ppvcube = fits.open(logbook_dummy.fitsname)[0].data
        properties_dummy.ppvcube_u = fits.open(logbook_dummy.fitsname_u)[0].data
        properties_dummy = get_disp_array(args_dummy, logbook_dummy, properties_dummy)
         
        if not np.array(properties_dummy.ppvcube[args_dummy.X, args_dummy.Y, :]).any():
            myprint('Chosen spaxel is empty. Select another.', args_dummy)
            myprint('Non empty spaxels are:', args_dummy)
            for i in range(np.shape(properties_dummy.ppvcube)[0]):
                for j in range(np.shape(properties_dummy.ppvcube)[1]):
                    if np.array(properties_dummy.ppvcube[i, j, :]).any():
                        print i, j

            myexit(args_dummy, text='Try again.')

        args_dummy.toscreen = True
        logbook_dummy.resoln = c / args_dummy.vres if args_dummy.spec_smear else c / args_dummy.vdisp
        wave = np.array(properties_dummy.dispsol)  # converting to numpy arrays
        flam = np.array(properties_dummy.ppvcube[args_dummy.X, args_dummy.Y, :])  # in units ergs/s/A/pc^2

        # -------------------------------------------------------------------------------------------
        fig = plt.figure(figsize=(12, 4))
        fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.12, left=0.1, right=0.98)
        which_pixel = ' at ' + str(args_dummy.X) + ', ' + str(args_dummy.Y) if not args_dummy.saveplot else ''
        if args.basic:
            conv = fits.open(logbook_dummy.convolved_filename)[0].data
            plt.plot(wave, conv[args_dummy.X, args_dummy.Y, :]*properties.flux_ratio, c='g', label='conv')
            plt.plot(wave, flam, color='r', label='ppv')
            plt.ylabel(r'f$_\lambda$ (erg/s/A/cm^2/arcsec^2)', fontsize=fs)
        else:
            flam_u = np.array(properties_dummy.ppvcube_u[args_dummy.X, args_dummy.Y,
                              :]) if 'ppvcube_u' in properties_dummy else np.zeros(
                np.shape(flam))  # in units ergs/s/A/pc^2

            lastgood, lastgood_u = flam[0], flam_u[0]
            for ind, ff in enumerate(flam):
                if not np.isnan(ff):
                    lastgood, lastgood_u = ff, flam_u[ind]
                    continue
                else:
                    flam[ind] = lastgood  # to get rid of NANs in flam
                    flam_u[ind] = lastgood_u  # to get rid of NANs in flam_u

            cont, cont_u = fitcont(wave, flam, flam_u, args_dummy, logbook_dummy)  # cont in units ergs/s/A/pc^2
            # print 'flux, flux_u (before contnorm)=', flam[:20], flam_u[:20]  #
            plt.scatter(wave, flam, c='k', s=5, lw=0, label='Spectrum before contfit' + which_pixel if not args_dummy.saveplot else None)
            plt.plot(wave, flam, c='k', label='Spectrum before contfit' + which_pixel)
            plt.plot(wave, flam_u, c='gray', linestyle='dashed', label='Error spectrum before contfit' + which_pixel)
            if args_dummy.contsub:
                flam_u = np.sqrt(flam_u ** 2 + cont_u ** 2)
                flam -= cont
            else:
                flam_u = np.sqrt((flam_u / cont) ** 2 + (flam * cont_u / (cont ** 2)) ** 2)  # flam_u = dimensionless
                flam /= cont  # flam = dimensionless
            # print 'cont', cont[:20], cont_u[:20]  #
            # print 'flux, flux_u (after contnorm)=', flam[:20], flam_u[:20]  #
            # -------------------------------------------------------------------------------------------
            scaling = 0.5 * 1e-18 if args_dummy.contsub else 1.
            #plt.plot(wave, flam / flam_u * scaling, c='b', alpha=0.3, label='SNRx' + str(scaling) + which_pixel)
            plt.plot(wave, cont, c='g', lw=1, alpha=1, label='Continuum' + which_pixel)
            plt.plot(wave, cont_u, c='cyan', lw=1, alpha=0.8, label='Continuum uncertainty' + which_pixel)
            for ii in logbook_dummy.wlist:
                plt.axvline(ii, ymin=0.9, c='blue')
            plt.ylabel(r'f$_\lambda$ (erg/s/A/cm$^2$/arcsec$^2$)', fontsize=fs)
            ax = plt.gca()
            ax.set_xlim(logbook_dummy.wmin, logbook_dummy.wmax)
            if not args_dummy.contsub:
                ax.set_ylim(-10, 20)
            else:
                if args_dummy.branch == 'blue': ax.set_ylim(-2e-19, 1e-16) #
                else: ax.set_ylim(-5e-18, 1e-16)
        plt.xlabel(r'Wavelength ($\AA$)', fontsize=fs)
        plt.legend(fontsize=fs, loc='top left')
        plt.ylim(2e-18, 3e-18) #
        ax = plt.gca()
        ax.set_xticklabels(list(ax.get_xticks()), fontsize=fs)
        ax.set_yticklabels(list(ax.get_yticks()), fontsize=fs)
        # -------------------------------------------------------------------------------------------
        if not args.basic:
            flux, flux_errors = fit_all_lines(args_dummy, logbook_dummy, wave, flam, flam_u, cont, args_dummy.X, args_dummy.Y, z=0., z_err=0.0001)
            if not args_dummy.saveplot: plt.title(logbook_dummy.fitsname + '\n' + 'Fitted spectrum' + which_pixel)
            elif args_dummy.showfit: fig.savefig(logbook_dummy.fitsname + ':Fitted_spec_at_' + str(args_dummy.X) + ',' + str(args_dummy.Y)+'.eps')
            else: fig.savefig(logbook_dummy.fitsname + ':Spec_at_' + str(args_dummy.X) + ',' + str(args_dummy.Y)+'.eps')

            flux_arr = np.hstack([flux_arr, flux]) if kk else flux
            flux_errors_arr = np.hstack([flux_errors_arr, flux_errors]) if kk else flux_errors
            llist_arr = np.hstack([llist_arr, logbook_dummy.llist]) if kk else logbook_dummy.llist
            wlist_arr = np.hstack([wlist_arr, logbook_dummy.wlist]) if kk else logbook_dummy.wlist
    if args_dummy.hide:
        plt.close()
    else:
        plt.show(block=False)

    if not args.basic:
        x_phys_coord = args.X * logbook.final_pix_size - args.galsize/2 - args.xcenter_offset
        y_phys_coord = args.Y * logbook.final_pix_size - args.galsize/2 - args.ycenter_offset
        dist = np.sqrt(x_phys_coord ** 2 + y_phys_coord ** 2)  # kpc
        print 'Galacto-centric distance =', dist, 'kpc'
        print 'Lines:', llist_arr
        print 'Flux=', flux_arr
        print 'Flux error=', flux_errors_arr
        print 'shape=', np.shape(properties.ppvcube), 'coord pixel (', args.X, args.Y, ') = (', x_phys_coord, y_phys_coord, ') kpc', 'SNR cut=', args.SNR_thresh, 'SNRs=', flux_arr / flux_errors_arr
        if args.SNR_thresh is not None: flux_arr = np.ma.masked_where(flux_arr / flux_errors_arr < args.SNR_thresh, flux_arr)
    if args.met and not args.basic:
        if args.mergespec is not None:
            n2_ind = np.where(llist_arr == 'NII6584')[0][0]
            o2a_ind = np.where(llist_arr == 'OII3727')[0][0]
            o2b_ind = np.where(llist_arr == 'OII3729')[0][0]
            dummy, inferred_met, inferred_met_u = get_KD02_met([flux_arr[n2_ind]], [flux_arr[o2a_ind] + flux_arr[o2b_ind]], num_err=[flux_errors_arr[n2_ind]], \
            den_err=[np.sqrt(flux_errors_arr[o2a_ind] ** 2 + flux_errors_arr[o2b_ind] ** 2)])
            print 'Pixel metallicity: inferred with diag KD02 =', inferred_met, '+/-', inferred_met_u, 'true =', dist * args.logOHgrad, \
                'diff =', (dist * args.logOHgrad - inferred_met)
        
        n2_ind = np.where(llist_arr == 'NII6584')[0][0]
        s2a_ind = np.where(llist_arr == 'SII6717')[0][0]
        s2b_ind = np.where(llist_arr == 'SII6730')[0][0]
        ha_ind = np.where(llist_arr == 'H6562')[0][0]
        dummy, inferred_met, inferred_met_u = get_D16_met([flux_arr[n2_ind]], [flux_arr[s2a_ind] + flux_arr[s2b_ind], flux_arr[ha_ind]], \
                                                          num_err=[flux_errors_arr[n2_ind]], den_err=[
                np.sqrt(flux_errors_arr[s2a_ind] ** 2 + flux_errors_arr[s2b_ind] ** 2), flux_errors_arr[ha_ind]])
        print 'Pixel metallicity: inferred with diag D16 =', inferred_met, '+/-', inferred_met_u, 'true =', dist * args.logOHgrad, \
            'diff =', (dist * args.logOHgrad - inferred_met)
    if not args.basic: return flux_arr, flux_errors_arr
    else: return -99, -99


# -------------------------------------------------------------------------------------------
def fitcont(wave, flux, flux_u, args, logbook):
    flux_masked = flux
    flux_u_masked = flux_u
    wave_masked = wave
    for thisw in logbook.wlist:
        linemask_width = thisw * 1.5 * args.vmask / c
        flux_masked = np.ma.masked_where(np.abs(thisw - wave) <= linemask_width, flux_masked)
        flux_u_masked = np.ma.masked_where(np.abs(thisw - wave) <= linemask_width, flux_u_masked)
        wave_masked = np.ma.masked_where(np.abs(thisw - wave) <= linemask_width, wave_masked)

    wave_masked = np.ma.masked_where(np.isnan(flux_masked), wave_masked)
    flux_u_masked = np.ma.masked_where(np.isnan(flux_masked), flux_u_masked)
    flux_masked = np.ma.masked_where(np.isnan(flux_masked), flux_masked)

    '''
    #--option 1: polynomial fit----
    wave_masked1 = np.ma.compressed(wave_masked)
    flux_masked1 = np.ma.compressed(flux_masked)
    inv_chisq, nn, order = [], 3, 20
    for ind in range(nn,len(wave_masked1)-nn):
        (dummy1, chisq, dummy2, dummy3, dummy4) = np.polyfit(wave_masked1[ind-nn:ind+nn+1], flux_masked1[ind-nn:ind+nn+1], 2, full=True) #fitting 2nd order polynomial to every set of adjacent 4 points and storing chi^2 of fit as measure of wigglyness
        try: inv_chisq.append(1./chisq[0])
        except IndexError: inv_chisq.append(inv_chisq[-1])
        #plt.plot(wave[ind-nn:ind+nn+1], np.poly1d(dummy1)(wave[ind-nn:ind+nn+1]),c='b') #

    inv_chisq = np.array(inv_chisq)
    weights = np.log10(np.concatenate(([inv_chisq[0]]*nn, inv_chisq, [inv_chisq[-1]]*nn)))

    weights /= np.max(weights)
    weights = np.power(weights,1./64) #
    contpar = np.polyfit(wave_masked1, flux_masked1, order, w=weights)
    cont = np.poly1d(contpar)(wave)
    '''
    '''
    #--option 2: spline fit------
    boxcar = 11
    flux_smoothed = con.convolve(np.array(flux), np.ones((boxcar,))/boxcar, boundary='fill', fill_value=np.nan)
    flux_smoothed = np.ma.masked_where(np.ma.getmask(flux_masked), flux_smoothed)
    cont = pd.DataFrame(flux_smoothed).interpolate(method='cubic').values.ravel().tolist()
    '''
    # ---option 3: legendre fit------
    # leg = np.polynomial.legendre.Legendre.fit(wave_masked, flux_masked, 4)
    # cont = np.polynomial.legendre.legval(wave, leg.coef)

    # ---option 4: basis spline fit--------
    # contfunc = list(si.splrep(wave_masked, flux_masked, k=5))
    # cont = si.splev(wave, contfunc)

    # ---option 5: mean across uniform spaced lambda and cubic spline interpolation--
    wave_masked1 = np.ma.compressed(wave_masked)
    flux_masked1 = np.ma.compressed(flux_masked)
    flux_u_masked1 = np.ma.compressed(flux_u_masked)
    wave_bins = np.linspace(logbook.wmin, logbook.wmax, args.nbin / 20.)
    wave_index = np.digitize(wave_masked1, wave_bins, right=True)
    wave_smoothed = np.array([wave_masked1[np.where(wave_index == ind)].mean() for ind in range(1, len(wave_bins) + 1)])
    flux_smoothed = np.array([flux_masked1[np.where(wave_index == ind)].mean() for ind in range(1, len(wave_bins) + 1)])
    flux_u_smoothed = np.array(
        [flux_u_masked1[np.where(wave_index == ind)].mean() for ind in range(1, len(wave_bins) + 1)])
    flux_smoothed = flux_smoothed[
        ~np.isnan(wave_smoothed)]  # dropping those fluxes where mean wavelength value in the bin is nan
    flux_u_smoothed = flux_u_smoothed[~np.isnan(wave_smoothed)]
    wave_smoothed = wave_smoothed[~np.isnan(wave_smoothed)]
    '''
    #---option 5b: weighted mean-----------
    weights = 1./(np.array([np.abs(wave_masked[np.where(wave_index==ind)] - wave_smoothed[ind-1]) for ind in range(1,len(wave_bins)+1)]) + 1e-3) #adding small value to avoid weight=0
    flux_bins = np.array([flux_masked[np.where(wave_index==ind)] for ind in range(1,len(wave_bins)+1)])
    flux_smoothed = [np.average(flux_bins[ind], weights=weights[ind]) for ind in range(len(wave_bins))]
    '''
    contfunc = interp1d(wave_smoothed, flux_smoothed, kind='cubic', fill_value='extrapolate')
    cont = contfunc(wave)
    '''
    #----to estimate uncertainty in continuum: option 1: fit the uncertainty in flux--------
    contu_func = interp1d(wave_smoothed, flux_u_smoothed, kind='cubic', fill_value='extrapolate')
    cont_u = contu_func(wave) #np.zeros(len(wave)) #
    '''

    # ----to estimate uncertainty in continuum: option 2: measure RMS of fitted cont w.r.t input flux--------
    cont_masked = contfunc(wave_masked1)
    cont_u = np.ones(len(wave)) * np.sqrt(np.sum((cont_masked - flux_masked1) ** 2) / len(wave))

    '''
    #----to estimate uncertainty in continuum: option 3: measure RMS of fitted smoothed flux w.r.t input flux--------
    cont_u = np.ones(len(wave)) * np.sqrt(np.sum((flux_smoothed - flux_masked1)**2)/len(wave))/np.sqrt(args,nbin/20.)
    '''
    '''
    #----to estimate uncertainty in continuum: option 3: measure deviation of fitted cont w.r.t input flux--------
    cont_masked = contfunc(wave_masked1)
    cont_u_masked = np.abs(cont_masked - flux_masked1)
    contufunc = interp1d(wave_masked1, cont_u_masked, kind='cubic', fill_value='extrapolate')
    cont_u = contufunc(wave)
    '''
    if 'fixcont' in args and args.fixcont:
        print np.shape(wave_smoothed), np.shape(cont), wave_smoothed, flux_smoothed, cont  #
        fig = plt.figure(figsize=(17, 5))
        fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.05, right=0.95)
        plt.plot(wave, cont, c='g', label='cont')
        plt.plot(wave, cont_u, c='g', label='cont_u', linestyle='dotted')
        plt.plot(wave_masked, flux_masked, c='r', label='masked flux')
        plt.scatter(wave_masked, flux_masked, c='r', lw=0)
        plt.plot(wave_masked, flux_u_masked, c='orange', linestyle='dashed', label='masked flux_u')
        plt.plot(wave_smoothed, flux_smoothed, c='k', label='smoothly sampled flux')
        plt.scatter(wave_smoothed, flux_smoothed, c='k', lw=0)
        plt.plot(wave_smoothed, flux_u_smoothed, c='gray', linestyle='dashed', label='smoothly sampled flux_u')
        plt.legend()
        plt.xlabel('Obs wavelength (A)')
        plt.ylabel('flambda (ergs/s/A/pc^2)')
        plt.title('Testing continuum fit for pixel (' + str(args.X) + ',' + str(args.Y) + ')')
        plt.xlim(logbook.wmin, logbook.wmax)
        #plt.ylim(-2e-19, 2e-18)  #
        plt.show(block=False)
        sys.exit()  #

    return np.array(cont), np.array(cont_u)

#  ---------------function for merging HII regions within args.mergeHII kpc distance -------------------------------------------------
def merge_HIIregions(table, args):


    return table

# -------------------------------------------------------------------------------------------
def calcpos(s, center, galsize, res):
    g = int(np.ceil(galsize / res))
    x = s['x(kpc)'] - center # so that s['x(kpc)'] = 655.4 kpc (which is the original center) becomes 0 kpc in new units
    y = s['y(kpc)'] - center # so now (x,y) range from -galsize/2 to galsize/2 kpc
    return g, x, y


# -------------------------------------------------------------------------------------------
def make2Dmap(data, xi, yi, ngrid, res, domean=False, makelog=False, weights=None, galsize=30.):
    map = np.zeros((ngrid, ngrid))

    for i in range(len(data)):
        x = int((xi[i]+galsize/2) / res) # xi, yi originally in range (-galsize/2, galsize/2) kpc
        y = int((yi[i]+galsize/2) / res)
        if makelog:
            quantity = 10. ** data[i]
        else:
            quantity = data[i]

        if weights is not None: quantity *= weights[i]

        map[x][y] += quantity

    if domean:
        map = np.divide(map, make2Dmap(np.ones(len(data)), xi, yi, ngrid, res))
    elif weights is not None:
        map = np.divide(map, make2Dmap(weights, xi, yi, ngrid, res))

    map[np.isnan(map)] = 0
    return map


# -------------------------------------------------------------------------------------------
def getskynoise(args, logbook, properties):
    # to read in sky noise files in physical units, convert to el/s, spectral-bin them and create map
    bluenoise = fits.open(HOME + '/models/Noise_model/NoiseData-99259_B.fits')[1].data[0]
    rednoise = fits.open(HOME + '/models/Noise_model/NoiseData-99259_R.fits')[1].data[0]
    skywave = np.hstack((bluenoise[0], rednoise[0]))  # in Angstrom
    wave = np.array(properties.dispsol)

    noise = np.hstack((bluenoise[1], rednoise[1]))  # in 10^-16 ergs/s/cm^2/A/spaxel
    noise = noise[(skywave >= np.min(wave)) & (skywave <= np.max(wave))]
    noise[noise > 1.] = 0.  # replacing insanely high noise values
    skywave = skywave[(skywave >= np.min(wave)) & (skywave <= np.max(wave))]
    f = interp1d(skywave, noise, kind='cubic', fill_value='extrapolate')
    interp_noise = f(wave)
    # interp_noise = np.lib.pad(interp_noise, (len(np.where(wave<skywave[0])[0]), len(np.where(wave>skywave[-1])[0])), 'constant', constant_values=(0,0))
    interp_noise *= 1e-16 #* np.multiply(interp_noise, properties.delta_lambda)  # to convert it to ergs/s/cm^2/A/spaxel from 10^-16 ergs/s/cm^2/A/spaxel
    factor = 0.5 ** 2  # each spaxel = 0.5 x 0.5 arcsec^2 in size
    interp_noise /= factor  # to transform into ergs/s/cm^2/A/arcsec^2
    return interp_noise


# -------------------------------------------------------------------------------------------
def plotmap(map, title, savetitle, cbtitle, args, logbook, makelog=True, addcircle_radius=None, issnrmap=False, text_on_plot=None):
    fig = plt.figure(figsize=(8, 6))
    top_space = 0.98 if args.saveplot else 0.92 #since, no plot title while saving plots for paper
    fig.subplots_adjust(hspace=0.7, top=top_space, bottom=0.1, left=0.14, right=0.87)
    ax = plt.subplot(111)
    fs = 10 if not args.saveplot else 15 # fontsize
    if 'correct_old_units' in args and args.correct_old_units and not issnrmap:
        correction = np.pi * (171.42857 * 1e4 / (180*3600))**2 # 171.42857 = properties.dist in Mpc # this is correction factor
                                                               # to correct previous ergs/s/pc^2 units to ergs/s/cm^2/arcsec^2 units
        map *= correction # only a factor of ~20
    if makelog:
        a = np.log10(map)  # just a shorter variable
        if issnrmap:
            cmin = np.nanmin(a[a != -np.inf]) if args.snr_cmin is None else args.snr_cmin
            cmax = np.nanmax(a[a != -np.inf]) if args.snr_cmax is None else args.snr_cmax
        else:
            cmin = np.nanmin(a[a != -np.inf]) if args.cmin is None else args.cmin
            cmax = np.nanmax(a[a != -np.inf]) if args.cmax is None else args.cmax
        map = np.ma.masked_where(a < cmin, map)
        p = ax.imshow(np.log10(np.transpose(map)), cmap='rainbow', vmin=cmin, vmax=cmax)
    else:
        a = map  # just a shorter variable
        if issnrmap:
            cmin = np.nanmin(a[a != -np.inf]) if args.snr_cmin is None else args.snr_cmin
            cmax = np.nanmax(a[a != -np.inf]) if args.snr_cmax is None else args.snr_cmax
        else:
            cmin = np.nanmin(a[a != -np.inf]) if args.cmin is None else args.cmin
            cmax = np.nanmax(a[a != -np.inf]) if args.cmax is None else args.cmax
        map = np.ma.masked_where(a < cmin, map)
        p = ax.imshow(np.transpose(map), cmap='rainbow', vmin=cmin,
                      vmax=cmax)  # transposing map due to make [X,Y] axis of array correspond to [X,Y] axis of plot
    plt.ylabel('y(kpc)', fontsize=fs)
    plt.xlabel('x(kpc)', fontsize=fs)
    lim = np.array([-10,10])# in kpc
    plt.xlim((np.array(lim)+args.xcenter_offset + args.galsize/2)/logbook.final_pix_size)
    plt.ylim((np.array(lim)+args.xcenter_offset + args.galsize/2)/logbook.final_pix_size)
    #ax.set_xticks([(item + args.xcenter_offset + args.galsize/2)/logbook.final_pix_size for item in np.linspace(-4, 4, 3)], minor=True)
    ax.set_xticks([(item + args.xcenter_offset + args.galsize/2)/logbook.final_pix_size for item in np.linspace(-10, 10, 5)])
    #ax.set_yticks([(item + args.xcenter_offset + args.galsize/2)/logbook.final_pix_size for item in np.linspace(-4, 4, 3)], minor=True)
    ax.set_yticks([(item + args.ycenter_offset + args.galsize/2)/logbook.final_pix_size for item in np.linspace(-10, 10, 5)])
    ax.set_xticklabels(['%.2F'%(i * logbook.final_pix_size - args.galsize/2 - args.xcenter_offset) for i in
                        list(ax.get_xticks())], fontsize=fs)  # xcenter_offset in kpc units
    ax.set_yticklabels(['%.2F'%(i * logbook.final_pix_size - args.galsize/2 - args.ycenter_offset) for i in
                        list(ax.get_yticks())], fontsize=fs)  # ycenter_offset in kpc units
    if not args.saveplot: plt.title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(p, cax=cax)
    cb.set_label(cbtitle, fontsize=fs)
    cb.ax.tick_params(labelsize=fs)
    if addcircle_radius:
        circle1 = plt.Circle(((0 + args.xcenter_offset + args.galsize/2.) / logbook.final_pix_size,
                              (0 + args.ycenter_offset + args.galsize/2.) / logbook.final_pix_size),
                             addcircle_radius / logbook.final_pix_size, color='k', fill=False, lw=0.3)
        ax.add_artist(circle1)
    if text_on_plot is not None:
        ax.text(ax.get_xlim()[-1]*0.9, ax.get_ylim()[-1]*0.9, text_on_plot, color='k', ha='right', va='center', fontsize=fs, bbox=dict(facecolor='white', alpha=1.0))
    if args.saveplot:
        fig.savefig(savetitle + '.eps')
        print 'Saved file '+savetitle + '.eps'
    if args.hide:
        plt.close(fig)
    else:
        plt.show(block=False)
    return map


# -------------------------------------------------------------------------------------------
def getfn(args):
    mergeHII_text = '_mergeHII=' + str(args.mergeHII) + 'kpc' if args.mergeHII is not None else ''
    return HOME + '/models/emissionlist' + args.outtag + '/emissionlist_' + args.file + '_Om' + str(args.Om) + mergeHII_text + '.txt'


# -------------------------------------------------------------------------------------------
def getsmoothparm(args, properties, logbook):
    if args.parm is None:
        if args.ker == 'AOPSF': args.pow, args.size = 2.5, 7  # power and size/sigma parameters for 2D Moffat kernal in pixel units
        else: args.pow, args.size = 4.7, 5  # power and size/sigma parameters for 2D Moffat kernal in pixel units
    else:
        args.pow, args.size = args.parm[0], args.parm[1]
    # ------------------------------------------------------------------
    # -----Compute effective seeing i.e. FWHM as: the resolution on sky (res_arcsec) -> physical resolution on target frame (res_phys) -> pixel units (fwhm)
    if np.round(properties.res_phys / args.res, 4) < args.pix_per_beam:
        #myexit(args, text='For given physical resolution (' + str('%.2F' % properties.res_phys) + ' kpc), pix per beam with base res(' + str(args.res) + ' kpc) becomes ' + str('%.2F' % \
        #(properties.res_phys / args.res)) + ' which is less than the pix per beam asked for (' + str(args.pix_per_beam) + '). Hence, quitting.\n')  # if the pix per beam using base resolution is less than required pix per beam then flag and exit
        logbook.fwhm = properties.res_phys / args.res
        #if args.binto is not None and float(logbook.fwhm) <= float(args.binto):
        if args.binto is not None and logbook.fwhm <= args.binto:  # this is not completely corect; replace with above line for next generation simulations
            args.binto = None  # no resampling after convolution
    else:
        logbook.fwhm = args.pix_per_beam
    new_res = properties.res_phys / logbook.fwhm  # need to re-bin the map such that each pixel is now of size 'new_res' => we have 'fwhm' pixels within every 'res_phys' physical resolution element
    logbook.intermediate_pix_size = args.galsize / int(np.round(args.galsize / new_res))  # actual pixel size we will end up with, after rebinning
    logbook.fwhm = properties.res_phys / logbook.intermediate_pix_size

    # ------------------------------------------------------------------
    if args.ker == 'gauss':
        logbook.sig = gf2s * logbook.fwhm
    elif args.ker == 'moff' or args.ker == 'AOPSF':
        logbook.sig = logbook.fwhm / (2 * np.sqrt(2 ** (1. / args.pow) - 1.))
    logbook.size = int((logbook.sig * args.size) // 2 * 2 + 1)  # rounding off to nearest odd integer because kernels need odd integer as size
    if args.binback:
        logbook.final_pix_size = args.res  # if re-sampling convolved cube back to original finer resolution
    elif args.binto is not None:
        dummy_new_res = properties.res_phys / float(args.binto)
        logbook.final_pix_size = args.galsize / int(
            np.round(args.galsize / dummy_new_res))  # if re-sampling convolved cube to 'binto' pix per beam
    else:
        logbook.final_pix_size = logbook.intermediate_pix_size  # if not
    return args, logbook


# -------------------------------------------------------------------------------------------
def getfitsname(args, properties):
    global info
    logbook = ap.Namespace()
    logbook.wlist, logbook.llist = readlist()
    logbook.wmin = logbook.wlist[0] - 50. if args.wmin is None else args.wmin
    logbook.wmax = logbook.wlist[-1] + 50. if args.wmax is None else args.wmax
    logbook.llist = logbook.llist[np.where(np.logical_and(logbook.wlist > logbook.wmin,
                                                          logbook.wlist < logbook.wmax))]  # truncate linelist as per wavelength range
    logbook.wlist = logbook.wlist[np.where(np.logical_and(logbook.wlist > logbook.wmin, logbook.wlist < logbook.wmax))]
    info = ''
    if args.spec_smear: info += '_specsmeared_' + str(int(args.vres)) + 'kmps'
    info1 = info
    if args.smooth:
        arc_string = '_arc=' + str(args.res_arcsec)
        args, logbook = getsmoothparm(args, properties, logbook)
        info += '_smeared_' + args.ker + '_parm' + str('%.1F' % logbook.fwhm) + ',' + str(
            '%.1F' % logbook.sig) + ',' + str('%.1F' % args.pow) + ',' + str(logbook.size)
        if args.binto:
            info += '_binto' + str(args.binto)
        elif args.binback:
            info += '_binback'
    else:
        logbook.final_pix_size = args.res
        arc_string = ''
    info2 = info
    if args.addnoise:
        info += '_noisy'
        if args.fixed_SNR is not None: info += '_fixed_SNR' + str(args.fixed_SNR)
        if args.fixed_noise is not None: info += '_fixed_noise' + str(args.fixed_noise)
    if not args.maketheory:
        info += '_obs'
        if args.exptime is not None:
            logbook.exptime = float(args.exptime)
        else:
            if args.scale_exptime:
                scalefactor = float(args.scale_exptime)
            elif args.scale_exp_SNR:
                scalefactor = 1200. * float(args.scale_exp_SNR)
            else:
                scalefactor = 1200.  # sec
            if not args.fixed_SNR: info += '_scaled'
            lambda_spacing = args.vres * 6562. / c if args.spec_smear else 10 * args.vdel * 6562. / (args.nhr * c)
            logbook.exptime = float(scalefactor) * (0.08 / logbook.final_pix_size) ** 2 * (
                    0.6567 / lambda_spacing)  # increasing exposure time quadratically with finer resolution
            # scaled to <scale_factor> sec for 2" (=0.8kpc cell size = 1.6kpc FWHM) spatial and 30kmps spectral resolution

        info += '_exp' + str(int(logbook.exptime)) + 's'
    if args.multi_realisation: info += '_real' + str(args.multi_realisation)
    if args.branch is not None and len(args.branch)>0: which_branch = '_'+args.branch # in case of making only the blue (or red) part of the spectra
    else: which_branch  = '' # no separate red/blue branch
    mergeHII_text = '_mergeHII=' + str(args.mergeHII) + 'kpc' if args.mergeHII is not None else ''

    logbook.H2R_filename = args.path + 'H2R_' + args.file + mergeHII_text + which_branch + 'Om=' + str(args.Om) + '_res' + str(
        args.res) + 'kpc_' + str(logbook.wmin) + '-' + str(logbook.wmax) + 'A' + info1 + args.gradtext + '.fits'
    logbook.skynoise_cubename = args.path + 'skycube_' + which_branch + str(logbook.wmin) + '-' + str(
        logbook.wmax) + 'A' + info1 + '.fits'
    logbook.convolved_filename = args.path + 'convolved_' + args.file + mergeHII_text + which_branch + 'Om=' + str(args.Om) + '_res' + str(
        args.res) + 'kpc' + arc_string + '_' + str(logbook.wmin) + '-' + str(
        logbook.wmax) + 'A' + info2 + args.gradtext + '.fits'
    logbook.fitsname = args.path + 'PPV_' + args.file + mergeHII_text + which_branch + 'Om=' + str(args.Om) + arc_string + '_' + str(
        logbook.wmin) + '-' + str(logbook.wmax) + 'A' + info + args.gradtext + '.fits'
    logbook.fitsname_u = logbook.fitsname.replace('PPV',
                                                  'ERROR')  # name of errorcube corresponding to PPV file to be read in
    logbook.fittedcube = logbook.fitsname.replace('PPV', 'fitted-map-cube')  # name of mapcube file to be read in
    logbook.fittederror = logbook.fittedcube.replace('map', 'error')

    if args.addnoise:
        no_noise_fitsname = logbook.fitsname[:logbook.fitsname.find('noisy')] + logbook.fitsname[
                                                                                logbook.fitsname.find('obs'):]
        FNULL = open(os.devnull, 'w')
        try:
            wildcard = repl_wildcard(no_noise_fitsname, ['exp', 'real'],
                                     repl='*')  # replacing the 'expXXXs' with 'exp*s' and 'realXX_' with 'real*_' so as to search for file with whatever exp time and realization but all other parameters same, bcz exp time and realization do not matter for no-noise line map
            result = subprocess.check_output(['ls ' + wildcard], stderr=FNULL, shell=1)
            logbook.no_noise_fitsname = result.split('\n')[0]
        except subprocess.CalledProcessError:
            logbook.no_noise_fitsname = repl_wildcard(no_noise_fitsname, ['real'], repl='1')
        FNULL.close()
        if args.branch == 'blue': logbook.no_noise_fitsname = logbook.no_noise_fitsname.replace('_blue','').replace('3680.0-3780.0','6400.0-6782.674')
    else:
        logbook.no_noise_fitsname = logbook.fitsname
    logbook.no_noise_fittedcube = logbook.no_noise_fitsname.replace('PPV',
                                                                    'fitted-map-cube')  # name of mapcube file to be read in
    logbook.no_noise_fittederror = logbook.no_noise_fittedcube.replace('map', 'error')
    if args.debug: print '\nDeb1023: logbook = ', logbook, '\n'
    return args, logbook


# -------------------------------------------------------------------------------------------
def write_fits(filename, data, args, fill_val=np.nan):
    hdu = fits.PrimaryHDU(np.ma.filled(data, fill_value=fill_val))
    hdulist = fits.HDUList([hdu])
    if filename[-5:] != '.fits':
        filename += '.fits'
    hdulist.writeto(filename, clobber=True)
    myprint('Written file ' + filename + '\n', args)


# ------to get fraction of non zero pixels in line map above given SNR within given radius------
def get_valid_frac(map, map_u, radius, args):
    x, y = np.ogrid[:np.shape(map)[0], :np.shape(map)[1]]
    dist = np.sqrt((x - np.shape(map)[0] / 2) ** 2 + (y - np.shape(map)[1] / 2) ** 2)
    radius = (radius / args.galsize) * np.shape(map)[0]
    map = np.ma.masked_where(dist > radius, map)
    ndatapoints = len(map.nonzero()[0])
    map = np.ma.masked_where(map / map_u < args.SNR_thresh, map)
    nvalid = len(map.nonzero()[0])
    return float(nvalid) / float(ndatapoints)  # fraction

# ------to rebin cubes without convolution: emulating Henry's procedure------
def emulate_henry_getmet(args, logbook, properties):
    global info
    g = np.shape(properties.mapcube)[0]
    b = np.linspace(-g / 2 + 1, g / 2, g) * (args.galsize) / g  # in kpc
    t = args.file + '_Met_Om' + str(args.Om)
    if args.smooth: t += '_arc' + str(args.res_arcsec)
    if args.spec_smear: t += '_vres=' + str(args.vres) + 'kmps'
    t += info + args.gradtext
    if args.SNR_thresh is not None: t += '_snr' + str(args.SNR_thresh)
    t += '_target_res'+str(args.target_res)+'pc'
    flux_limit = 1e-15  # in ergs/s; discard fluxes below this as machine precision errors
    met_minlim, met_maxlim = -1.5, 0.5
    fs = 20  # fontsize of labels
    if args.mergespec is not None: diag_arr = ['D16', 'KD02']
    elif args.useKD: diag_arr = ['KD02']
    else: diag_arr = ['D16']
    col = ['r', 'b', 'k']

    if not args.hide:
        fig, axes = plt.subplots(len(diag_arr), 1, sharex=True, figsize=(8, 4*len(diag_arr)))
        fig.subplots_adjust(hspace=0.1, wspace=0.25, top=0.9, bottom=0.15, left=0.14, right=0.98)

    properties_orig = deepcopy(properties)

    for (ii,diag) in enumerate(diag_arr):
        if diag == 'D16':
            args.num_arr = [['NII6584'], ['NII6584']]
            args.den_arr = [['SII6717', 'SII6730'], ['H6562']]
        elif diag == 'KD02':
            args.num_arr = [['NII6584']]
            args.den_arr = [['OII3727', 'OII3729']]
        for kk in range(len(properties.mask_ar)):
            d = np.sqrt((b[:, None] - args.xcenter_offset) ** 2 + (b - args.ycenter_offset) ** 2)
            properties.mapcube = properties_orig.mapcube
            properties.errorcube = properties_orig.errorcube
            properties.mapcube = np.ma.masked_where(properties.mask_ar[kk], properties.mapcube)
            properties.errorcube = np.ma.masked_where(properties.mask_ar[kk], properties.errorcube)

            map_num, map_num_u, map_den, map_den_u = [], [], [], []

            for num_grp in args.num_arr:
                map_num_grp, map_num_grp_u = [], []
                if not iterable(num_grp): num_grp = [num_grp]

                for (jj, num) in enumerate(num_grp):
                    temp_u = properties.errorcube[:, :, np.where(logbook.llist == num)[0][0]] * (logbook.final_pix_size * 1e3) ** 2
                    temp_u = np.ma.masked_where(temp_u <= 0., temp_u)
                    map_num_grp_u.append(temp_u)
                    temp = properties.mapcube[:, :, np.where(logbook.llist == num)[0][0]] * (logbook.final_pix_size * 1e3) ** 2
                    temp = np.ma.masked_where(temp <= flux_limit, temp)  # discard too low fluxes that are machine precision errors
                    if args.SNR_thresh is not None: temp = np.ma.masked_where(temp / temp_u < args.SNR_thresh, temp)
                    map_num_grp.append(temp)
                map_num.append(map_num_grp)
                map_num_u.append(map_num_grp_u)

            for den_grp in args.den_arr:
                map_den_grp, map_den_grp_u = [], []
                if not iterable(den_grp): den_grp = [den_grp]

                for (jj, den) in enumerate(den_grp):
                    temp_u = properties.errorcube[:, :, np.where(logbook.llist == den)[0][0]] * (logbook.final_pix_size * 1e3) ** 2
                    temp_u = np.ma.masked_where(temp_u <= 0., temp_u)
                    map_den_grp_u.append(temp_u)
                    temp = properties.mapcube[:, :, np.where(logbook.llist == den)[0][0]] * (logbook.final_pix_size * 1e3) ** 2
                    temp = np.ma.masked_where(temp <= flux_limit, temp)  # discard too low fluxes that are machine precision errors
                    if args.SNR_thresh is not None: temp = np.ma.masked_where(temp / temp_u < args.SNR_thresh, temp)
                    map_den_grp.append(temp)
                map_den.append(map_den_grp)
                map_den_u.append(map_den_grp_u)

            map_num_series = [mysum(map_num[ind]) for ind in range(len(args.num_arr))]
            map_num_u_series = [mysum(map_num_u[ind], iserror=True) for ind in range(len(args.num_arr))]
            map_den_series = [mysum(map_den[ind]) for ind in range(len(args.den_arr))]
            map_den_u_series = [mysum(map_den_u[ind], iserror=True) for ind in range(len(args.den_arr))]

            if diag == 'KD02': logOHsol, logOHobj_map, logOHobj_map_u = get_KD02_met(map_num_series, map_den_series, num_err=map_num_u_series,
                                                                      den_err=map_den_u_series)  # =log(O/H)_obj - log(O/H)_sun; in log units
            elif diag == 'D16': logOHsol, logOHobj_map, logOHobj_map_u = get_D16_met(map_num_series, map_den_series, num_err=map_num_u_series,
                                                                     den_err=map_den_u_series)  # =log(O/H)_obj - log(O/H)_sun; in log units
            # ---------------------------------------------------
            Z_list = logOHobj_map.flatten()
            Z_u_list = logOHobj_map_u.flatten()

            d = np.ma.masked_array(d, logOHobj_map.mask | logOHobj_map_u.mask)
            if args.showcoord: coord_list = np.transpose(np.where(~d.mask))
            d_list = d.flatten()
            d_list = np.ma.compressed(d_list)
            Z_u_list = np.ma.masked_array(Z_u_list, Z_list.mask)
            Z_list = np.ma.masked_array(Z_list, Z_u_list.mask)
            Z_u_list = np.ma.compressed(Z_u_list)
            Z_list = np.ma.compressed(Z_list)
            Z_list = np.array([x for (y, x) in sorted(zip(d_list, Z_list), key=lambda pair: pair[0])])  # sorting by distance
            Z_u_list = np.array([x for (y, x) in sorted(zip(d_list, Z_u_list), key=lambda pair: pair[0])])
            d_list = np.sort(d_list)
            if args.fitupto is not None:  # to fit metallicity gradient only upto 'args.fitupto' x args.scale_length
                Z_list = Z_list[d_list <= float(args.fitupto) * args.scale_length]
                Z_u_list = Z_u_list[d_list <= float(args.fitupto) * args.scale_length]
                d_list = d_list[d_list <= float(args.fitupto) * args.scale_length]

            #----------plotting begins------------------
            if not args.hide:
                ax = axes[ii] if len(diag_arr) > 1 else axes
                if not kk == len(properties.mask_ar)-1: ax.scatter(d_list, Z_list, s=5, lw=0, c=col[kk], alpha=0.3) # plot individual data points for only HII and only DIG cases

            limit = args.galsize / 2. if args.fitupto is None else float(args.fitupto) * args.scale_length
            d_bin, Z_bin, Z_u_bin = bin_data(d_list, Z_list, np.linspace(0, limit, int(limit) + 1), err=Z_u_list)
            if not args.hide: ax.errorbar(d_bin, Z_bin, yerr=Z_u_bin, ls='None', c=col[kk], fmt='o', alpha=0.5)

            # ---to compute apparent gradient and scatter---
            myprint(diag+':'+properties.text_ar[kk]+': Fitting gradient..' + '\n', args)
            myprint('No. of pixels being considered for gradient fit = ' + str(len(Z_list)) + '\n', args)
            if len(Z_list) <= 4:  # no .of data points must exceed order+2 for covariance estimation; in our case order = deg+1 = 2
                myprint('Not enough data points for vres= ' + str(args.vres) + ' above given SNR_thresh of ' + str(args.SNR_thresh) + '\n.', args)
                slope_text = ''
            else:
                linefit, linecov = np.polyfit(d_list, Z_list, 1, cov=True, w=1. / Z_u_list)
                slope_text = '%.4F dex/kpc'%linefit[0]
                # linefit, linecov =  curve_fit(met_profile, d_list, Z_list, p0=[-0.05, 0.], sigma=Z_u_list, absolute_sigma=True) #gives same result as np.polyfit
                myprint('Fit paramters: ' + str(linefit) + '\n', args)
                myprint('Fit errors: ' + str(linecov) + '\n', args)
                x_arr = np.arange(args.galsize / 2)
                if not args.hide: ax.plot(x_arr, np.poly1d(linefit)(x_arr), c=col[kk])
                if not args.nowrite:
                    snr_cut_text = str(args.SNR_thresh) if args.SNR_thresh is not None else '-99'
                    gradfile = 'emulate_henry_met_grad_' + args.file + 'Om=' + str(args.Om) + '_arc' + str(args.res_arcsec) + \
                               '_vres'+str(args.vres)+'kmps'+'_fixed_noise'+str(args.fixed_noise)+'_logOHgrad'+str(args.logOHgrad)
                    if args.fitupto is not None: gradfile += '_fitupto' + str(args.fitupto)
                    gradfile += '.txt'
                    if not os.path.exists(gradfile):
                        head = '#File to store metallicity gradient information for different telescope parameters, as pandas dataframe\n\
#Columns are:\n\
#diagnostic : D16 or KD02\n\
#component: DIG or HII or total\n\
#res_bin : resolution by rebinning (in pcspaxel)\n\
#slope, intercept : fitted parameters\n\
#slope_u, intercept_u : uncertainty in above quantities\n\
#snr_cut : snr_cut applied, if any (-99 if no cut applied)\n\
#by Ayan\n\
diagnostic      component     res_bin     slope       slope_u     snr_cut         \n'
                        open(gradfile, 'w').write(head)
                    with open(gradfile, 'a') as fout:
                        output = '\n' + diag + '\t\t' + properties.text_ar[kk] + '\t\t' + \
                                 str(args.galsize*1e3/np.shape(properties.mapcube)[0]) + '\t\t' + str('%.4F' % linefit[0]) + '\t\t' + \
                                 str('%0.4F' % np.sqrt(linecov[0][0])) + '\t\t' + snr_cut_text
                        fout.write(output)

        if not args.hide:
            ax.set_ylim(met_minlim, met_maxlim)
            ax.set_ylabel(r'$\log{(Z/Z_{\bigodot})}$', fontsize=fs)
            ax.set_xlim(0, args.galsize / 2)
            ax.text(ax.get_xlim()[-1] * 0.9, ax.get_ylim()[-1] * 0.75 - kk * 0.2,
                    properties.text_ar[kk] + ': ' + slope_text, color=col[kk], ha='right', va='center',
                    fontsize=fs * 0.7)
            ax.axhline(0, c='k', linestyle='--')  # line for solar metallicity
            ax.text(ax.get_xlim()[0]*2., ax.get_ylim()[-1]*0.65, diag, color='k', ha='left', va='center', fontsize=fs)

    if not args.hide:
        ax.set_xticklabels(['%.1F' % (i / args.scale_length) for i in list(ax.get_xticks())], fontsize = fs)
        plt.xlabel('R/Re', fontsize=fs)
        if args.saveplot:
            fig.savefig(args.path + t + '.eps')
            myprint('Saved file ' + args.path + t + '.eps', args)

# ------to rebin cubes without convolution: emulating Henry's procedure------
def emulate_henry_getcube(args, logbook):
    if args.target_res is None:
        args.target_res = logbook.final_pix_size*1e3 #kpc #keepig resolution same
    else:
        args.target_res = float(args.target_res)
    new_suffix = '_target_res'+str(args.target_res)+'pc'
    temp_fitsname = os.path.splitext(logbook.fitsname)[0] + new_suffix + '.fits'
    temp_fitsname_u = os.path.splitext(logbook.fitsname_u)[0] + new_suffix + '.fits'
    if not os.path.exists(temp_fitsname):
        if args.target_res == logbook.final_pix_size*1e3: # no rebinning required
            myprint('Emulate Henry: No rebinning necessary as per target_res, hence simply re-saving ppvcube.\n', args)
            ppvcube = fits.open(logbook.fitsname)[0].data
            ppvcube_u = fits.open(logbook.fitsname_u)[0].data
            write_fits(temp_fitsname, ppvcube, args)
            write_fits(temp_fitsname_u, ppvcube_u, args)
        else:
            myprint('Emulate Henry: Trying to parallely rebin with ' + str(args.ncores) + ' core...\n', args)
            funcname = HOME + WD + 'parallel_rebin.py'
            if args.silent:
                silent = ' --silent'
            else:
                silent = ''
            if args.toscreen:
                toscreen = ' --toscreen'
            else:
                toscreen = ''
            if args.debug:
                debug = ' --debug'
            else:
                debug = ''
            if args.saveplot:
                saveplot = ' --saveplot'
            else:
                saveplot = ''
            if args.hide:
                hide = ' --hide'
            else:
                hide = ''
            if args.cmin is not None:
                cmin = ' --cmin ' + str(args.cmin)
            else:
                cmin = ''
            if args.cmax is not None:
                cmax = ' --cmax ' + str(args.cmax)
            else:
                cmax = ''

            command = args.which_mpirun + ' -np ' + str(args.ncores) + ' python ' + funcname + ' --fitsname ' + logbook.fitsname + \
                      ' --fitsname_u ' + logbook.fitsname_u + ' --target_res ' + str(args.target_res) + ' --outfile ' + args.outfile + \
                    ' --rebinned_filename ' + temp_fitsname + ' --rebinned_u_filename ' + temp_fitsname_u + ' --file ' + args.file + \
                      ' --xcenter_offset ' + str(args.xcenter_offset) + ' --ycenter_offset ' + str(args.ycenter_offset) + \
                      ' --galsize ' + str(args.galsize) + ' --center ' + str(args.center) + silent + toscreen + debug + cmin + cmax + saveplot + hide
            myprint(command + '\n', args)
            subprocess.call([command], shell=True)

    logbook.fitsname = temp_fitsname
    logbook.fitsname_u = temp_fitsname_u
    logbook.fittedcube = logbook.fitsname.replace('PPV', 'fitted-map-cube')  # name of mapcube file to be read in
    logbook.fittederror = logbook.fittedcube.replace('map', 'error')
    return args, logbook

# -------------------------------------------------------------------------------------------
def congrid(a, newdims, method='linear', centre=False, minusone=True):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).

    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array(a.shape)
    ndims = len(a.shape)
    if len(newdims) != ndims:
        print "[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions."
        return None
    newdims = np.asarray(newdims, dtype=float)
    dimlist = []

    if method == 'neighbour':
        for i in range(ndims):
            base = np.indices(newdims)[i]
            dimlist.append((old[i] - m1) / (newdims[i] - m1) \
                           * (base + ofs) - ofs)
        cd = np.array(dimlist).round().astype(int)
        newa = a[list(cd)]

    elif method in ['nearest', 'linear']:
        # calculate new dims
        for i in range(ndims):
            base = np.arange(newdims[i])
            dimlist.append((old[i] - m1) / (newdims[i] - m1) \
                           * (base + ofs) - ofs)
        # specify old dims
        olddims = [np.arange(i, dtype=np.float) for i in list(a.shape)]

        # first interpolation - for ndims = any
        mint = si.interp1d(olddims[-1], a, kind=method)
        newa = mint(dimlist[-1])

        trorder = [ndims - 1] + range(ndims - 1)
        for i in range(ndims - 2, -1, -1):
            newa = newa.transpose(trorder)

            mint = si.interp1d(olddims[i], newa, kind=method)
            newa = mint(dimlist[i])

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose(trorder)

    elif method in ['spline']:
        oslices = [slice(0, j) for j in old]
        oldcoords = np.ogrid[oslices]
        nslices = [slice(0, j) for j in list(newdims)]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        # make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = ndimage.map_coordinates(a, newcoords)
    else:
        print "Congrid error: Unrecognized interpolation type.\n", \
            "Currently only \'neighbour\', \'nearest\',\'linear\',", \
            "and \'spline\' are supported."
        return None
    for ind in range(len(newdims)): newa *= old[ind] / newdims[ind]  # Ayan's ammendment; to keep total flux conserved.
    return newa


# -----------------------------------------------------------------
def rebin(array, dimensions=None, scale=None):
    """ Return the array ``array`` to the new ``dimensions`` conserving flux the flux in the bins
    The sum of the array will remain the same

    >>> ar = numpy.array([
        [0,1,2],
        [1,2,3],
        [2,3,4]
        ])
    >>> rebin(ar, (2,2))
    array([
        [1.5, 4.5]
        [4.5, 7.5]
        ])
    Raises
    ------

    AssertionError
        If the totals of the input and result array don't agree, raise an error because computation may have gone wrong

    Reference
    =========
    +-+-+-+
    |1|2|3|
    +-+-+-+
    |4|5|6|
    +-+-+-+
    |7|8|9|
    +-+-+-+
    """
    if dimensions is not None:
        if isinstance(dimensions, float):
            dimensions = [int(dimensions)] * len(array.shape)
        elif isinstance(dimensions, int):
            dimensions = [dimensions] * len(array.shape)
        elif len(dimensions) != len(array.shape):
            raise RuntimeError('')
    elif scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            dimensions = map(int, map(round, map(lambda x: x * scale, array.shape)))
        elif len(scale) != len(array.shape):
            raise RuntimeError('')
    else:
        raise RuntimeError('Incorrect parameters to rebin.\n\trebin(array, dimensions=(x,y))\n\trebin(array, scale=a')
    if np.shape(array) == dimensions: return array  # no rebinning actually needed
    import itertools
    # dY, dX = map(divmod, map(float, array.shape), dimensions)

    result = np.zeros(dimensions)
    for j, i in itertools.product(*map(xrange, array.shape)):
        (J, dj), (I, di) = divmod(j * dimensions[0], array.shape[0]), divmod(i * dimensions[1], array.shape[1])
        (J1, dj1), (I1, di1) = divmod(j + 1, array.shape[0] / float(dimensions[0])), divmod(i + 1,
                                                                                            array.shape[1] / float(
                                                                                                dimensions[1]))

        # Moving to new bin
        # Is this a discrete bin?
        dx, dy = 0, 0
        if (I1 - I == 0) | ((I1 - I == 1) & (di1 == 0)):
            dx = 1
        else:
            dx = 1 - di1
        if (J1 - J == 0) | ((J1 - J == 1) & (dj1 == 0)):
            dy = 1
        else:
            dy = 1 - dj1
        # Prevent it from allocating outide the array
        I_ = min(dimensions[1] - 1, I + 1)
        J_ = min(dimensions[0] - 1, J + 1)
        result[J, I] += array[j, i] * dx * dy
        result[J_, I] += array[j, i] * (1 - dy) * dx
        result[J, I_] += array[j, i] * dy * (1 - dx)
        result[J_, I_] += array[j, i] * (1 - dx) * (1 - dy)
    allowError = 0.1
    assert (array.sum() < result.sum() * (1 + allowError)) & (array.sum() > result.sum() * (1 - allowError))
    return result


# -------------------------------------------------------------------------------------------
def calc_dist(z, H0=70.):
    dist = z * c / H0  # Mpc
    return dist


# -------------------------------------------------------------------------------------------
def masked_data(data):
    data = np.ma.masked_where(np.isnan(data), data)
    #data = np.ma.masked_where(data<=0, data)
    data = np.ma.compressed(data.flatten())
    return data


# -------------------------------------------------------------------------------------------
def mydiag(title, data, args):
    myprint(title + ': Mean, median, stdev, max, min, non-zero min = ' + str(np.mean(masked_data(data))) + ',' + str(
        np.median(masked_data(data))) + ',' + str(np.std(masked_data(data))) + ',' + \
            str(np.max(masked_data(data))) + ',' + str(np.min(masked_data(data))) + ',' + str(
        np.min(np.ma.masked_where(data <= 0, data))) + '\n', args)


# -------------------------------------------------------------------------------------------
def myprint(text, args):
    if not text[-1] == '\n': text += '\n'
    if not args.silent:
        if args.toscreen or args.debug:
            print text
        else:
            ofile = open(args.outfile, 'a')
            ofile.write(text)
            ofile.close()


# -------------------------------------------------------------------------------------------
def myexit(args, text=''):
    myprint(text + ' Exiting by encountering sys.exit() in the code.', args)
    sys.exit()


# -------------------End of functions------------------------------------------------------------------------
# -------------------Begin main code------------------------------------------------------------------------
global info
col_ar = ['m', 'blue', 'steelblue', 'aqua', 'lime', 'darkolivegreen', 'goldenrod', 'orangered', 'darkred', 'dimgray']
logOHsun = 8.77
c = 3e5  # km/s
H0 = 70.  # km/s/Mpc Hubble's constant
planck = 6.626e-27  # ergs.sec Planck's constant
nu = 5e14  # Hz H-alpha frequency to compute photon energy approximately
f_esc = 0.0
f_dust = 0.0
if __name__ == '__main__':
    properties = ap.Namespace()
    # -------------------arguments parsed-------------------------------------------------------
    parser.add_argument('--bptpix', dest='bptpix', action='store_true')
    parser.set_defaults(bptpix=False)
    parser.add_argument('--gridoverlay', dest='gridoverlay', action='store_true')
    parser.set_defaults(gridoverlay=False)
    parser.add_argument('--annotategrid', dest='annotategrid', action='store_true')
    parser.set_defaults(annotategrid=False)
    parser.add_argument('--theory', dest='theory', action='store_true')
    parser.set_defaults(theory=True)
    parser.add_argument('--map', dest='map', action='store_true')
    parser.set_defaults(map=False)
    parser.add_argument('--sfr', dest='sfr', action='store_true')
    parser.set_defaults(sfr=False)
    parser.add_argument('--fixfit', dest='fixfit', action='store_true')
    parser.set_defaults(fixfit=False)
    parser.add_argument('--fixcont', dest='fixcont', action='store_true')
    parser.set_defaults(fixcont=False)
    parser.add_argument('--met', dest='met', action='store_true')
    parser.set_defaults(met=False)
    parser.add_argument('--getmap', dest='getmap', action='store_true')
    parser.set_defaults(getmap=False)
    parser.add_argument('--ppv', dest='ppv', action='store_true')
    parser.set_defaults(ppv=False)
    parser.add_argument('--plotintegmap', dest='plotintegmap', action='store_true')
    parser.set_defaults(plotintegmap=False)
    parser.add_argument('--inspect', dest='inspect', action='store_true')
    parser.set_defaults(inspect=False)
    parser.add_argument('--plotspec', dest='plotspec', action='store_true')
    parser.set_defaults(plotmap=False)
    parser.add_argument('--savecube', dest='savecube', action='store_true')
    parser.set_defaults(savecube=False)
    parser.add_argument('--saveplot', dest='saveplot', action='store_true')
    parser.set_defaults(saveplot=False)
    parser.add_argument('--spec_smear', dest='spec_smear', action='store_true')
    parser.set_defaults(spec_smear=False)
    parser.add_argument('--smooth', dest='smooth', action='store_true')
    parser.set_defaults(smooth=False)
    parser.add_argument('--addnoise', dest='addnoise', action='store_true')
    parser.set_defaults(addnoise=False)
    parser.add_argument('--keepprev', dest='keepprev', action='store_true')
    parser.set_defaults(keepprev=False)
    parser.add_argument('--hide', dest='hide', action='store_true')
    parser.set_defaults(hide=False)
    parser.add_argument('--maketheory', dest='maketheory', action='store_true')
    parser.set_defaults(maketheory=False)
    parser.add_argument('--clobber', dest='clobber', action='store_true')
    parser.set_defaults(clobber=False)
    parser.add_argument('--calcgradient', dest='calcgradient', action='store_true')
    parser.set_defaults(calcgradient=False)
    parser.add_argument('--nowrite', dest='nowrite', action='store_true')
    parser.set_defaults(nowrite=False)
    parser.add_argument('--silent', dest='silent', action='store_true')
    parser.set_defaults(silent=False)
    parser.add_argument('--get_scale_length', dest='get_scale_length', action='store_true')
    parser.set_defaults(get_scale_length=False)
    parser.add_argument('--toscreen', dest='toscreen', action='store_true')
    parser.set_defaults(toscreen=False)
    parser.add_argument('--useKD', dest='useKD', action='store_true')
    parser.set_defaults(useKD=False)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--showfit', dest='showfit', action='store_true')
    parser.set_defaults(showfit=False)  # to show spectrum fitting plot
    parser.add_argument('--nomap', dest='nomap', action='store_true')
    parser.set_defaults(nomap=False)
    parser.add_argument('--showerr', dest='showerr', action='store_true')
    parser.set_defaults(showerr=False)
    parser.add_argument('--noskynoise', dest='noskynoise', action='store_true')
    parser.set_defaults(noskynoise=False)
    parser.add_argument('--showcoord', dest='showcoord', action='store_true')
    parser.set_defaults(showcoord=False)
    parser.add_argument('--nocreate', dest='nocreate', action='store_true')
    parser.set_defaults(nocreate=False)
    parser.add_argument('--snrmap', dest='snrmap', action='store_true')
    parser.set_defaults(snrmap=False)
    parser.add_argument('--snrhist', dest='snrhist', action='store_true')
    parser.set_defaults(snrhist=False)
    parser.add_argument('--binback', dest='binback', action='store_true')
    parser.set_defaults(binback=False)
    parser.add_argument('--showbin', dest='showbin', action='store_true')
    parser.set_defaults(showbin=False)
    parser.add_argument('--fitbin', dest='fitbin', action='store_true')
    parser.set_defaults(fitbin=False)
    parser.add_argument('--noweight', dest='noweight', action='store_true')
    parser.set_defaults(noweight=False)
    parser.add_argument('--show_annulus', dest='show_annulus', action='store_true')
    parser.set_defaults(show_annulus=False)
    parser.add_argument('--nooutlier', dest='nooutlier', action='store_true')
    parser.set_defaults(nooutlier=False)
    parser.add_argument('--contsub', dest='contsub', action='store_true')
    parser.set_defaults(contsub=False)
    parser.add_argument('--basic', dest='basic', action='store_true')
    parser.set_defaults(basic=False)
    parser.add_argument('--emulate_henry', dest='emulate_henry', action='store_true')
    parser.set_defaults(emulate_henry=False)
    parser.add_argument('--correct_old_units', dest='correct_old_units', action='store_true')
    parser.set_defaults(correct_old_units=False)
    parser.add_argument('--plot_1dkernel', dest='plot_1dkernel', action='store_true')
    parser.set_defaults(plot_1dkernel=False)

    parser.add_argument('--scale_exptime')
    parser.add_argument('--scale_exp_SNR')
    parser.add_argument('--multi_realisation')
    parser.add_argument("--path")
    parser.add_argument("--outfile")
    parser.add_argument("--file")
    parser.add_argument("--Om")
    parser.add_argument("--line")
    parser.add_argument("--ppb")
    parser.add_argument("--z")
    parser.add_argument("--res")
    parser.add_argument("--arc")
    parser.add_argument("--nhr")
    parser.add_argument("--nbin")
    parser.add_argument("--nres")
    parser.add_argument("--vdisp")
    parser.add_argument("--vdel")
    parser.add_argument("--vres")
    parser.add_argument("--X")
    parser.add_argument("--Y")
    parser.add_argument("--parm")
    parser.add_argument("--ker")
    parser.add_argument("--wmin")
    parser.add_argument("--wmax")
    parser.add_argument("--branch")
    parser.add_argument("--mergespec")
    parser.add_argument("--mergeHII")
    parser.add_argument("--cmin")
    parser.add_argument("--cmax")
    parser.add_argument("--snr_cmin")
    parser.add_argument("--snr_cmax")
    parser.add_argument("--rad")
    parser.add_argument("--gain")
    parser.add_argument("--exptime")
    parser.add_argument("--epp")
    parser.add_argument("--snr")
    parser.add_argument("--Zgrad")
    parser.add_argument("--fixed_SNR")
    parser.add_argument("--fixed_noise")
    parser.add_argument("--scale_SNR")
    parser.add_argument("--ncores")
    parser.add_argument("--galsize")
    parser.add_argument("--center")
    parser.add_argument("--outtag")
    parser.add_argument("--oneHII")
    parser.add_argument("--vmask")
    parser.add_argument("--scale_length")
    parser.add_argument("--xcenter_offset")
    parser.add_argument("--ycenter_offset")
    parser.add_argument("--binto")
    parser.add_argument("--fitupto")  # in units of args.scale_length
    parser.add_argument("--choice")  # color coding of metallicity gradient plot
    parser.add_argument("--which_mpirun")
    parser.add_argument("--add_text_to_plot")
    parser.add_argument("--target_res")
    parser.add_argument('--strehl')
    args, leftovers = parser.parse_known_args()
    if args.debug:  # debug mode over-rides
        args.toscreen = True
        args.silent = False
        args.calcgradient = True
        args.nowrite = True
        args.saveplot = False

    if args.debug: myprint('Starting in debugging mode. Brace for storm of stdout statements and plots...\n', args)
    if args.outtag is None: args.outtag = '_sph_logT4.0_MADtemp_Z0.05,5.0_age0.0,5.0_lnII5.0,12.0_lU-4.0,-1.0_4D'

    if args.galsize is not None:
        args.galsize = float(args.galsize)
    else:
        args.galsize = 30. # kpc

    if args.center is not None:
        args.center = float(args.galsize)
    else:
        args.center = 0.5*1310.72022072 # kpc units, from Goldbaum simulations in cell units

    if args.path is None:
        args.path = HOME + '/Desktop/bpt/'
    subprocess.call(['mkdir -p ' + args.path], shell=True)  # create output directory if it doesn't exist

    if args.file is None:
        args.file = 'DD0600_lgf'  # which simulation to use

    if args.Om is not None:
        args.Om = float(args.Om)
    else:
        args.Om = 0.5

    if args.ppb is not None:
        args.pix_per_beam = float(args.ppb)
    else:
        args.pix_per_beam = 6.

    if args.gain is not None:
        args.gain = float(args.gain)
    else:
        args.gain = 1.5

    if args.epp is not None:
        args.el_per_phot = float(args.epp)
    else:
        args.el_per_phot = 0.5  # instrumental throughput, not all photons get converted to electrons

    if args.z is not None:
        args.z = float(args.z)
    else:
        args.z = 0.013

    if args.rad is not None:
        args.rad = float(args.rad)
    else:
        args.rad = 5.  # in metres, for a 10m class telescope

    if args.res is not None:
        args.res = float(args.res)
    else:
        args.res = 0.02  # kpc: input resolution to constructPPV cube, usually (close to) the base resolution of simulation

    if args.arc is not None:
        args.res_arcsec = float(args.arc)
        if args.res_arcsec < 0:
            args.pix_per_beam = 0.001  # to convolve with an extremely small beam -> limiting case to no convolution
            required_res_phys = args.res * args.pix_per_beam  # physical resolution that 0.01 pixel corresponds to, so that no rebinning needs to be done before convolution
            args.res_arcsec = required_res_phys * (3600 * 180) / (
                    properties.dist * 1e3 * np.pi)  # back-calculating res in arcsec such that properties.res_phys = required_res_phys
            args.binto = None  # no resampling after convolution
    else:
        args.res_arcsec = 0.5  # arcsec

    properties.dist = calc_dist(args.z)  # distance to object; in Mpc

    properties.res_phys = args.res_arcsec * np.pi / (3600 * 180) * (properties.dist * 1e3)  # kpc

    if args.line is None:
        args.line = 'H6562'  # #whose emission map to be made

    if args.cmin is not None:
        args.cmin = float(args.cmin)
    else:
        args.cmin = None

    if args.cmax is not None:
        args.cmax = float(args.cmax)
    else:
        args.cmax = None

    if args.snr_cmin is not None:
        args.snr_cmin = float(args.snr_cmin)
    else:
        args.snr_cmin = None

    if args.snr_cmax is not None:
        args.snr_cmax = float(args.snr_cmax)
    else:
        args.snr_cmax = None

    if args.nhr is not None:
        args.nhr = int(args.nhr)
    else:
        args.nhr = 100  # no. of bins used to resolve the range lamda +/- 5sigma around emission lines

    if args.nbin is not None:
        args.nbin = int(args.nbin)
    else:
        args.nbin = 1000  # no. of bins used to bin the continuum into (without lines)

    if args.nres is not None:
        args.nres = int(args.nres)
    else:
        args.nres = 10  # no. of spectral resolution elements included on either side during fitting a line/group of lines

    if args.vdisp is not None:
        args.vdisp = float(args.vdisp)
    else:
        args.vdisp = 15.  # km/s vel dispersion to be added to emission lines from MAPPINGS while making PPV

    if args.vdel is not None:
        args.vdel = float(args.vdel)
    else:
        args.vdel = 100.  # km/s; vel range in which spectral resolution is higher is sig = 5*vdel/c
        # so wavelength range of +/- sig around central wavelength of line is binned into further nhr bins

    if args.vres is not None:
        args.vres = float(args.vres)
    else:
        args.vres = 30.  # km/s instrumental vel resolution to be considered while making PPV

    if args.wmin is not None:
        args.wmin = float(args.wmin)
    else:
        args.wmin = None  # Angstrom; starting wavelength of PPV cube

    if args.wmax is not None:
        args.wmax = float(args.wmax)
    else:
        args.wmax = None  # Angstrom; ending wavelength of PPV cube

    if args.snr is not None:
        args.SNR_thresh = float(args.snr)
    else:
        args.SNR_thresh = None

    if not args.keepprev:
        plt.close('all')

    if args.parm is not None:
        args.parm = [float(ar) for ar in args.parm.split(',')]
    else:
        args.parm = None  # set of parameters i.e. telescope properties to be used for smearing cube/map

    if args.Zgrad is not None:
        args.logOHcen, args.logOHgrad = [float(ar) for ar in args.Zgrad.split(',')]
        args.gradtext = '_Zgrad' + str(args.logOHcen) + ',' + str(args.logOHgrad)
    else:
        args.logOHcen, args.logOHgrad = logOHsun, 0.
        args.gradtext = ''
    args.outtag = args.gradtext + args.outtag

    if args.ker is not None:
        args.ker = args.ker
    else:
        args.ker = 'moff'  # convolution kernel to be used for smearing cube/map

    if args.scale_SNR is not None:
        args.scale_SNR = float(args.scale_SNR) # fine tune input SNR to get "good" values for output SNR
    else:
        args.scale_SNR = 1. # no fine tuning

    if args.fixed_SNR is not None:
        args.fixed_SNR = float(args.fixed_SNR)  # fixed SNR used in makenoisy() function
        args.fixed_SNR /= args.scale_SNR
    else:
        args.fixed_SNR = None

    if args.fixed_noise is not None:
        args.fixed_noise = float(args.fixed_noise)  # fixed SNR used in makenoisy() function
        args.fixed_noise /= args.scale_SNR
    else:
        args.fixed_noise = None

    if args.ncores is not None:
        args.ncores = int(args.ncores)  # number of cores used in parallel segments
    else:
        args.ncores = mp.cpu_count() / 2

    if args.oneHII is not None:
        args.oneHII = int(args.oneHII)  # serial no. of HII region to be used for the one HII region test case
    else:
        args.oneHII = None

    if args.vmask is not None:
        args.vmask = float(args.vmask)  # vel width to mask around nebular lines, before continuum fitting
    else:
        args.vmask = 100.  # km/s

    if args.useKD:
        args.num_arr = [['NII6584']]
        args.den_arr = [['OII3727', 'OII3729']]
        args.ratio_lim = [(0, 2)]
    else:
        args.num_arr = [['NII6584'], ['NII6584']]
        args.den_arr = [['SII6717', 'SII6730'], ['H6562']]
        args.ratio_lim = [(0, 2), (0, 0.5)]

    if args.scale_length is not None:
        args.scale_length = float(args.scale_length)  # scale_length of the disk galaxy
    else:
        args.scale_length = 4.  # kpc

    if args.xcenter_offset is not None:
        args.xcenter_offset = float(
            args.xcenter_offset)  # x coordinate of center of galaxy (in kpc), tp take into account any offset from centre of FoV
    else:
        args.xcenter_offset = 0.  # kpc

    if args.ycenter_offset is not None:
        args.ycenter_offset = float(
            args.ycenter_offset)  # x coordinate of center of galaxy (in kpc), tp take into account any offset from centre of FoV
    else:
        args.ycenter_offset = 0.  # kpc

    if args.which_mpirun is None:
        #args.which_mpirun = '/pkg/linux/anaconda/bin/mpirun' # mpirun to be used for avatar
        args.which_mpirun = 'mpirun' # mpirun to be used for raijin

    if args.strehl is None:
        args.strehl = 0.1 # Strehl ratio for AO assisted PSF
    else:
        args.strehl = float(args.strehl)

    if args.mergeHII is not None:
        args.mergeHII = float(args.mergeHII) # kpc, within which if two HII regions are they'll be treated as merged
    # -------------------------------------------------------------------------------------------
    args, logbook = getfitsname(args, properties)  # name of fits file to be written into
    properties.flux_ratio = np.pi / (4 * (3.086e21 * logbook.final_pix_size * 180 * 3600)**2) # converting emitting luminosity to luminosity seen from earth, 3.08e21 factor to convert kpc to cm
                                                                                              # multiplying this factor with ergs/s/A gives ergs/s/A/cm^2/arcsec^2
    if args.debug:
        myprint('Deb1272: Distance for z= ' + str(args.z) + ' is %.4F Mpc\n' % properties.dist, args)
        myprint('Deb1273: Flux ratio= (1. / (4 * np.pi * (dist= %.4E * 3.086e24)^2 * (arcsec= %.2F)^2) cm^-2 arcsec^-2' % (properties.dist, args.res_arcsec),
                args)
        myprint('Deb1274: Flux ratio= %.4E cm^-2 arcsec^-2' % properties.flux_ratio, args)
    # -------------------------------------------------------------------------------------------
    if args.outfile is None:
        args.outfile = logbook.fitsname.replace('PPV', 'output_PPV')[
                       :-5] + '.txt'  # name of fits file to be written into
    # ------------write starting conditions in output txt file or stdout-------------
    starting_text = ''
    if not len(sys.argv) > 1:
        starting_text += 'Insuffiecient information. Here is an example how to this routine might be called:\n'
        starting_text += 'run plotobservables.py --addnoise --smooth --keep --vres 600 --spec_smear --plotspec\n'
    else:
        if float(args.arc) < 0: starting_text += 'Input arcsec < 0 i.e. limit of no convolution. Hence convolving with tiny beam of ' + str(
            args.pix_per_beam) + \
                                            ' pixels. Avoiding pre-convolution binning, that corresponds to phys_res = ' + str(
            required_res_phys) + ' kpc and requires \
arcsec = ' + str(args.res_arcsec) + '". Also setting no resampling after convolution.' + '\n'
        starting_text += 'Path: ' + args.path + ' Use --path option to specify.' + '\n'
        starting_text += 'Outfile: ' + args.outfile + ' Use --outfile option to specify.' + '\n'
        starting_text += 'Simulation= ' + args.file + '. Use --file option to specify.' + '\n'
        starting_text += 'Omega= ' + str(
            args.Om) + '. Use --om option to specify Omega. You can supply , separated multiple omega values.' + '\n'
        starting_text += 'Maximum pix_per_beam of ' + str(
            args.pix_per_beam) + '. Use --ppb option to specify pix_per_beam.' + '\n'
        starting_text += 'Gain= ' + str(args.gain) + '. Use --gain option to specify gain.' + '\n'
        starting_text += 'Electrons per photon= ' + str(
            args.el_per_phot) + '. Use --epp option to specify el_per_phot.' + '\n'
        starting_text += 'Redshift of ' + str(args.z) + '. Use --z option to specify redshift.' + '\n'
        starting_text += 'Telescope mirror radius= ' + str(
            args.rad) + ' m. Use --rad option to specify radius in metres.' + '\n'
        starting_text += 'Simulation resolution= ' + str(
            args.res) + ' kpc. Use --res option to specify simulation resolution.' + '\n'
        starting_text += 'Telescope spatial resolution= ' + str(
            args.res_arcsec) + '. Use --arc option to specify telescope resolution.' + '\n'
        starting_text += 'Resolution of telescope on object frame turns out to be res_phys~' + str(
            properties.res_phys) + ' kpc.' + '\n'
        if args.exptime:
            starting_text += 'Exposure time set to ' + str(
                logbook.exptime) + ' seconds. Use --exptime option to specify absolute exposure time in seconds.' + '\n'
        else:
            starting_text += 'Exposure time scaled to ' + str(
                logbook.exptime) + ' seconds. Use --scale_exptime option to specify scale factor in seconds.' + '\n'
        starting_text += 'Line:' + args.line + '. Use --line option to specify line.' + '\n'
        starting_text += 'No. of bins used to resolve+/- 5sigma around emission lines= ' + str(
            args.nhr) + '. Use --nhr to specify.' + '\n'
        starting_text += 'No. of bins used to bin the continuum into (without lines)= ' + str(
            args.nbin) + '. Use --nbin to specify.' + '\n'
        starting_text += 'Velocity dispersion to be added to emission lines= ' + str(
            args.vdisp) + ' km/s.' + '. Use --vdisp to specify.' + '\n'
        starting_text += 'Velocity range in which spectral resolution is higher around central wavelength of line= ' + str(
            args.vdel) + ' km/s.' + '. Use --vdel to specify.' + '\n'
        starting_text += 'Instrumental velocity resolution to be considered while making PPV= ' + str(
            args.vres) + ' km/s.' + '. Use --vres to specify.' + '\n'
        if args.wmin:
            starting_text += 'Starting wavelength of PPV cube= ' + str(args.wmin) + ' A.' + '\n'
        else:
            starting_text += 'Starting wavelength of PPV cube at beginning of line list.' + '\n'
        if args.wmax:
            starting_text += 'Ending wavelength of PPV cube= ' + str(args.wmax) + ' A.' + '\n'
        else:
            starting_text += 'Ending wavelength of PPV cube at end of line list.' + '\n'
        if args.snr:
            starting_text += 'Applying SNR cut-off= ' + str(args.SNR_thresh) + ' on fitted lines.' + '\n'
        else:
            starting_text += 'No SNR cut-off will be applied.' + '\n'
        starting_text += 'Will run the parallel segments on ' + str(args.ncores) + ' cores.' + '\n'
        if args.smooth:
            if args.parm:
                starting_text += 'Parameter for smoothing= ' + str(args.parm[0]) + ', ' + str(args.parm[1]) + '\n'
            else:
                starting_text += 'Default smoothing parameter settings. Use --parm option to specify smearing parameters set.' + '\n'
            if args.ker:
                starting_text += 'Smoothing profile used: ' + args.ker + '\n'
            else:
                starting_text += 'Default Moffat profile for smoothing.' + '\n'
        if args.Zgrad:
            starting_text += 'Using metallicity painted HII regions, with central logOH+12 = ' + str(
                args.logOHcen) + ', and gradient = ' + str(args.logOHgrad) + ' dex per kpc' + '\n'
        else:
            starting_text += 'No additional metallicity gradient painted.' + '\n'
        if args.mergeHII:
            starting_text += 'Will be merging HII regions within' +str(args.mergeHII)+ 'kpc.' + '\n'
        starting_text += '\n'
        starting_text += 'Will be using/creating ' + logbook.H2R_filename + ' file.' + '\n'
        starting_text += 'Will be using/creating ' + logbook.skynoise_cubename + ' file.' + '\n'
        starting_text += 'Will be using/creating ' + logbook.convolved_filename + ' file.' + '\n'
        starting_text += 'Will be using/creating ' + logbook.fitsname + ' file.' + '\n'
        if args.emulate_henry:
            if args.target_res is not None: starting_text += 'Will be emulating procedure of Henry with new resolution of '+args.target_res+' pc.' + '\n'
            else: starting_text += 'Will be emulating procedure of Henry but keeping resolution same.' + '\n'
    myprint(starting_text, args)
    # -------------------------------------------------------------------------------------------------------
    logbook.s = pd.read_table(getfn(args), comment='#', delim_whitespace=True)
    initial_nh2r = len(logbook.s)
    if args.nooutlier: logbook.s = logbook.s[(logbook.s['nII'] >= 10 ** (5.2 - 4 + 6)) & (logbook.s['nII'] <= 10 ** (
            6.7 - 4 + 6))]  # D16 models have 5.2 < lpok < 6.7 #DD600_lgf with Zgrad8.77,-0.1 has 4.3 < lpok < 8
    if args.mergeHII:
        logbook.s = merge_HIIregions(logbook.s, args) # merging HII regions within args.mergeHII kpc distance
    myprint('Using ' + str(len(logbook.s)) + ' out of ' + str(initial_nh2r) + ' HII regions.\n', args)
    myprint('Deb1432: starting shape, intended res_phy, final pix size, final shape= '+str(args.galsize / args.res)+', '+str(properties.res_phys)+', '+str(logbook.final_pix_size)+' kpc, '+str(args.galsize / logbook.final_pix_size)+'\n', args)  #
    myprint('Deb1433:  final pix per beam to smooth, final pix per beam sampled, intermediate pix size, binning factor, intermediate shape= '+str(logbook.fwhm)+', '+str(properties.res_phys / logbook.final_pix_size)+', '+str(logbook.intermediate_pix_size)+' kpc, '+str(logbook.intermediate_pix_size / args.res)+', '+str(args.galsize / logbook.intermediate_pix_size)+'\n', args)
    #sys.exit() #
    if args.debug: mydiag('Deb1437: for H2R Ha luminosity: in ergs/s:', logbook.s['H6562'], args)
    # -----------------------jobs fetched--------------------------------------------------------------------
    if args.get_scale_length:
        properties.scale_length = get_scale_length(args, logbook)
    elif args.ppv and args.mergespec is None:
        properties.ppvcube = spec(args, logbook, properties)
    else:
        if args.mergespec is None:
            if os.path.exists(logbook.fitsname):
                myprint('Reading existing ppvcube from ' + logbook.fitsname + '\n', args)
                properties.ppvcube = fits.open(logbook.fitsname)[0].data
                if not os.path.exists(logbook.fitsname_u):
                    if not args.nocreate:
                        properties.ppvcube = spec(args, logbook, properties)
                    else:
                        myexit(args, text='ERROR file does not exist and not creating it. Just exiting.' + '\n')
                properties.ppvcube_u = fits.open(logbook.fitsname_u)[0].data
            elif not args.nocreate:
                myprint('PPV file does not exist. Creating ppvcube..' + '\n', args)
                properties.ppvcube = spec(args, logbook, properties)
            else:
                myexit(args, text='PPV file does not exist and not creating it. Just exiting.' + '\n')
            properties = get_disp_array(args, logbook, properties)
        else:
            properties_orig = deepcopy(properties)
            args_orig = deepcopy(args)
            mergespec = args.mergespec.split(',')
            nbranches = len(mergespec) / 3  # args
            for i in range(nbranches):
                args_dummy = deepcopy(args_orig)
                args_dummy.branch = mergespec[3 * i]
                args_dummy.wmin = float(mergespec[3 * i + 1])
                args_dummy.wmax = float(mergespec[3 * i + 2])
                properties_dummy = deepcopy(properties_orig)
                args_dummy, logbook_dummy = getfitsname(args_dummy, properties_dummy)
                logbook_dummy.s = logbook.s
                if os.path.exists(logbook_dummy.fitsname):
                    myprint('Reading existing ' + args_dummy.branch + ' ppvcube from ' + logbook_dummy.fitsname + '\n',
                        args_dummy)
                elif not args.nocreate:
                    myprint('PPV file does not exist. Creating ppvcube..' + '\n', args_dummy)
                    properties_dummy.ppvcube = spec(args_dummy, logbook_dummy, properties_dummy)
                else:
                    myexit(args_dummy, args_dummy.branch + ' ppvcube does not exist and not creating it.' + '\n')

                properties_dummy.ppvcube = fits.open(logbook_dummy.fitsname)[0].data
                properties_dummy.ppvcube_u = fits.open(logbook_dummy.fitsname_u)[0].data
                properties_dummy = get_disp_array(args_dummy, logbook_dummy, properties_dummy)
                properties.dispsol = np.hstack([properties.dispsol, properties_dummy.dispsol]) if i else properties_dummy.dispsol
                properties.ppvcube = np.dstack([properties.ppvcube, properties_dummy.ppvcube]) if i else properties_dummy.ppvcube
                properties.ppvcube_u = np.dstack([properties.ppvcube_u, properties_dummy.ppvcube_u]) if i else properties_dummy.ppvcube_u
                logbook.llist = np.hstack([logbook.llist, logbook_dummy.llist]) if i else logbook_dummy.llist
                logbook.wlist = np.hstack([logbook.wlist, logbook_dummy.wlist]) if i else logbook_dummy.wlist
                if args_dummy.wmin < args.wmin: args.wmin = args_dummy.wmin
                if args_dummy.wmax > args.wmax: args.wmax = args_dummy.wmax


    if (not args.plotintegmap) * (not args.plotspec) * (not args.fixfit) * (not args.fixcont) == 0: # if no need to create mapcube
        if args.X is not None:
            args.X = int(args.X)
        else:
            args.X = int((logbook.s['x(kpc)'][args.oneHII] - args.center + args.galsize/2) / args.res) if args.oneHII is not None else int(
                args.galsize / logbook.final_pix_size) / 2  # p-p values at which point to extract spectrum from the ppv cube
        if args.plotspec and not args.silent: myprint(
            'X position at which spectrum to be plotted= ' + str(args.X) + '\n', args)

        if args.Y is not None:
            args.Y = int(args.Y)
        else:
            args.Y = int((logbook.s['y(kpc)'][args.oneHII] - args.center + args.galsize/2) / args.res) if args.oneHII is not None else int(
                args.galsize / logbook.final_pix_size) / 2  # p-p values at which point to extract spectrum from the ppv cube
        if args.plotspec and not args.silent: myprint(
            'Y position at which spectrum to be plotted= ' + str(args.Y) + '\n', args)

        if args.plotintegmap:
            dummy = plotintegmap(args, logbook, properties)
        elif args.plotspec:
            dummy = spec_at_point(args, logbook, properties)
        elif args.fixfit or args.fixcont:
            dummy, dummy_u = fixfit(args, logbook, properties)

    else: # need to perform operations on a mapcube
        if args.mergespec is None: # if dealing with only one spectral branch and not needed to mergespec two branches
            mergespec = [args.branch if args.branch is not None else '', str(logbook.wmin), str(logbook.wmax)]
        else: # if need to combine two or more spectral branches of datacubes
            mergespec = args.mergespec.split(',')
        nbranches = len(mergespec)/3 # args
        properties_orig = deepcopy(properties)
        args_orig = deepcopy(args)
        for i in range(nbranches):
            args_dummy = deepcopy(args_orig)
            args_dummy.branch = mergespec[3 * i]
            args_dummy.wmin = float(mergespec[3 * i + 1])
            args_dummy.wmax = float(mergespec[3 * i + 2])
            properties_dummy = deepcopy(properties_orig)
            args_dummy, logbook_dummy = getfitsname(args_dummy, properties_dummy)
            if args.emulate_henry:
                if float(args_dummy.target_res) == logbook_dummy.final_pix_size*1e3 and os.path.exists(logbook_dummy.fittedcube):# no rebinning required
                    mapcube = fits.open(logbook_dummy.fittedcube)[0].data
                    errorcube = fits.open(logbook_dummy.fittederror)[0].data
                args_dummy, logbook_dummy = emulate_henry_getcube(args_dummy, logbook_dummy)
                properties_dummy.ppvcube = fits.open(logbook_dummy.fitsname)[0].data
                properties_dummy.ppvcube_u = fits.open(logbook_dummy.fitsname_u)[0].data
                properties.ppvcube = np.dstack([properties.ppvcube, properties_dummy.ppvcube]) if i else properties_dummy.ppvcube
                properties.ppvcube_u = np.dstack([properties.ppvcube_u, properties_dummy.ppvcube_u]) if i else properties_dummy.ppvcube_u
                if args_dummy.target_res == logbook_dummy.final_pix_size*1e3:# no rebinning required
                    myprint('Emulate Henry: No rebinning necessary as per target_res, hence simply re-saving mapcube.\n', args_dummy)
                    write_fits(logbook_dummy.fittedcube, mapcube, args_dummy)
                    write_fits(logbook_dummy.fittederror, errorcube, args_dummy)
                logbook.final_pix_size = args.galsize/np.shape(properties.ppvcube)[0]

            if os.path.exists(logbook_dummy.fittedcube) and not args_dummy.clobber:
                myprint('Reading existing mapcube from ' + logbook_dummy.fittedcube + '\n', args_dummy)
            elif not args_dummy.nocreate:
                myprint('Mapfile does not exist. Creating mapcube..' + '\n', args_dummy)

                if args_dummy.spec_smear:
                    smear = ' --spec_smear'
                else:
                    smear = ''
                if args_dummy.silent:
                    silent = ' --silent'
                else:
                    silent = ''
                if args_dummy.toscreen:
                    toscreen = ' --toscreen'
                else:
                    toscreen = ''
                if args_dummy.debug:
                    debug = ' --debug'
                else:
                    debug = ''
                if args_dummy.showfit:
                    showfit = ' --showfit'
                else:
                    showfit = ''
                if args_dummy.oneHII is not None:
                    oneHII = ' --oneHII ' + str(args_dummy.oneHII)
                else:
                    oneHII = ''
                if args_dummy.addnoise:
                    addnoise = ' --addnoise'
                else:
                    addnoise = ''
                if args_dummy.contsub:
                    contsub = ' --contsub'
                else:
                    contsub = ''

                funcname = HOME + WD + 'parallel_fitting.py'
                command = args_dummy.which_mpirun+' -np ' + str(args_dummy.ncores) + ' python ' + funcname + ' --fitsname ' + logbook_dummy.fitsname + \
                          ' --no_noise_fitsname ' + logbook_dummy.no_noise_fitsname + ' --fitsname_u ' + logbook_dummy.fitsname_u + ' --nbin ' + str(
                    args_dummy.nbin) + \
                          ' --vdel ' + str(args_dummy.vdel) + ' --vdisp ' + str(args_dummy.vdisp) + ' --vres ' + str(
                    args_dummy.vres) + ' --nhr ' + str(args_dummy.nhr) + ' --wmin ' + \
                          str(logbook_dummy.wmin) + ' --wmax ' + str(
                    logbook_dummy.wmax) + ' --fittedcube ' + logbook_dummy.fittedcube + ' --fittederror ' + logbook_dummy.fittederror + \
                          ' --outfile ' + args_dummy.outfile + ' --nres ' + str(args_dummy.nres) + ' --vmask ' + str(
                    args_dummy.vmask) + smear + silent + toscreen \
                          + debug + showfit + oneHII + addnoise + contsub
                subprocess.call([command], shell=True)
            else:
                myprint('Mapfile does not exist and not creating it. Just exiting.' + '\n', args_dummy)
                sys.exit()

            properties_dummy.mapcube = fits.open(logbook_dummy.fittedcube)[0].data
            properties_dummy.errorcube = fits.open(logbook_dummy.fittederror)[0].data
            properties.mapcube = np.dstack([properties.mapcube, properties_dummy.mapcube]) if i else properties_dummy.mapcube
            properties.errorcube = np.dstack([properties.errorcube, properties_dummy.errorcube]) if i else properties_dummy.errorcube
            if args_dummy.wmin < args.wmin: args.wmin = args_dummy.wmin
            if args_dummy.wmax > args.wmax: args.wmax = args_dummy.wmax

        if args.emulate_henry:
            s2a_ind = np.where(logbook.llist == 'SII6717')[0][0]
            ha_ind = np.where(logbook.llist == 'H6562')[0][0]
            s2a_map = properties.mapcube[:,:,s2a_ind]
            ha_map = properties.mapcube[:,:,ha_ind]
            ratio_map = s2a_map/ha_map
            isdig = np.zeros(np.shape(properties.mapcube)) # to make a 3D mask
            isdig[:,:,:] = ratio_map[:,:,np.newaxis] >= 0.3 # to mask out DIG based on SII/Ha criteria
            properties.mask_ar = [isdig, np.logical_not(isdig), np.zeros(np.shape(properties.mapcube))]
            properties.text_ar = ['onlyHII', 'onlyDIG', 'HII+DIG']
            properties_orig = deepcopy(properties)
            if args.map:
                for i in range(3):
                    properties.mapcube = properties_orig.mapcube
                    properties.errorcube = properties_orig.errorcube
                    properties.mapcube = np.ma.masked_where(properties.mask_ar[i], properties.mapcube)
                    properties.errorcube = np.ma.masked_where(properties.mask_ar[i], properties.errorcube)
                    dummy = emissionmap(args, logbook, properties, additional_text=properties.text_ar[i])
            elif args.met and args.inspect:
                for i in range(3):
                    properties.mapcube = properties_orig.mapcube
                    properties.errorcube = properties_orig.errorcube
                    properties.mapcube = np.ma.masked_where(properties.mask_ar[i], properties.mapcube)
                    properties.errorcube = np.ma.masked_where(properties.mask_ar[i], properties.errorcube)
                    args.useKD = False
                    args.num_arr = [['NII6584'], ['NII6584']]
                    args.den_arr = [['SII6717', 'SII6730'], ['H6562']]
                    args.ratio_lim = [(0, 2), (0, 0.5)]
                    properties, axes = metallicity(args, logbook, properties)
                    inspectmap(args, logbook, properties, axes=axes, additional_text=properties.text_ar[i])
                    if args.mergespec is not None:
                        args.useKD = True
                        args.num_arr = [['NII6584']]
                        args.den_arr = [['OII3727', 'OII3729']]
                        args.ratio_lim = [(0, 2)]
                        properties, axes = metallicity(args, logbook, properties)
                        inspectmap(args, logbook, properties, axes=axes, additional_text=properties.text_ar[i])
            elif args.met:
                emulate_henry_getmet(args, logbook, properties)

        elif args.bptpix:
            bpt_pixelwise(args, logbook, properties)
        elif args.met:
            args.useKD = False
            args.num_arr = [['NII6584'], ['NII6584']]
            args.den_arr = [['SII6717', 'SII6730'], ['H6562']]
            args.ratio_lim = [(0, 2), (0, 0.5)]
            properties, axes = metallicity(args, logbook, properties)
            if args.inspect: inspectmap(args, logbook, properties, axes=axes)
            if args.mergespec is not None:
                args.useKD = True
                args.num_arr = [['NII6584']]
                args.den_arr = [['OII3727', 'OII3729']]
                args.ratio_lim = [(0, 2)]
                properties, axes = metallicity(args, logbook, properties)
                if args.inspect: inspectmap(args, logbook, properties, axes=axes)
        elif args.map:
            properties.map = emissionmap(args, logbook, properties)
        elif args.sfr:
            properties.SFRmap_real, properties.SFRmapHa = SFRmaps(args, logbook, properties)
        else:
            myprint(
                'Wrong choice. Choose from:\n --bptpix, --map, --sfr, --met, --ppv, --plotinteg, --plotspec' + '\n',
                args)
    # -------------------------------------------------------------------------------------------
    if args.hide:
        plt.close()
    else:
        plt.show(block=False)
    myprint('Completed in %s minutes\n' % ((time.time() - start_time) / 60), args)
