import time
start_time = time.time()
import datetime
import numpy as np
import subprocess
from matplotlib import pyplot as plt
from astropy.io import ascii, fits
import os
HOME = os.getenv('HOME')
import sys
sys.path.append(HOME+'/Work/astro/ayan_codes/mageproject/ayan/')
sys.path.append(HOME+'/Work/astro/ayan_codes/enzo_model_code/')
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
from scipy import asarray as ar,exp
#------------Reading pre-defined line list-----------------------------
def readlist():
    target = []
    finp = open(HOME+'/Mappings/lab/targetlines.txt','r')
    l = finp.readlines()
    for lin in l:
        if len(lin.split())>1 and lin.split()[0][0] != '#':
            target.append([float(lin.split()[0]),lin.split()[1]])
    finp.close()
    target = sorted(target, key=itemgetter(0))
    target = np.array(target)
    llist = np.array(target)[:,1]
    wlist = np.asarray(target[:,0], float)
    return wlist, llist
#-------------------------------------------------------------------------------------------
def Ke13b(x, z):
    return 0.61/(x - 0.02 - 0.1833*z) + 1.2 + 0.03*z #Kewley2013b
#-------------------------------------------------------------------------------------------
def Ke13a(x):
    return 0.61/(x + 0.08) + 1.1 #Kewley2013a
#-------------------------------------------------------------------------------------------
def Ke01(x):
    return 0.61/(x - 0.47) + 1.19 #Kewley2001
#-------------------------------------------------------------------------------------------
def Ka03(x):
    return 0.61/(x - 0.05) + 1.3 #Kauffmann2003
#-------------------------------------------------------------------------------------------
def title(fn):
    if fn == 'DD0600_lgf':
        return 'Low gas fraction (0.1): after 600Myr\n'
    elif fn == 'DD0600':
        return 'Medium gas fraction (0.2): after 600Myr\n'
    elif fn == 'DD0300_hgf':
        return 'High gas fraction (0.4): after 300Myr\n'
    else:
        return ''
#-------------------------------------------------------------------------------------------
def plottheory():
    x = np.linspace(-1.3, -0.4, 1000)
    #plt.plot(x, Ke13a(x), linestyle='solid', label='Kewley 2013a')
    plt.plot(x, Ka03(x), linestyle='dashed', label='Kauffmann 2003')
    plt.plot(x, Ke01(x), linestyle='dotted', label='Kewley 2001')
    plt.legend(bbox_to_anchor=(0.45, 0.23), bbox_transform=plt.gcf().transFigure)
#-------------------------------------------------------------------------------------------------
def get_erf(lambda_array, height, centre, width, delta_lambda):
    return np.sqrt(np.pi/2) * height * width * ( erf((centre + delta_lambda/2 - lambda_array)/(np.sqrt(2)*width)) - \
    erf((centre - delta_lambda/2 - lambda_array)/(np.sqrt(2)*width)) ) / delta_lambda #https://www.wolframalpha.com/input/?i=integrate+a*exp(-(x-b)%5E2%2F(2*c%5E2))*dx+from+(w-d%2F2)+to+(w%2Bd%2F2)
#-------------------------------------------------------------------------------------------------
def gauss(w, f, w0, f0, v, vz):
    w0 = w0*(1+vz/c) #shift central wavelength wrt w0 due to LOS vel v_z of HII region as compared to systemic velocity
    sigma = w0*v/c #c=3e5 km/s
    A = f0/np.sqrt(2*np.pi*sigma**2) #height of Gaussian, such that area = f0
    dw = w[np.where(w>=w0)[0][0]] - w[np.where(w>=w0)[0][0] - 1]
    g = get_erf(w, A, w0, sigma, dw)
    #g = A*np.exp(-((w-w0)**2)/(2*sigma**2))    
    if args.oneHII is not None: print 'Debugging76: input gaussian parm (ergs/s/A/pc^2) =', f[0]/(args.res*1e3)**2, \
    (f0/np.sqrt(2*np.pi*sigma**2))/(args.res*1e3)**2, w0, sigma #
    f += g
    return f
#-------------------------------------------------------------------------------------------
def fixcont_gaus(x,cont,n,*p):
    result = cont
    for xx in range(0,n):
        result += p[3*xx+0]*exp(-((x-p[3*xx+1])**2)/(2*p[3*xx+2]**2))
    return result
#-------------------------------------------------------------------------------------------
def fixcont_erf(x,cont,n,*p):
    result = cont
    for xx in range(0,n):
        dw = x[np.where(x>=p[3*xx+1])[0][0]] - x[np.where(x>=p[3*xx+1])[0][0] - 1]
        result += get_erf(x, p[3*xx+0], p[3*xx+1], p[3*xx+2], dw)
    return result
#-------------------------------------------------------------------------------------------
def bpt_pixelwise(args, logbook, properties):
    global info
    fig = plt.figure(figsize=(8,6))
    fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.1, right=0.95)
    ax = plt.subplot(111)
    if args.theory: plottheory()
    plt.ylabel('Log(O[III] 5007/Hb)')
    plt.xlabel('Log(N[II]6584/Ha)')
    if args.gridoverlay:
        plt.xlim(-3.5,0.5)
        plt.ylim(-2.5,1.5)
        gridoverlay(annotate=args.annotategrid)
    else:
        plt.xlim(-1.4,-0.2)
        plt.ylim(-1.5,1.5)
    mapn2 = properties.mapcube[:,:,np.where(logbook.llist == 'NII6584')[0][0]]
    mapo3 = properties.mapcube[:,:,np.where(logbook.llist == 'OIII5007')[0][0]]
    mapha = properties.mapcube[:,:,np.where(logbook.llist == 'H6562')[0][0]]
    maphb = properties.mapcube[:,:,np.where(logbook.llist == 'Hbeta')[0][0]]

    if args.SNR_thresh is not None:
        #loading all the flux uncertainties
        mapn2_u = properties.errorcube[:,:,np.where(args.llist == 'NII6584')[0][0]]
        mapo3_u = properties.errorcube[:,:,np.where(args.llist == 'OIII5007')[0][0]]
        mapha_u = properties.errorcube[:,:,np.where(args.llist == 'H6562')[0][0]]
        maphb_u = properties.errorcube[:,:,np.where(args.llist == 'Hbeta')[0][0]]
        mapn2_u = np.ma.masked_where(mapn2_u<=0., mapn2_u)
        mapo3_u = np.ma.masked_where(maps2a_u<=0., mapo3_u)
        mapha_u = np.ma.masked_where(maps2b_u<=0., mapha_u)
        maphb_u = np.ma.masked_where(mapha_u<=0., maphb_u)        
        #imposing SNR cut
        mapn2 = np.ma.masked_where(mapn2/mapn2_u <SNR_thresh, mapn2)
        mapo3 = np.ma.masked_where(mapo3/mapo3_u <SNR_thresh, mapo3)
        mapha = np.ma.masked_where(mapha/mapha_u <SNR_thresh, mapha)
        maphb = np.ma.masked_where(maphb/maphb_u <SNR_thresh, maphb)

    mapn2ha = np.divide(mapn2,mapha)
    mapo3hb = np.divide(mapo3,maphb)
    t = title(fn)+' Galactrocentric-distance color-coded, BPT of model \n\
for Omega = '+str(args.Om)+', resolution = '+str(args.res)+' kpc'+info
    plt.scatter((np.log10(mapn2ha)).flatten(),(np.log10(mapo3hb)).flatten(), s=4, c=d.flatten(), lw=0,vmin=0,vmax=args.galsize/2)
    plt.title(t)
    cb = plt.colorbar()
    #cb.ax.set_yticklabels(str(res*float(x.get_text())) for x in cb.ax.get_yticklabels())
    cb.set_label('Galactocentric distance (in kpc)')
    if args.saveplot:
        fig.savefig(path+t+'.png')

#-------------------------------------------------------------------------------------------
def iterable(item):
    try:
        iter(item)
        return True
    except:
        return False
#-------------------------------------------------------------------------------------------
def meshplot2D(x,y):
    for i in range(0,np.shape(x)[0]):
        plt.plot(x[i,:],y[i,:], c='red')
    for i in range(0,np.shape(x)[1]):
        plt.plot(x[:,i],y[:,i], c='blue')
    plt.title(title(fn)+' BPT of models')

#-------------------------------------------------------------------------------------------
def meshplot3D(x,y, args):
    age_arr = np.linspace(0.,5.,6) #in Myr
    lnII = np.linspace(5., 12., 6) #nII in particles/m^3
    lU = np.linspace(-4.,-1., 4) #dimensionless
    n = 3
    for k in range(0,np.shape(x)[2]):
        for i in range(0,np.shape(x)[0]):
            plt.plot(x[i,:,k],y[i,:,k], c='red', lw=0.5)
            if args.annotate and k==n: ax.annotate(str(age_arr[i])+' Myr', xy=(x[i,-1,k],y[i,-1,k]), \
            xytext=(x[i,-1,k]-0.4,y[i,-1,k]-0), color='red', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3",color='red'))
            #plt.pause(1) #
        for i in range(0,np.shape(x)[1]):
            plt.plot(x[:,i,k],y[:,i,k], c='blue', lw=0.5)
            if args.annotate and k==n: ax.annotate('log nII= '+str(lnII[i]), xy=(x[-2,i,k],y[-2,i,k]), \
            xytext=(x[-2,i,k]+0.,y[-2,i,k]-0.4),color='blue',fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3",color='blue'))
            #plt.pause(1) #
        if args.annotate: ax.annotate('log U= '+str(lU[k]), xy=(x[5,-1,k],y[5,-1,k]), \
            xytext=(x[5,-1,k]-0.6,y[5,-1,k]-1.),color='black',fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3",color='black'))
    
    for i in range(0,np.shape(x)[0]):
        for j in range(0, np.shape(x)[1]):
            plt.plot(x[i,j,:],y[i,j,:], c='black', lw=0.5)
            #plt.pause(1) #
    
    plt.title('MAPPINGS grid of models')

#-------------------------------------------------------------------------------------------
def gridoverlay(args):
    s = ascii.read(HOME+'/Mappings/lab/totalspec.txt',comment='#',guess=False)
    y = np.reshape(np.log10(np.divide(s['OIII5007'],s['HBeta'])),(6, 6, 4))
    x = np.reshape(np.log10(np.divide(s['NII6584'],s['H6562'])),(6, 6, 4))
    x[[3,4],:,:]=x[[4,3],:,:] #for clearer connecting lines between grid points
    y[[3,4],:,:]=y[[4,3],:,:] #
    plt.scatter(x,y, c='black', lw=0, s=2)
    meshplot3D(x,y, args)
    if args.saveplot:
        fig.savefig(args.path+args.file+':BPT overlay')
#----------Function to measure scale length of disk---------------------------------------------------------------------------------
def knee(r, r_s, alpha):
    return r**(1./alpha) - np.exp(r/r_s)

def powerlaw(x, alpha):
    return x**(1/alpha)

def exponential(x, r_s):
    return np.exp(-x/r_s)

def func(r, r_s, alpha):
    from scipy.integrate import quad
    quad = np.vectorize(quad)
    r_knee = fminbound(knee,0,3,args=(r_s,alpha)) #0.
    if args.toscreen: print 'deb166:r_s, alpha, r_knee=', r_s, alpha, r_knee, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    y = np.zeros(len(r))
    y += quad(powerlaw, 0., r, args=(alpha,))[0] * (r <= r_knee)
    y += (quad(powerlaw, 0., r_knee, args=(alpha,))[0] + quad(exponential, r_knee, r, args=(r_s,))[0]) * (r > r_knee)
    y /= (quad(powerlaw, 0., r_knee, args=(alpha,))[0] + quad(exponential, r_knee, np.inf, args=(r_s,))[0])
    return y

def get_scale_length(args, logbook):
    start = time.time()
    g,x,y = calcpos(logbook.s, args.galsize, args.res)
    
    d_list = np.sqrt((x-args.galsize/2)**2 + (y-args.galsize/2)**2) #kpc
    ha_list = [x for (y,x) in sorted(zip(d_list,logbook.s['H6562']), key=lambda pair: pair[0])]
    d_list = np.sort(d_list)
    ha_cum = np.cumsum(ha_list)/np.sum(ha_list)
    popt, pcov = curve_fit(func, d_list, ha_cum, p0 = [4,4], bounds = ([1,0.01], [7,8]))
    scale_length, alpha = popt
    if not args.hide:
        fig = plt.figure()
        plt.scatter(d_list, ha_cum, s=10, lw=0, c='b', label='pixels')
        plt.plot(d_list, func(d_list,*popt), c='r',label='Fit with scale length=%.2F kpc, pow=%.2F'%(scale_length, alpha))
        ylab = 'Cumulative Halpha luminosity'
    '''
    ha_map = make2Dmap(logbook.s['H6562']/res**2, x, y, g, res)
    ha_map = np.ma.masked_where(ha_map<=0., ha_map)
    b = np.linspace(-g/2 + 1,g/2,g)*(galsize)/g #in kpc
    d = np.sqrt(b[:,None]**2+b**2)
    ha_list = np.log(ha_map.flatten())
    d_list = d.flatten()
    x_arr = np.arange(0,galsize/2,0.1)
    digitized = np.digitize(d_list, x_arr)    
    ha_list = [ha_list[digitized == i].mean() for i in range(1, len(x_arr))]
    d_list = [d_list[digitized == i].mean() for i in range(1, len(x_arr))]
    ha_list = np.ma.masked_where(np.isnan(ha_list), ha_list)
    d_list = np.ma.masked_array(d_list, ha_list.mask)
    ha_list = np.ma.compressed(ha_list)
    d_list = np.ma.compressed(d_list)            
    linefit = np.polyfit(d_list, ha_list, 1)
    properties.scale_length = -1./linefit[0] #kpc
    if not args.hide:
        fig = plt.figure()
        plt.scatter(d_list,ha_list, s=10, lw=0, c='b', label='pixels')
        plt.plot(x_arr, np.poly1d(linefit)(x_arr), c='r',label='Inferred Ha gradient')
        plt.axhline(-1+linefit[1], linestyle='--', c='k')
        plt.axvline(properties.scale_length, linestyle='--', c='k', label='scale_length=%.2F kpc'%scale_length)
        ylab = 'log Halpha surface brightness'
    '''
    if not args.hide:
        plt.xlabel('Galactocentric distance (kpc)')
        plt.ylabel(ylab)
        plt.xlim(0,galsize/2)
        plt.legend(bbox_to_anchor=(0.9, 0.42), bbox_transform=plt.gcf().transFigure)
        plt.title('Measuring star formation scale length')
        plt.show(block=False)
    
    output = 'Scale length from Halpha map = '+str(scale_length)+' kpc, in %.2F min\n'%((time.time()-start)/60.)
    myprint(output, args)
    return scale_length
#-------------From Dopita 2016------------------------------------------------------------------------------
def get_D16_met(map_num_series, map_den_series):
    mapn2 = map_num_series[0]
    maps2, mapha = map_den_series[0], map_den_series[1]
    logOHsol = 8.77 #log(O/H)+12 value used for Solar metallicity in MAPPINGS-V, Dopita 2016
    log_ratio = np.log10(np.divide(mapn2,maps2)) + 0.264*np.log10(np.divide(mapn2,mapha))
    logOHobj_map = log_ratio + 0.45*(log_ratio + 0.3)**5 # + 8.77
    return logOHsol, logOHobj_map
    
#-------------From Kewley 2002------------------------------------------------------------------------------
def get_KD02_met(map_num_series, map_den_series):
    mapn2 = map_num_series[0]
    mapo2 = map_den_series[0]
    logOHsol = 8.93 #log(O/H)+12 value used for Solar metallicity in earlier MAPPINGS in 2001
    log_ratio = np.log10(np.divide(mapn2,mapo2))
    logOHobj_map = np.log10(1.54020 + 1.26602*log_ratio + 0.167977*log_ratio**2) #+ 8.93
    return logOHsol, logOHobj_map

#------------Function to measure metallicity-------------------------------------------------------------------------------
def metallicity(args, logbook, properties, axes=None):
    global info
    g = np.shape(properties.mapcube)[0]
    b = np.linspace(-g/2 + 1,g/2,g)*(args.galsize)/g #in kpc
    d = np.sqrt(b[:,None]**2+b**2)
    t = args.file+':Met_Om'+str(args.Om)
    if args.smooth: t += '_arc'+str(args.res_arcsec)+'"'
    if args.spec_smear: t+= '_vres='+str(args.vres)+'kmps_'
    t += info+args.gradtext
    if args.SNR_thresh is not None: t += '_snr'+str(args.SNR_thresh)

    nnum, nden = len(np.unique(args.num_arr)), len(np.unique(args.den_arr))
    nplots = nnum + nden + len(args.num_arr) + 1
    nrow, ncol = int(np.ceil(float(nplots/3))), min(nplots,3)
    marker_arr = ['s', '^']
    met_maxlim = 0.5

    if not args.hide:
        if not args.inspect:
            choice = 0
            
            if choice == 0:
                col = 'b'
            if choice == 1:
                col_map = np.log10(np.inner(properties.ppvcube, properties.delta_lambda))
                cbtitle = 'Summed up surface brightness (log ergs/s/pc^2)'
            elif choice == 2:
                col_map = np.log10(np.sum(properties.ppvcube, axis=2))
                cbtitle = 'Summed up surface brightness (log ergs/s/A/pc^2)'
            elif choice == 3:
                col_map = np.log10(mapha)
                cbtitle = 'Ha surface brightness (log ergs/s/pc^2)'
            elif choice == 4:
                dummy,x,y = calcpos(logbook.s, args.galsize, args.res)
                col_map = np.log10(make2Dmap(logbook.s['nII'], x, y, g, args.galsize/g, weights=10**logbook.s['logQ0']))
                cbtitle = 'Luminosity weighted average H2R density of pixel (log cm^-3)'
            elif choice == 5:
                dummy,x,y = calcpos(logbook.s, args.galsize, args.res)
                col_map = make2Dmap(logbook.s['age(MYr)'], x, y, g, args.galsize/g, weights=10**logbook.s['logQ0'])
                cbtitle = 'Luminosity weighted average H2R age of pixel (Myr)'
            elif choice == 6:
                dummy,x,y = calcpos(logbook.s, args.galsize, args.res)
                col_map = np.log10(make2Dmap(10**logbook.s['logQ0'], x, y, g, args.galsize/g))
                cbtitle = 'Bolometric luminosity (log ergs/s)'
            elif choice == 7:
                dummy,x,y = calcpos(logbook.s, args.galsize, args.res)
                fit = mapha # np.add(maps2a,maps2b) # 
                lines = logbook.s['H6562'] #np.add(logbook.s['SII6717'],logbook.s['SII6730']) # 
                true = make2Dmap(lines, x, y, g, args.galsize/g)/(logbook.final_pix_size*1e3)**2
                mydiag('fitted SII map', fit, args)
                mydiag('true SII map', true, args)
                col_map = np.log10(np.abs(fit/true))
                cbtitle = 'Inferred / True Ha SB (log ergs/s/pc^2)'

            if choice:
                col = col_map.flatten()
                cmin = np.min(np.ma.masked_where(col<0, col))
                cmax = np.max(np.ma.masked_where(col<0, col))
            else:
                cmin, cmax = None, None
            
            fig, axes = plt.subplots(nrow, ncol, sharex=True, figsize=(12,8))
            fig.subplots_adjust(hspace=0.1, wspace=0.25, top=0.9, bottom=0.1, left=0.07, right=0.98)
        else:
            col = 'b'
            cmin, cmax = None, None
            fig = plt.gcf()
        
        map_num, map_num_u, map_den, map_den_u = [],[],[],[]
        plot_index, already_plotted = 0, []
        
        for num_grp in args.num_arr:
            ax = axes[plot_index/ncol][plot_index%ncol]
            map_num_grp, map_num_grp_u = [],[]
            if not iterable(num_grp): num_grp = [num_grp]
            if num_grp not in already_plotted:
                plot = True
                already_plotted.append(num_grp)
            else: plot=False
            for (jj,num) in enumerate(num_grp):
        
                if args.SNR_thresh is not None:
                    temp_u = properties.errorcube[:,:,np.where(logbook.llist == num)[0][0]]*(logbook.final_pix_size*1e3)**2
                    temp_u = np.ma.masked_where(temp<=0., temp)
                    map_num_grp_u.append(temp_u)
        
                temp = properties.mapcube[:,:,np.where(logbook.llist == num)[0][0]]*(logbook.final_pix_size*1e3)**2
                temp = np.ma.masked_where(temp<=0., temp)
                if args.SNR_thresh is not None: temp = np.ma.masked_where(temp/temp_u<args.SNR_thresh, temp)
                if plot:
                    if args.inspect:
                        myprint('fitted: '+num+' max= '+str(np.max(temp))+', min= '+str(np.min(temp))+', integrated= '+str(np.sum(temp*(args.galsize*1000./g)**2)), args)
                        if not args.nomap: dummy = plotmap(temp, num+' map after fitting', 'Metallicity', 'log flux(ergs/s)', args, logbook, islog=True)
                    ax.scatter(d.flatten(), temp.flatten(), s=5, lw=0, marker=marker_arr[jj], c=col, label='pixel' if plot_index==1 else None, vmin=cmin, vmax=cmax)
                map_num_grp.append(temp)
            map_num.append(map_num_grp)
            if args.SNR_thresh is not None: map_num_u.append(map_num_grp_u)
            if plot:
                ax.set_ylabel(','.join(num_grp))
                if not args.inspect: ax.set_ylim(-0.1*np.max(temp), 1*np.max(temp))
                plot_index += 1
    
        for den_grp in args.den_arr:
            ax = axes[plot_index/ncol][plot_index%ncol]
            map_den_grp, map_den_grp_u = [],[]
            if not iterable(den_grp): den_grp = [den_grp]
            if den_grp not in already_plotted:
                plot = True
                already_plotted.append(den_grp)
            else: plot=False
            for (jj,den) in enumerate(den_grp):
        
                if args.SNR_thresh is not None:
                    temp_u = properties.errorcube[:,:,np.where(logbook.llist == den)[0][0]]*(logbook.final_pix_size*1e3)**2
                    temp_u = np.ma.masked_where(temp<=0., temp)
                    map_den_grp_u.append(temp_u)
        
                temp = properties.mapcube[:,:,np.where(logbook.llist == den)[0][0]]*(logbook.final_pix_size*1e3)**2
                temp = np.ma.masked_where(temp<=0., temp)
                if args.SNR_thresh is not None: temp = np.ma.masked_where(temp/temp_u<args.SNR_thresh, temp)
                if plot:
                    if args.inspect: 
                        myprint('fitted: '+den+' max= '+str(np.max(temp))+', min= '+str(np.min(temp))+', integrated= '+str(np.sum(temp*(args.galsize*1000./g)**2)), args)
                        if not args.nomap: dummy = plotmap(temp, den+' map after fitting', 'Metallicity', 'log flux(ergs/s)', args, logbook, islog=True)
                    ax.scatter(d.flatten(), temp.flatten(), s=5, lw=0, marker=marker_arr[jj], c=col, label='pixel' if plot_index==1 else None, vmin=cmin, vmax=cmax)
                map_den_grp.append(temp)
            map_den.append(map_den_grp)
            if args.SNR_thresh is not None: map_den_u.append(map_den_grp_u)
            if plot:
                ax.set_ylabel(','.join(den_grp))
                if not args.inspect: ax.set_ylim(-0.1*np.max(temp), 1*np.max(temp))
                plot_index += 1

        map_num_series = [np.ma.sum(map_num[ind], axis=0) for ind in range(len(args.num_arr))]        
        map_den_series = [np.ma.sum(map_den[ind], axis=0) for ind in range(len(args.den_arr))]        

        if args.useKD:
            logOHsol, logOHobj_map = get_KD02_met(map_num_series,map_den_series)
            t += '_KD02'
        else:
            logOHsol, logOHobj_map = get_D16_met(map_num_series,map_den_series)
            t += '_D16'    
        myprint('logOHobj_map before conversion med, min '+str(np.median(logOHobj_map))+' '+str(np.min(logOHobj_map))+'\n', args)
        #---------------------------------------------------
        Z_list =logOHobj_map.flatten()
        myprint('Z_list after conversion med, mean, min '+str(np.median(Z_list))+' '+str(np.mean(Z_list))+' '+str(np.min(Z_list))+'\n', args)

        for ii in range(len(map_num_series)):
            ax = axes[plot_index/ncol][plot_index%ncol]
            ax.scatter(d.flatten(), np.divide(map_num_series[ii],map_den_series[ii]).flatten(), s=5, lw=0, marker=marker_arr[jj], c=col, vmin=cmin, vmax=cmax)
            ax.set_ylabel(','.join(args.num_arr[ii])+'/'+','.join(args.den_arr[ii]))
            ax.set_ylim(args.ratio_lim[ii])
            plot_index += 1
            
        ax = axes[plot_index/ncol][plot_index%ncol]
        ax.scatter(d.flatten(),Z_list, s=5, lw=0, c=col, vmin=cmin, vmax=cmax)
        ax.set_ylabel('Z/Z_sun')
        if not args.inspect and choice: plt.colorbar().set_label(cbtitle)
        
        ax.set_ylim(-1.3,met_maxlim)
        ax.plot(np.arange(args.galsize/2), np.poly1d((args.logOHgrad, args.logOHcen))(np.arange(args.galsize/2)) - logOHsol,c='r', label='True gradient')
        ax.text(0.08, met_maxlim-0.02, 'True slope= %.4F'%(args.logOHgrad), va='top', color='r', fontsize=10)
        ax.text(0.08, met_maxlim-0.12, 'True interecept= %.4F'%(args.logOHcen-logOHsol), va='top', color='r', fontsize=10)
        ax.axhline(0,c='k',linestyle='--',label='Zsol') #line for solar metallicity
        ax.set_xlim(0,args.galsize/2)
        fig.text(0.5, 0.03, 'Galactocentric distance (kpc)', ha='center')
        fig.text(0.5,0.95, t, ha='center')
        if args.saveplot:
            fig.savefig(path+t+'.png')
        if args.getmap: map = plotmap(logOHobj_map, t, 'Metallicity', 'Z/Z_sol', args, logbook, islog=False)
    #---to compute apparent gradient and scatter---
    if args.calcgradient:
        myprint('Fitting gradient..'+'\n', args)
        try:
            d_list = np.ma.masked_array(d.flatten(), Z_list.mask)
            Z_list = np.ma.compressed(Z_list)
            d_list = np.ma.compressed(d_list)            
            if len(Z_list) == 0: raise ValueError
            linefit, linecov = np.polyfit(d_list, Z_list, 1, cov=True)
            myprint('Fit paramters: '+str(linefit)+'\n', args)
            myprint('Fit errors: '+str(linecov)+'\n', args)
            properties.logOHgrad, properties.logOHcen = linefit
            if not args.nowrite:
                gradfile = 'met_grad_log_paint'
                if args.scale_exptime: gradfile += '_exp'
                if args.fixed_SNR is not None: gradfile += '_fixed_SNR'+str(args.fixed_SNR)
                gradfile += '.txt'
                if not os.path.exists(gradfile):
                    head = '#File to store metallicity gradient information for different telescope parameters, as pandas dataframe\n\
#Columns are:\n\
#simulation : name of simulation from which star particles were extracted\n\
#res_arcsec : spatial resoltn of telescope\n\
#vres : spectral resoltn of telescope\n\
#pow, size : smoothing parameters, assuming Moffat profile\n\
#SNR_thresh : SNR threshold applied before fitting the metallicity gradient, if any\n\
#slope, intercept : fitted parameters\n\
#scatter : RMS deviation = sqrt((sum of squared deviation from fit)/ number of data points)\n\
#by Ayan\n\
simulation  res_arcsec   res_phys   vres        power   size    logOHcen        logOHgrad       SNR_thresh      slope       slope_u     \
intercept       intercept_u       scale_exptime       realisation\n'
                    open(gradfile,'w').write(head)
                with open(gradfile,'a') as fout:
                    output = '\n'+args.file+'\t\t'+str(args.res_arcsec)+'\t\t'+str(logbook.final_pix_per_beam * \
                    args.galsize/np.shape(properties.mapcube)[0])+'\t\t'+str(args.vres)+'\t\t'+str(args.parm[0])+'\t\t'+\
                    str(args.parm[1])+'\t\t'+str(args.logOHcen)+'\t\t'+'\t\t'+str(args.logOHgrad)+'\t\t'+'\t\t'+\
                    str(args.SNR_thresh)+'\t\t'+'\t\t'+str('%0.4F'%properties.logOHgrad)+'\t\t'+str('%0.4F'%np.sqrt(linecov[0][0]))+\
                    '\t\t'+str('%0.4F'%properties.logOHcen)+'\t\t'+'\t\t'+str('%0.4F'%np.sqrt(linecov[1][1]))+'\t\t'+\
                    '\t\t'+str(float(args.scale_exptime))+'\t\t'+'\t\t'+str(args.multi_realisation)
                    fout.write(output)
                
            x_arr = np.arange(0,10,0.1)
            if not args.hide:
                axes[-1][-1].plot(x_arr, np.poly1d(linefit)(x_arr), c='b',label='Inferred gradient')
                axes[-1][-1].text(0.08, met_maxlim-0.25, 'Inferred slope= %.4F +/- %.4F'%(properties.logOHgrad, np.sqrt(linecov[0][0])), va='top', color='b', fontsize=10)
                axes[-1][-1].text(0.08, met_maxlim-0.35, 'Inferred interecept= %.4F +/- %.4F'%(properties.logOHcen, np.sqrt(linecov[1][1])), va='top', color='b', fontsize=10)
        except (TypeError, IndexError, ValueError):
            myprint('No data points for vres= '+str(args.vres)+' above given SNR_thresh of '+str(args.SNR_thresh)+'\n', args)
            pass
    return properties       
#-------------Fucntion for fitting multiple lines----------------------------
def fit_all_lines(args, logbook, wave, flam, cont, pix_i, pix_j, z=0, z_err=0.0001) :
    kk, count, flux_array, flux_error_array = 1, 0, [], []
    ndlambda_left, ndlambda_right = [args.nres]*2 #how many delta-lambda wide will the window (for line fitting) be on either side of the central wavelength, default 5
    try:
        count = 1
        first, last = [logbook.wlist[0]]*2
    except IndexError:
        pass
    while kk <= len(logbook.llist):
        center1 = last
        if kk == len(logbook.llist):
            center2 = 1e10 #insanely high number, required to plot last line
        else:
            center2 = logbook.wlist[kk]
        if center2*(1. - ndlambda_left/logbook.resoln) > center1*(1. + ndlambda_right/logbook.resoln):
            leftlim = first*(1.-ndlambda_left/logbook.resoln) 
            rightlim = last*(1.+ndlambda_right/logbook.resoln)
            wave_short = wave[(leftlim < wave) & (wave < rightlim)]
            flam_short = flam[(leftlim < wave) & (wave < rightlim)]
            cont_short = cont[(leftlim < wave) & (wave < rightlim)]
            if args.debug: myprint('Trying to fit '+str(logbook.llist[kk-count:kk])+' line/s at once. Total '+str(count)+'\n', args)
            try: 
                popt, pcov = fitline(wave_short, flam_short, logbook.wlist[kk-count:kk], logbook.resoln, z=z, z_err=z_err)
                popt, pcov = np.array(popt), np.array(pcov)
                popt = np.concatenate(([1.], popt)) #for fitting after continuum normalised, so continuum is fixed=1 and has to be inserted to popt[] by hand after fitting
                pcov = np.hstack((np.zeros((np.shape(pcov)[0]+1,1)), np.vstack((np.zeros(np.shape(pcov)[1]),pcov)))) #for fitting after continuum normalised, so error in continuum is fixed=0 and has to be inserted to pcov[] by hand after fitting
                if args.showplot:
                    plt.axvline(leftlim, linestyle='--',c='g')
                    plt.axvline(rightlim, linestyle='--',c='g')
                ndlambda_left, ndlambda_right = [args.nres]*2
                if args.debug: myprint('Done this fitting!'+'\n', args)
            
            except TypeError, er:
                if args.debug: myprint('Trying to re-do this fit with broadened wavelength window..\n', args)
                ndlambda_left+=1
                ndlambda_right+=1
                continue
            except (RuntimeError, ValueError), e:
                popt = np.concatenate(([1.],np.zeros(count*3))) #if could not fit the line/s fill popt with zeros so flux_array gets zeros
                pcov = np.zeros((count*3 + 1,count*3 + 1)) #if could not fit the line/s fill popt with zeros so flux_array gets zeros
                myprint('Could not fit lines '+str(logbook.llist[kk-count:kk])+' for pixel '+str(pix_i)+', '+str(pix_j)+'\n', args)
                pass
             
            for xx in range(0,count):
                #in popt for every bunch of lines,
                #elements (0,1,2) or (3,4,5) etc. are the height(b), mean(c) and width(d)
                #so, for each line the elements (cont=a,0,1,2) or (cont=a,3,4,5) etc. make the full suite of (a,b,c,d) gaussian parameters
                #so, for each line, flux f (area under gaussian) = sqrt(2pi)*(b-a)*d
                #also the full covariance matrix pcov looks like:
                #|00 01 02 03 04 05 06 .....|
                #|10 11 12 13 14 15 16 .....|
                #|20 21 22 23 24 25 26 .....|
                #|30 31 32 33 34 35 36 .....|
                #|40 41 42 43 44 45 46 .....|
                #|50 51 52 53 54 55 56 .....|
                #|60 61 62 63 64 65 66 .....|
                #|.. .. .. .. .. .. .. .....|
                #|.. .. .. .. .. .. .. .....|
                #
                #where, 00 = var_00, 01 = var_01 and so on.. (var = sigma^2)
                #let var_aa = vaa (00), var_bb = vbb(11), var_ab = vab(01) = var_ba = vba(10) and so on..
                #for a single gaussian, f = const * (b-a)*d
                #i.e. sigma_f^2 = d^2*(saa^2 + sbb^2) + (b-a)^2*sdd^2 (dropping the constant for clarity of explanation)
                #i.e. var_f = d^2*(vaa + vbb) + (b-a)^2*vdd
                #the above holds if we assume covariance matrix to be diagonal (off diagonal terms=0) but thats not the case here
                #so we also need to include off diagnoal covariance terms while propagating flux errors
                #so now, for each line, var_f = d^2*(vaa + vbb) + (b-a)^2*vdd + 2d^2*vab + 2d*(b-a)*(vbd - vad)
                #i.e. in terms of element indices,
                #var_f = 3^2(00 + 11) + (1-0)^2*33 - (2)*3^2*10 + (2)*3*(1-0)*(13-03),
                #var_f = 6^2(00 + 44) + (4-0)^2*66 - (2)*6^2*40 + (2)*6*(4-0)*(46-06),
                #var_f = 9^2(00 + 77) + (1-0)^2*99 - (2)*9^2*70 + (2)*9*(7-0)*(79-09), etc.
                #
                popt_single= np.concatenate(([popt[0]],popt[3*xx+1:3*(xx+1)+1]))
                cont_at_line = cont[np.where(wave >= logbook.wlist[kk+xx-count])[0][0]]
                if args.oneHII is not None: print 'Debugging534: linefit param at (',pix_i,',',pix_j,') for ',logbook.llist[kk+xx-count],'(ergs/s/pc^2/A) =',popt_single #
                flux = np.sqrt(2*np.pi)*(popt_single[1] - popt_single[0])*popt_single[3]*cont_at_line #total flux = integral of guassian fit ; resulting flux in ergs/s/pc^2 units
                if args.oneHII is not None: print 'Debugging536: lineflux at (',pix_i,',',pix_j,') for ',logbook.llist[kk+xx-count],'(ergs/s/pc^2/A) =',flux #                
                flux_array.append(flux)
                flux_error = np.sqrt(2*np.pi*(popt_single[3]**2*(pcov[0][0] + pcov[3*xx+1][3*xx+1])\
                + (popt_single[1]-popt_single[0])**2*pcov[3*(xx+1)][3*(xx+1)]\
                - 2*popt_single[3]**2*pcov[3*xx+1][0]\
                + 2*(popt_single[1] - popt_single[0])*popt_single[3]*(pcov[3*xx+1][3*(xx+1)] - pcov[0][3*(xx+1)])\
                ))*cont_at_line # var_f = 3^2(00 + 11) + (1-0)^2*33 - (2)*3^2*10 + (2)*3*(1-0)*(13-03)
                flux_error_array.append(flux_error)
                if args.showplot:
                    leftlim = popt_single[2]*(1.-args.nres/logbook.resoln) 
                    rightlim = popt_single[2]*(1.+args.nres/logbook.resoln)
                    wave_short_single = wave[(leftlim < wave) & (wave < rightlim)]
                    cont_short_single = cont[(leftlim < wave) & (wave < rightlim)]
                    plt.plot(wave_short_single, su.gaus(wave_short_single,1, *popt_single)*cont_short_single,lw=1, c='r')
            if args.showplot:
                if count >1: plt.plot(wave_short, su.gaus(wave_short, count, *popt)*cont_short,lw=2, c='g')                   
                plt.draw()
                        
            first, last = [center2]*2
            count = 1
        else:
            last = center2
            count += 1
        kk += 1
    #-------------------------------------------------------------------------------------------
    flux_array = np.array(flux_array)
    flux_error_array = np.array(flux_error_array)
    return flux_array, flux_error_array
#-------------------------------------------------------------------------------------------
def fitline(wave, flam, wtofit, resoln, z=0, z_err=0.0001):
    v_maxwidth = 10*c/resoln #10*vres in km/s
    z_allow = 3*z_err #wavelengths are at restframe; assumed error in redshift
    p_init, lbound, ubound = [],[],[]
    for xx in range(0, len(wtofit)):
        fl = np.max(flam) #flam[(np.abs(wave - wtofit[xx])).argmin()] #flam[np.where(wave <= wtofit[xx])[0][0]]
        p_init = np.append(p_init, [fl-flam[0], wtofit[xx], wtofit[xx]*2.*gf2s/resoln])
        lbound = np.append(lbound,[0., wtofit[xx]*(1.-z_allow/(1.+z)), wtofit[xx]*1.*gf2s/resoln])
        ubound = np.append(ubound,[np.inf, wtofit[xx]*(1.+z_allow/(1.+z)), wtofit[xx]*v_maxwidth*gf2s/c])
    popt, pcov = curve_fit(lambda x, *p: fixcont_erf(x, 1., len(wtofit), *p),wave,flam,p0= p_init, max_nfev=10000, bounds = (lbound, ubound))
    return popt, pcov
#-------------------------------------------------------------------------------------------
def emissionmap(args, logbook, properties):
    map = properties.mapcube[:,:,np.where(logbook.llist == args.line)[0][0]]
    map = np.ma.masked_where(map<0., map)
    if args.SNR_thresh is not None:
        map_u = properties.errorcube[:,:,np.where(logbook.llist == args.line)[0][0]]        
        map = np.ma.masked_where(map/map_u <args.SNR_thresh, map)
    t = args.line+'_map:\n'+logbook.fitsname
    map = plotmap(map, t, args.line, 'Log '+args.line+' surface brightness in erg/s/pc^2', args, logbook)
    return map
#-------------------------------------------------------------------------------------------
def SFRmaps(args, logbook, properties):
    global info
    ages = logbook.s['age(MYr)']
    masses = logbook.s['mass(Msun)']
    #----------to get correct conversion rate for lum to SFR----------------------------------------------
    SBmodel = 'starburst08' #this has continuous 1Msun/yr SFR 
    input_quanta = HOME+'/SB99-v8-02/output/'+SBmodel+'/'+SBmodel+'.quanta'
    SB99_age = np.array([float(x.split()[0]) for x in open(input_quanta).readlines()[6:]])
    SB99_logQ = np.array([float(x.split()[1]) for x in open(input_quanta).readlines()[6:]])
    const = 1./np.power(10,SB99_logQ)
    #-------------------------------------------------------------------------------------------
    #const = 7.9e-42*1.37e-12 #factor to convert Q0 to SFR, value from literature
    const = 1.0*const[-1] #factor to convert Q0 to SFR, value from corresponding SB99 file (07)
    #d = np.sqrt(b[:,None]**2+b**2)
    SFRmapHa = properties.mapcube[:,:,np.where(logbook.llist == 'H6562')[0][0]]
    g,x,y = calcpos(logbook.s, args.galsize, args.res)
    SFRmap_real = make2Dmap(masses, x, y, g, args.res)/((res*1e3)**2)
    agemap = 1e6*make2Dmap(ages, x, y, g, args.res, domean=True)
    #SFRmap_real /= agemap #dividing by mean age in the box
    SFRmap_real /= 5e6 #dividing by straight 5M years
    SFRmap_real[np.isnan(SFRmap_real)]=0
    SFRmap_real = rebin_old(SFRmap_real,np.shape(SFRmapHa))
    SFRmap_real = np.ma.masked_where(SFRmap_real<=0., SFRmap_real)
    SFRmapHa *= (const/1.37e-12) #Msun/yr/pc^2
    
    t = title(args.file)+'SFR map for Omega = '+str(args.Om)+', resolution = '+str(logbook.final_pix_size)+' kpc'+info
    if args.getmap:
        #SFRmapQ0 = plotmap(SFRmapQ0, t, 'SFRmapQ0', 'Log SFR(Q0) density in Msun/yr/pc^2', args, logbook)
        SFRmap_real = plotmap(SFRmap_real, t, 'SFRmap_real', 'Log SFR(real) density in Msun/yr/pc^2', args, logbook)
        SFRmapHa = plotmap(SFRmapHa, t, 'SFRmapHa', 'Log SFR(Ha) density in Msun/yr/pc^2', galsize, args, logbook)
        #SFRmap_comp = plotmap(SFRmap_comp, t, 'Log SFR(Q0)/SFR(real) in Msun/yr/pc^2', args, logbook, islog=False)   
    else:
        fig = plt.figure(figsize=(8,6))
        fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.1, left=0.1, right=0.95)
        ax = plt.subplot(111)
        #ax.scatter((np.log10(SFRmap_real)).flatten(),(np.log10(SFRmapQ0)).flatten(), s=4, c='r', lw=0, label='SFR(Q0)')
        ax.scatter((np.log10(SFRmap_real)).flatten(),(np.log10(SFRmapHa)).flatten(), s=4, c='b', lw=0, label='SFR(Ha)')
        #ax.scatter((np.log10(SFRmap_real)).flatten(),(np.log10(SFRmapHa)).flatten(), s=4, c=col_ar[i], lw=0, label=str(res_phys))
        #ax.scatter((np.log10(SFRmap_real)).flatten(),(np.log10(SFRmapHa)).flatten(), s=4, c=d.flatten(), lw=0, label='SFR(Ha)')
        #-----to plot x=y line----------#
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label = 'x=y line')
        #-------------------------------#
        t= 'SFR comparison for '+args.file+', res '+str(logbook.final_pix_size)+' kpc'+info
        plt.ylabel('Log (Predicted SFR density) in Msun/yr/pc^2')
        plt.xlabel('Log (Actual SFR density) in Msun/yr/pc^2')
        plt.title(t)
        #plt.colorbar().set_label('Galactocentric distance (in pix)')
        plt.legend(bbox_to_anchor=(0.35, 0.88), bbox_transform=plt.gcf().transFigure)  
        if saveplot:
            fig.savefig(args.path+title(args.file)[:-2]+'_'+t+'.png')
        '''
        ax.scatter(reso,mean,s=4)
        plt.xlabel('res(kpc)')
        plt.ylabel('log(mean_SFR_density) in Msun/yr/kpc^2')
        plt.title('Resolution dependence of SFR density for '+fn)
        '''
    return SFRmap_real, SFRmapHa
#-------------------------------------------------------------------------------------------------
def readSB(wmin, wmax):
    inpSB=open(HOME+'/SB99-v8-02/output/starburst08/starburst08.spectrum','r') #sb08 has cont SF
    speclines = inpSB.readlines()[5:]
    age = [0., 1., 2., 3., 4., 5.] #in Myr
    funcar = []
    for a in age:
        cw, cf = [],[]
        for line in speclines:
            if float(line.split()[0])/1e6 == a:
                if wmin-150. <= float(line.split()[1]) <= wmax+150.:
                    cw.append(float(line.split()[1])) #column 1 is wavelength in A
                    cf.append(10**float(line.split()[3])) #column 3 is stellar continuum in ergs/s/A
        funcar.append(interp1d(cw, cf, kind='cubic'))
        #plt.plot(cw, np.divide(cf,5e34),lw=0.5, linestyle='--') #
    return funcar
#-------------------------------------------------------------------------------------------
def spec_at_point(args, logbook, properties):
    global info
    fig = plt.figure(figsize=(14,6))
    fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.05, right=0.95)
    for i in logbook.wlist:
        plt.axvline(i,ymin=0.9,c='black')    
    cbarlab = 'Log surface brightness in erg/s/pc^2' #label of color bar
    plt.plot(properties.dispsol, np.log10(properties.ppvcube[args.X][args.Y][:]),lw=1, c='b')
    t = 'Spectrum at pp '+str(args.X)+','+str(args.Y)+' for '+logbook.fitsname
    plt.title(t)
    plt.ylabel(cbarlab)
    plt.xlabel('Wavelength (A)')
    plt.ylim(30,37)
    plt.xlim(logbook.wmin,logbook.wmax)
    if not args.hide:
        plt.show(block=False)
    if args.saveplot:
        fig.savefig(path+t+'.png')
#-------------------------------------------------------------------------------------------
def plotintegmap(args, logbook, properties):
    ppv = properties.ppvcube[:,:,(properties.dispsol >= logbook.wmin) & (properties.dispsol <= logbook.wmax)]
    cbarlab = 'Log surface brightness in erg/s/pc^2' #label of color bar
    line = 'lambda-integrated wmin='+str(logbook.wmin)+', wmax='+str(logbook.wmax)+'\n'
    map = np.sum(ppv,axis=2)
    t = title(args.file)+line+' map for Omega = '+str(args.Om)+', res = '+str(logbook.final_pix_size)+' kpc'
    dummy = plotmap(map, t, line, cbarlab, args, logbook)
    return dummy
#-------------------------------------------------------------------------------------------
def spec_total(w, ppv, thistitle, args, logbook):
    cbarlab = 'Surface brightness in erg/s/pc^2' #label of color bar
    fig = plt.figure(figsize=(14,6))
    fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.1, right=0.95)
    ax = plt.subplot(111)
    for i in logbook.wlist:
        plt.axvline(i,ymin=0.9,c='black')    

    #-------------------------------------------------------------------------------------------
    y = np.sum(ppv,axis=(0,1))
    plt.plot(w, y,lw=1)
    t = thistitle+', for '+title(args.file)+' Nebular+ stellar for Om = '+str(args.Om)+', res = '+str(logbook.final_pix_size)+' kpc' + info
    #-------------------------------------------------------------------------------------------
    plt.title(t)
    plt.ylabel(cbarlab)
    plt.xlabel('Wavelength (A)')
    plt.ylim(np.min(y)*0.9, np.max(y)*1.1)
    plt.xlim(logbook.wmin,logbook.wmax)
    plt.show(block=False)
    if args.saveplot:
        fig.savefig(args.path+t+'.png')
    if args.debug: plt.show(block=False)
#-------------------------------------------------------------------------------------------
def get_disp_array(args, logbook, properties):
    sig = 5*args.vdel/c 
    w = np.linspace(logbook.wmin, logbook.wmax, args.nbin)
    for ii in logbook.wlist:
        w1 = ii*(1-sig)
        w2 = ii*(1+sig)
        highres = np.linspace(w1, w2, args.nhr)
        w = np.hstack((w[:np.where(w<w1)[0][-1]+1], highres, w[np.where(w>w2)[0][0]:]))
    properties.w = w
    #-------------------------------------------------------------------------------------------
    if args.spec_smear:        
        properties.new_w = [properties.w[0]]
        while properties.new_w[-1] <= properties.w[-1]:
            properties.new_w.append(properties.new_w[-1]*(1+args.vres/c))
        properties.bin_index = np.digitize(properties.w, properties.new_w)
    
    properties.dispsol = np.array(properties.new_w[1:]) if args.spec_smear else np.array(properties.w)
    properties.nwbin = len(properties.dispsol)
    properties.delta_lambda = np.array([properties.dispsol[1]-properties.dispsol[0]]+[(properties.dispsol[i+1]-properties.dispsol[i-1])/2 for i in range(1,len(properties.dispsol)-1)]+[properties.dispsol[-1]-properties.dispsol[-2]]) #wavelength spacing for each wavecell; in Angstrom     

    if args.debug: myprint('Deb663: for vres= '+str(args.vres)+', length of dispsol= '+str(len(properties.dispsol))+'\n', args)
    return properties
#-------------------------------------------------------------------------------------------
def spec(args, logbook, properties):
    global info
    properties = get_disp_array(args, logbook, properties)
    if os.path.exists(logbook.H2R_filename):
        ppv = fits.open(logbook.H2R_filename)[0].data
        if not args.silent: myprint('Reading existing H2R file cube from '+logbook.H2R_filename+'\n', args)
    else:
        #-------------------------------------------------------------------------------------------
        g,x,y = calcpos(logbook.s, args.galsize, args.res)
        ppv = np.zeros((g,g,properties.nwbin))
        funcar = readSB(logbook.wmin, logbook.wmax)
        #-------------------------------------------------------------------------------------------            
        if args.oneHII is not None:
            print 'Debugging750: Mappings fluxes (ergs/s/pc^2) for H2R #',args.oneHII,'=', np.array([logbook.s[line][args.oneHII] for line in ['H6562','NII6584','SII6717','SII6730']])/(args.res*1e3)**2
            startHII, endHII = args.oneHII, args.oneHII + 1
        else:
            startHII, endHII = 0, len(logbook.s)
        for j in range(startHII, endHII):
            myprint('Particle '+str(j+1)+' of '+str(len(logbook.s))+'\n', args)
            vz = float(logbook.s['vz'][j])
            a = int(round(logbook.s['age(MYr)'][j]))
            f = np.multiply(funcar[a](properties.w),(300./1e6)) #to scale the continuum by 300Msun, as the ones produced by SB99 was for 1M Msun
                                                                #the continuum is in ergs/s/A

            if args.debug and j==0:
                fig = plt.figure(figsize=(14,6))
                plt.plot(properties.w, f, label='cont')
                plt.xlim(logbook.wmin, logbook.wmax)
                plt.xlabel('Wavelength (A)')
                plt.ylabel('flam (ergs/s/A)')
                
            flist=[]
            for l in logbook.llist:
                try: flist.append(logbook.s[l][j]) #ergs/s
                except: continue
            flist = np.multiply(np.array(flist), const) #to multiply with nebular flux to make it comparable with SB continuum
        
            for i, fli in enumerate(flist):
                f = gauss(properties.w, f, logbook.wlist[i], fli, args.vdisp, vz) #adding every line flux on top of continuum; gaussians are in ergs/s/A
            
            if args.debug and j==0: plt.plot(properties.w, f, label='cont + lines')
            
            if args.spec_smear: f = np.array([f[properties.bin_index == ii].sum() for ii in range(1, properties.nwbin + 1)]) #spectral smearing i.e. rebinning of spectrum
            
            if args.debug and j==0:
                if args.spec_smear: plt.plot(properties.dispsol, f, label='cont+lines+smeared:vres= '+str(args.vres))
                plt.legend()
                plt.show(block=False)
            
            f *=  properties.flux_ratio #converting from emitted flux to flux actually entering each pixel
            ppv[int(x[j]/args.res)][int(y[j]/args.res)][:] += f  #f is ergs/s/A, ppv becomes ergs/s/A/pixel
        #-------------------------------------------------------------------------------------------
        if not args.silent: myprint('Done reading in all HII regions in '+str((time.time() - start_time)/60)+' minutes.\n', args)
        write_fits(logbook.H2R_filename, ppv, args)
    #------------------------------------------------------------------------#
    if args.debug:
        spec_total(properties.dispsol, ppv, 'Spectrum for only H2R after spec smear', args, logbook)
        myprint('Deb705: Trying to calculate some statistics on the cube of shape ('+str(np.shape(ppv)[0])+','+str(np.shape(ppv)[1])+\
        ','+str(np.shape(ppv)[2])+'), please wait for ~5 minutes. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)
        time_temp = time.time()
        mydiag('Deb706: in ergs/s/A/pixel: for H2R cube',ppv, args)
        myprint('This done in %s minutes\n' % ((time.time() - time_temp)/60), args)
    #----to find a pixel which has non zero dat and plot its spectrum--------#
    XX, YY = np.shape(ppv)[0], np.shape(ppv)[1]
    x, y= 0, 0
    dx = 0
    dy = -1
    for i in range(max(XX, YY)**2):
        if (-XX/2 < x <= XX/2) and (-YY/2 < y <= YY/2):
            if np.array(ppv[x+XX/2,y+YY/2,:]).any(): break
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy
    xx, yy = x+XX/2, y+YY/2
    
    if not args.hide:
        fig = plt.figure(figsize=(15,5))
        for ii in logbook.wlist:
            plt.axvline(ii,ymin=0.9,c='black')    
        plt.ylabel('flam in erg/s/A')
        plt.xlabel('Wavelength (A)')
        plt.ylim(-0.2*np.max(ppv[xx,yy,:]), 1.2*np.max(ppv[xx,yy,:]))
        plt.xlim(logbook.wmin, logbook.wmax)
        plt.title('Spectrum along 1 LOS')
        plt.scatter(properties.dispsol, ppv[xx,yy,:], c='k', s=5, label='H2R spec pix ('+str(xx)+','+str(yy)+')')
    sizex, sizey = np.shape(ppv)[0], np.shape(ppv)[1]
    
    #-------------------------Now ideal PPV is ready: do whatever with it------------------------------------------------------------------
    if args.smooth:
        if os.path.exists(logbook.convolved_filename): #read it in if the convolved cube already exists
            myprint('Reading existing convolved cube from '+logbook.convolved_filename+'\n', args)
        else: #make the convolved cube and save it if it doesn't exist already
            if args.debug: print 'Deb718: shape before rebinning = ', np.shape(ppv)
            myprint('Trying to rebin cube before convolving...\n', args)
            ppv_rebinned = np.zeros((int(args.galsize/logbook.final_pix_size), int(args.galsize/logbook.final_pix_size), np.shape(ppv)[2]))
            for ind in range(0,np.shape(ppv)[2]):
                ppv_rebinned[:,:,ind] = rebin(ppv[:,:,ind], args.res, logbook.new_res) #re-bin 2d array before convolving to make things faster(previously each pixel was of size res)    
            if args.debug:
                print 'Deb723: shape after rebinning = ', np.shape(ppv_rebinned)
                mydiag('Deb724: in ergs/s/A/pixel: for just rebinned PPV cube (before convolution)', ppv_rebinned, args)
            myprint('Rebinning complete. Trying to parallely convolve with '+str(args.ncores)+' core...\n', args)
            myprint('Using '+args.ker+' kernel.\nUsing parameter set: sigma= '+str(logbook.sig)+', size= '+str(logbook.size,)+'\n', args)
            binned_cubename = args.path + 'temp_binned_cube.fits'
            write_fits(binned_cubename, ppv_rebinned, args, fill_val=np.nan)

            funcname = HOME+'/Work/astro/ayan_codes/enzo_model_code/parallel_convolve_old.py'
            if args.silent: silent = ' --silent'
            else: silent = ''
            if args.toscreen: toscreen = ' --toscreen'
            else: toscreen = ''
            if args.debug: debug = ' --debug'
            else: debug = ''
            if args.addnoise: noise = ' --addnoise'
            else: noise = ''
            if args.saveplot: saveplot = ' --saveplot'
            else: saveplot = ''
            if args.hide: hide = ' --hide'
            else: hide = ''
            if args.fixed_SNR is not None: fixed_SNR = ' --fixed_SNR '+str(args.fixed_SNR)
            else: fixed_SNR = ''
            if args.scale_exp_SNR is not None: scale_exp_SNR = ' --scale_exp_SNR '+str(args.scale_exp_SNR)
            else: scale_exp_SNR = ''
            if args.cmin is not None: cmin = ' --cmin '+str(args.cmin)
            else: cmin = ''
            if args.cmax is not None: cmax = ' --cmax '+str(args.cmax)
            else: cmax = ''
            
            command = 'mpirun -np '+str(args.ncores)+' python '+funcname+ ' --fitsname '+logbook.fitsname+\
            ' --sig '+str(logbook.sig) + ' --pow '+str(args.pow)+' --size '+str(logbook.size)+' --ker '+args.ker+' --convolved_filename '+\
            logbook.convolved_filename+' --outfile '+args.outfile+' --binned_cubename '+binned_cubename +\
            ' --exptime '+str(logbook.exptime)+' --final_pix_size '+ str(logbook.final_pix_size) + ' --galsize '+str(args.galsize)+\
            fixed_SNR + scale_exp_SNR + noise + silent + toscreen + debug + cmin + cmax + saveplot + hide
            subprocess.call([command],shell=True)
            subprocess.call(['rm -r '+binned_cubename],shell=True)
            
        ppv = fits.open(logbook.convolved_filename)[0].data #reading in convolved cube from file
        if args.debug:
            myprint('Trying to calculate some statistics on the cube, please wait...', args)
            mydiag('Deb737: in ergs/s/A/pixel: for convolved cube', ppv, args)
    if not args.maketheory:
        last_cubename = logbook.convolved_filename if args.smooth else logbook.H2R_filename
        funcname = HOME+'/Work/astro/ayan_codes/enzo_model_code/parallel_makeobs_old.py'
        if args.silent: silent = ' --silent'
        else: silent = ''
        if args.toscreen: toscreen = ' --toscreen'
        else: toscreen = ''
        if args.debug: debug = ' --debug'
        else: debug = ''
        if args.addnoise: noise = ' --addnoise'
        else: noise = ''
        if args.spec_smear: smear = ' --spec_smear '
        else: smear = ''
        if args.saveplot: saveplot = ' --saveplot'
        else: saveplot = ''
        if args.hide: hide = ' --hide'
        else: hide = ''
        if args.fixed_SNR is not None: fixed_SNR = ' --fixed_SNR '+str(args.fixed_SNR)
        else: fixed_SNR = ''
        if args.cmin is not None: cmin = ' --cmin '+str(args.cmin)
        else: cmin = ''
        if args.cmax is not None: cmax = ' --cmax '+str(args.cmax)
        else: cmax = ''
        parallel = 'mpirun -np '+str(args.ncores)+' python '+funcname+' --parallel'
        series = 'python '+funcname
        series_or_parallel = parallel #USE parallel OR series
        
        command = series_or_parallel + ' --fitsname '+logbook.fitsname+' --outfile '+args.outfile+' --last_cubename '+\
        last_cubename+' --nbin '+str(args.nbin)+' --vdel '+str(args.vdel)+' --vdisp '+str(args.vdisp)+' --vres '+str(args.vres)+\
        ' --nhr '+str(args.nhr)+' --wmin '+str(logbook.wmin)+' --wmax '+str(logbook.wmax)+' --epp '+str(args.el_per_phot)+\
        ' --gain '+str(args.gain)+' --exptime '+str(logbook.exptime)+' --final_pix_size '+\
        str(logbook.final_pix_size) + ' --skynoise_cubename '+logbook.skynoise_cubename+' --galsize '+str(args.galsize)+\
        ' --rad ' + str(args.rad) + fixed_SNR + smear + noise + silent + toscreen + debug + cmin + cmax + saveplot + hide
        subprocess.call([command],shell=True)
    else:    
        write_fits(logbook.fitsname, ppv, args, fill_val=np.nan) #writing the last cube itself as the ppv cube, if asked to make theoretical cube
             
    ppv = fits.open(logbook.fitsname)[0].data #reading in ppv cube from file

    if not args.hide:
        xx, yy = xx * np.shape(ppv)[0]/sizex, yy * np.shape(ppv)[1]/sizey
        plt.plot(properties.dispsol, ppv[xx,yy,:]*(logbook.final_pix_size*1e3)**2, c='r', label='obs spec pix ('+str(xx)+','+str(yy)+')')
        plt.legend()
        plt.show(block=False)
    
    if args.debug:
        myprint('Trying to calculate some statistics on the cube, please wait for ~10 minutes. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)
        time_temp = time.time()
        mydiag('Deb739: in ergs/s/pc^2/A: for final PPV cube', ppv, args)
        myprint('This done in %s minutes\n' % ((time.time() - time_temp)/60), args)
    
    if not args.silent: myprint('Final pixel size on target frame = '+str(logbook.final_pix_size)+' kpc'+' and shape = ('+str(np.shape(ppv)[0])+','+str(np.shape(ppv)[1])+','+str(np.shape(ppv)[2])+') \n', args)
    #-------------------------Now realistic (smoothed, noisy) PPV is ready------------------------------------------------------------------
    #if not args.hide: spec_total(properties.dispsol, ppv, 'Spectrum for total', args, logbook)
    myprint('Returning PPV as variable "ppvcube"'+'\n', args)
    return np.array(ppv)                
#-------------------------------------------------------------------------------------------
def makeobservable(cube, args, logbook, properties, core_start, core_end, prefix):
    nslice = np.shape(cube)[2]
    new_cube = np.zeros(np.shape(cube))
    for k in range(core_start, core_end):        
        if not args.silent: myprint(prefix+'Making observable slice '+str(k+1)+' of '+str(core_end)+' at: {:%Y-%m-%d %H:%M:%S}\n'.format(datetime.datetime.now()), args)
        map = cube[:,:,k]
        factor = logbook.exptime * args.el_per_phot / (args.gain * planck * (c*1e3) / (properties.dispsol[k]*1e-10)) #to bring ergs/s/A/pixel to units of counts/pixel (ADUs)
        if args.debug:
            mydiag(prefix+'Deb 754: in ergs/s/A/pixel: before factor', map, args)
            myprint(prefix+'Deb1434: Factor = exptime= %.4E * el_per_phot= %.1F / (gain= %.2F * planck= %.4E * c= %.4E / lambda=%.2E)'%(logbook.exptime,\
            args.el_per_phot, args.gain, planck, c*1e3, properties.dispsol[k]*1e-10), args)
            myprint(prefix+'Deb1436: Factor= %.4E\n'%factor, args)
        map *= factor #to get in counts from ergs/s
        if args.debug:
            mydiag(prefix+'Deb 756: in counts/A/pixel: after multiplying by factor', map, args)
        if args.addnoise:
            skynoiseslice = properties.skynoise[k] if properties.skynoise is not None else None
            map = makenoisy(map, args, logbook, properties, skynoiseslice=skynoiseslice, factor=args.gain, slice=k, prefix=prefix) #factor=gain as it is already in counts (ADU), need to convert to electrons for Poisson statistics
        if args.debug: myprint(prefix+'Deb898: # of non zero elements in map before clipping= '+str(len(map.nonzero()[0])), args)
        #map = np.ma.masked_where(np.log10(map)<0., map) #clip all that have less than 1 count
        map = np.ma.masked_where(np.log10(map)>5., map) #clip all that have more than 100,000 count i.e. saturating
        if args.debug: myprint(prefix+'Deb898: # of non zero elements in map after clipping= '+str(len(map.nonzero()[0])), args)
        map /= factor #convert back to ergs/s from counts
        if args.debug: mydiag(prefix+'Deb 762: in ergs/s/A/pixel: after dividing by factor', map, args)
        map /= (logbook.final_pix_size*1e3)**2 #convert to ergs/s/pc^2/A from ergs/s/pix/A
        if args.debug: mydiag(prefix+'Deb 764: in ergs/s/pc^2/A: after dividing by pixel area', map, args)
        new_cube[:,:,k] = map
    return new_cube
#-------------------------------------------------------------------------------------------
def makenoisy(data, args, logbook, properties, skynoiseslice=None, factor=1., slice=None, prefix=None):
    dummy = copy.copy(data)
    slice_slab = 500
    if args.debug:
        if slice is not None and slice%slice_slab == 0: dummy = plotmap(dummy, 'slice '+str(slice)+': before adding any noise', 'junk', 'counts', args, logbook, islog=False)
        mydiag(prefix+'Deb 767: in counts/A/pixel: before adding any noise', dummy, args)
    size = args.galsize/np.shape(data)[0]
    data *= factor #to transform into electrons from physical units
    if args.debug: mydiag(prefix+'Deb 771: in el/A/pixel: after mutiplying gain factor', data, args)
    if args.fixed_SNR is not None: #adding only fixed amount of SNR to ALL spaxels
        noisydata = data + np.random.normal(loc=0., scale=np.abs(data/args.fixed_SNR), size=np.shape(data)) #drawing from normal distribution about a mean value of 0 and width =counts/SNR
        if args.debug:
            if slice is not None and slice%slice_slab == 0: dummy = plotmap(noisydata, 'slice '+str(slice)+': after fixed_SNR '+str(args.fixed_SNR)+' noise', 'junk', 'counts', args, logbook, islog=False)
            mydiag(prefix+'Deb 775: in el/A/pixel: after fixed_SNR '+str(args.fixed_SNR)+' noise', noisydata, args)
    else:
        noisydata = np.random.poisson(lam=data, size=None) #adding poisson noise to counts (electrons)
        noisydata = noisydata.astype(float)
        poissonnoise = noisydata - dummy
        if args.debug:
            if slice is not None and slice%slice_slab == 0: dummy = plotmap(noisydata, 'slice '+str(slice)+': after poisson', 'junk', 'counts', args, logbook, islog=False)
            mydiag(prefix+'Deb 783: in el/A/pixel: after adding poisson noise', noisydata, args)
        readnoise = np.random.normal(loc=0., scale=3.5, size=np.shape(noisydata)) #to draw gaussian random variables from distribution N(0,3.5) where 3.5 is width in electrons per pixel
                                    #multiply sqrt(14) to account for the fact that for SAMI each spectral fibre is 2 pix wide and there are 7 CCD frames for each obsv
        if args.debug: mydiag(prefix+'Deb 781: in el/A/pixel: only RDNoise', readnoise, args)
        noisydata += readnoise #adding readnoise
        if args.debug:
            if slice is not None and slice%slice_slab == 0: dummy = plotmap(noisydata, 'slice '+str(slice)+': after readnoise', 'junk', 'counts', args, logbook, islog=False)
            mydiag(prefix+'Deb 783: in el/A/pixel: after adding readnoise', noisydata, args)
        if skynoiseslice is not None and skynoiseslice != 0: 
            skynoise = np.random.normal(loc=0., scale=np.abs(skynoiseslice), size=np.shape(noisydata)) #drawing from normal distribution about a sky noise value at that particular wavelength
            skynoise *= logbook.exptime #converting skynoise from el/s to el
            noisydata += skynoise #adding sky noise
        else: skynoise = np.zeros(np.shape(noisydata))
        if args.debug:
            if slice is not None and slice%slice_slab == 0: dummy = plotmap(noisydata, 'slice '+str(slice)+': after skynoise', 'junk', 'counts', args, logbook, islog=False)
            mydiag(prefix+'Deb 795: in el/A/pixel: after adding skynoise', noisydata, args)
        totalnoise = poissonnoise + readnoise + skynoise
        snr_map = dummy/totalnoise
        snr_map = np.ma.masked_where(dummy<=0, snr_map)
        if args.debug:
            if slice is not None and slice%slice_slab == 0: dummy = plotmap(snr_map, 'slice '+str(slice)+': SNR map', 'junk', 'ratio', args, logbook, islog=False)
            if args.scale_exp_SNR:
                goodfrac = len(snr_map > float(args.scale_exp_SNR))/np.ma.count(snr_map)
                myprint(prefix+' fraction of pixels above SNR '+str(args.scale_exp_SNR)+' = '+str(goodfrac), args)
            mydiag(prefix+'Deb 797: in ratio-units: SNR map', snr_map, args)
        
    noisydata /= factor #converting back to physical units from electrons
    if args.debug: mydiag(prefix+'Deb 803: in counts/A/pixel: after dividing gain factor', noisydata, args)
    if args.debug:
        noise = noisydata - dummy
        myprint(prefix+'Net effect of all noise:\n'+\
        'makenoisy: array median std min max'+'\n'+\
        'makenoisy: data '+str(np.median(dummy))+' '+str(np.std(dummy))+' '+str(np.min(masked_data(dummy)))+' '+str(np.max(dummy))+'\n'+\
        'makenoisy: noisydata '+str(np.median(noisydata))+' '+str(np.std(noisydata))+' '+str(np.min(masked_data(noisydata)))+' '+str(np.max(noisydata))+'\n'+\
        'makenoisy: noise '+str(np.median(noise))+' '+str(np.std(noise))+' '+str(np.min(masked_data(noise)))+' '+str(np.max(noise))+'\n'\
        , args)
    return noisydata
#-------------------------------------------------------------------------------------------
def inspectmap(args, logbook, properties):
    g,x,y = calcpos(logbook.s, args.galsize, args.res)
    t = args.file+':Met_Om'+str(args.Om)
    if args.smooth: t += '_arc'+str(args.res_arcsec)+'"'
    if args.spec_smear: t+= '_vres='+str(args.vres)+'kmps_'
    t += info+args.gradtext
    if args.SNR_thresh is not None: t += '_snr'+str(args.SNR_thresh)    
    if args.useKD: t+= '_KD02'
    else: t += '_D16'
    
    nnum, nden = len(np.unique(args.num_arr)), len(np.unique(args.den_arr))
    nplots = nnum + nden + len(args.num_arr) + 1
    nrow, ncol = int(np.ceil(float(nplots/3))), min(nplots,3)
    marker_arr = ['s', '^']
    met_maxlim = 0.5

    fig, axes = plt.subplots(nrow, ncol, sharex=True, figsize=(14,8))
    fig.subplots_adjust(hspace=0.1, wspace=0.25, top=0.9, bottom=0.1, left=0.07, right=0.98)
    
    #----------plotting individual H2R---------------------------
    d = np.sqrt((x-args.galsize/2)**2 + (y-args.galsize/2)**2)
    plot_index, col, already_plotted = 0, 'r', []
    
    indiv_map_num, indiv_map_den = [],[]
    for num_grp in args.num_arr:
        ax = axes[plot_index/ncol][plot_index%ncol]
        indiv_map_num_grp = []
        if not iterable(num_grp): num_grp = [num_grp]
        if num_grp not in already_plotted:
            plot = True
            already_plotted.append(num_grp)
        else: plot=False
        for (jj,num) in enumerate(num_grp):        
            temp = logbook.s[num]*properties.flux_ratio
            indiv_map_num_grp.append(temp)
            if plot:
                ax.scatter(d.flatten(), temp.flatten(), s=5, lw=0, marker=marker_arr[jj], c=col, label='pixel' if plot_index==1 else None)
                myprint('all HIIR: '+num+' median= '+str(np.median(temp))+', integrated= '+str(np.sum(temp)), args)
        indiv_map_num.append(indiv_map_num_grp)
        ax.set_ylim(-0.1*np.max(temp), 1*np.max(temp))
        if plot: plot_index += 1

    for den_grp in args.den_arr:
        indiv_map_den_grp = []
        ax = axes[plot_index/ncol][plot_index%ncol]
        if not iterable(den_grp): den_grp = [den_grp]
        if den_grp not in already_plotted:
            plot = True
            already_plotted.append(den_grp)
        else: plot=False
        for (jj,den) in enumerate(den_grp):
            temp = logbook.s[den]*properties.flux_ratio
            indiv_map_den_grp.append(temp)
            if plot:
                ax.scatter(d.flatten(), temp.flatten(), s=5, lw=0, marker=marker_arr[jj], c=col)
                myprint('all HIIR: '+den+' median= '+str(np.median(temp))+', integrated= '+str(np.sum(temp)), args)
        indiv_map_den.append(indiv_map_den_grp)
        ax.set_ylim(-0.1*np.max(temp), 1*np.max(temp))
        if plot: plot_index += 1

    indiv_map_num_series = [np.ma.sum(indiv_map_num[ind], axis=0) for ind in range(len(args.num_arr))]        
    indiv_map_den_series = [np.ma.sum(indiv_map_den[ind], axis=0) for ind in range(len(args.den_arr))]        

    if args.useKD: logOHsol, logOHobj_indiv_map = get_KD02_met(indiv_map_num_series,indiv_map_den_series)
    else: logOHsol, logOHobj_indiv_map = get_D16_met(indiv_map_num_series,indiv_map_den_series)
    myprint('all HIIR: median Z/Zsol= '+str(np.median(10**logOHobj_indiv_map)), args)

    for ii in range(len(indiv_map_num_series)):
        ax = axes[plot_index/ncol][plot_index%ncol]
        ax.scatter(d.flatten(), np.divide(indiv_map_num_series[ii],indiv_map_den_series[ii]).flatten(), s=5, lw=0, marker=marker_arr[jj], c=col)
        plot_index += 1
        
    ax = axes[plot_index/ncol][plot_index%ncol]
    ax.scatter(d.flatten(),logOHobj_indiv_map.flatten(), s=5, lw=0, c=col)
    
    #-------------plotting binned maps------------------------------------
    b = np.linspace(-g/2 + 1,g/2,g)*(args.galsize)/g #in kpc
    d = np.sqrt(b[:,None]**2+b**2)
    plot_index, col, already_plotted = 0, 'g', []

    binned_map_num, binned_map_den = [],[]
    for num_grp in args.num_arr:
        ax = axes[plot_index/ncol][plot_index%ncol]
        binned_map_num_grp = []
        if not iterable(num_grp): num_grp = [num_grp]
        if num_grp not in already_plotted:
            plot = True
            already_plotted.append(num_grp)
        else: plot=False
        for (jj,num) in enumerate(num_grp):        
            temp = make2Dmap(logbook.s[num], x, y, g, args.galsize/g)*properties.flux_ratio
            binned_map_num_grp.append(temp)
            if plot:
                if not args.nomap: plotmap(temp, num+': H2R binned', 'binned', 'log flux(ergs/s)', args, logbook, islog=True)
                ax.scatter(d.flatten(), temp.flatten(), s=5, lw=0, marker=marker_arr[jj], c=col, label='pixel' if plot_index==1 else None)
                myprint('all HIIR: '+num+' median= '+str(np.median(temp))+', integrated= '+str(np.sum(temp)), args)
        binned_map_num.append(binned_map_num_grp)
        if plot:
            ax.set_ylim(-0.1*np.max(temp), 1*np.max(temp))
            ax.set_ylabel(','.join(num_grp))
            plot_index += 1

    for den_grp in args.den_arr:
        binned_map_den_grp = []
        ax = axes[plot_index/ncol][plot_index%ncol]
        if not iterable(den_grp): den_grp = [den_grp]
        if den_grp not in already_plotted:
            plot = True
            already_plotted.append(den_grp)
        else: plot=False
        for (jj,den) in enumerate(den_grp):
            temp = make2Dmap(logbook.s[den], x, y, g, args.galsize/g)*properties.flux_ratio
            binned_map_den_grp.append(temp)
            if plot:
                if not args.nomap: plotmap(temp, den+': H2R binned', 'binned', 'log flux(ergs/s)', args, logbook, islog=True)
                ax.scatter(d.flatten(), temp.flatten(), s=5, lw=0, marker=marker_arr[jj], c=col)
                myprint('binned HIIR: '+den+' median= '+str(np.median(temp))+', integrated= '+str(np.sum(temp)), args)
        binned_map_den.append(binned_map_den_grp)
        if plot:
            ax.set_ylim(-0.1*np.max(temp), 1*np.max(temp))
            ax.set_ylabel(','.join(den_grp))
            plot_index += 1

    binned_map_num_series = [np.ma.sum(binned_map_num[ind], axis=0) for ind in range(len(args.num_arr))]        
    binned_map_den_series = [np.ma.sum(binned_map_den[ind], axis=0) for ind in range(len(args.den_arr))]        

    if args.useKD: logOHsol, logOHobj_binned_map = get_KD02_met(binned_map_num_series,binned_map_den_series)
    else: logOHsol, logOHobj_binned_map = get_D16_met(binned_map_num_series,binned_map_den_series)
    myprint('binned HIIR: median Z/Zsol= '+str(np.median(10**logOHobj_binned_map)), args)

    for ii in range(len(binned_map_num_series)):
        ax = axes[plot_index/ncol][plot_index%ncol]
        ax.scatter(d.flatten(), np.divide(binned_map_num_series[ii],binned_map_den_series[ii]).flatten(), s=5, lw=0, marker=marker_arr[jj], c=col)
        ax.set_ylim(args.ratio_lim[ii])
        plot_index += 1
        
    ax = axes[plot_index/ncol][plot_index%ncol]
    ax.scatter(d.flatten(),logOHobj_binned_map.flatten(), s=5, lw=0, c=col)
    
    ax.set_ylabel('Z/Z_sun')
    ax.set_ylim(-1.3,met_maxlim)
    ax.set_xlim(0,args.galsize/2)
    fig.text(0.5, 0.03, 'Galactocentric distance (kpc)', ha='center')
    fig.text(0.5,0.95, t, ha='center')                  
    
    return axes
#-------------------------------------------------------------------------------------------
def fixfit(args, logbook, properties):

    if not np.array(properties.ppvcube[args.X, args.Y, :]).any():
        myprint('Chosen spaxel is empty. Select another.', args)
        myprint('Non empty spaxels are:', args)
        for i in range(np.shape(properties.ppvcube)[0]):
            for j in range(np.shape(properties.ppvcube)[1]):
                if np.array(properties.ppvcube[i, j, :]).any():
                    print i, j
       
        myexit(args, text='Try again.')

    args.showplot = True
    args.toscreen = True
    logbook.resoln = c/args.vres if args.spec_smear else c/args.vdisp
    
    wave = np.array(properties.dispsol)  #converting to numpy arrays
    flam = np.array(properties.ppvcube[args.X,args.Y,:]) #in units ergs/s/A/pc^2
    
    lastgood = flam[0]
    for ind,ff in enumerate(flam):
        if not np.isnan(ff):
            lastgood = ff
            continue
        else: flam[ind] = lastgood #to get rid of NANs in flam

    cont = fitcont(wave, flam, args, logbook) #cont in units ergs/s/A/pc^2
    flam /= cont #flam = dimensionless
    #-------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(17,5))
    fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.05, right=0.95)
    plt.plot(wave, flam*cont, c='k', label='Spectrum at '+str(args.X)+', '+str(args.Y))
    for ii in logbook.wlist:
        plt.axvline(ii,ymin=0.9,c='black')    
    plt.ylabel('flam in erg/s/A/pc^2')
    plt.xlabel('Wavelength (A)')
    plt.xlim(logbook.wmin, logbook.wmax)
    plt.title(logbook.fitsname +'\n'+'Fitted spectrum at pp '+str(args.X)+','+str(args.Y))
    plt.legend()
    if args.hide: plt.close()
    else: plt.show(block=False)
    #-------------------------------------------------------------------------------------------
    flux, flux_errors = fit_all_lines(args, logbook, wave, flam, cont, args.X, args.Y, z=0., z_err=0.0001)
    print 'Flux=', flux
    print 'Flux error=', flux_errors
    print np.shape(properties.ppvcube), args.X, args.Y, args.SNR_thresh, flux/flux_errors #
    return flux, flux_errors
#-------------------------------------------------------------------------------------------
def fitcont(wave, flux, args, logbook):
    flux_masked = flux
    wave_masked = wave
    for thisw in logbook.wlist:
        linemask_width = thisw * 1.5*args.vmask/c
        flux_masked = np.ma.masked_where(np.abs(thisw - wave) <= linemask_width, flux_masked)
        wave_masked = np.ma.masked_where(np.abs(thisw - wave) <= linemask_width, wave_masked)
    
    wave_masked = np.ma.masked_where(np.isnan(flux_masked), wave_masked)
    flux_masked = np.ma.masked_where(np.isnan(flux_masked), flux_masked)
    
    #if args.addnoise:
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
    #else:
    #--option 2: spline fit------
    boxcar = 11
    flux_smoothed = con.convolve(np.array(flux), np.ones((boxcar,))/boxcar, boundary='fill', fill_value=np.nan)
    flux_smoothed = np.ma.masked_where(np.ma.getmask(flux_masked), flux_smoothed)
    cont = pd.DataFrame(flux_masked).interpolate(method='cubic').values.ravel().tolist()
    
    #---option 3: legendre fit------
    #leg = np.polynomial.legendre.Legendre.fit(wave_masked, flux_masked, 4)
    #cont = np.polynomial.legendre.legval(wave, leg.coef)
    
    #---option 4: basis spline fit--------
    #contfunc = list(si.splrep(wave_masked, flux_masked, k=5))
    #cont = si.splev(wave, contfunc)
    
    if 'fixfit' in args and args.fixfit:
        fig = plt.figure(figsize=(14,5))
        plt.plot(wave, cont, c='g', label='cont')
        plt.plot(wave_masked, flux_masked, c='r', label='masked flux')
        plt.plot(wave, flux_smoothed, c='k', linewidth=1, label='smoothed flux')
        plt.legend()
        plt.xlabel('Obs wavelength (A)')
        plt.ylabel('flambda (ergs/s/A/pc^2)')
        plt.title('Testing continuum fit for pixel ('+str(args.X)+','+str(args.Y)+')')
        #plt.ylim(0.8e-19, 1.05e-19) #
        plt.show(block=False)
        sys.exit()#
    
    return np.array(cont)
#-------------------------------------------------------------------------------------------
def calcpos(s, galsize, res):
    g = int(np.ceil(galsize/res))
    x = s['x'] * args.base_res #x should range from 0 to galsize, s['x'] ranges from 0 to ~1300 as the base resolution is 20pc
    y = s['y'] * args.base_res
    return g, x, y
#-------------------------------------------------------------------------------------------
def make2Dmap(data, xi, yi, ngrid, res, domean=False, islog=False, weights=None):
    map = np.zeros((ngrid,ngrid))
        
    for i in range(len(data)):
        x = int(xi[i]/res)
        y = int(yi[i]/res)
        if islog: quantity = 10.**data[i]
        else: quantity = data[i]
        
        if weights is not None: quantity *= weights[i]        

        map[x][y] += quantity        

    if domean: map = np.divide(map, make2Dmap(np.ones(len(data)), xi, yi, ngrid, res))
    elif weights is not None: map = np.divide(map, make2Dmap(weights, xi, yi, ngrid, res))
        
    map[np.isnan(map)] = 0
    return map 
#-------------------------------------------------------------------------------------------
def getskynoise(args, logbook, properties):
    #to read in sky noise files in physical units, convert to el/s, spectral-bin them and create map
    bluenoise = fits.open(HOME+'/models/Noise_model/NoiseData-99259_B.fits')[1].data[0]
    rednoise  = fits.open(HOME+'/models/Noise_model/NoiseData-99259_R.fits')[1].data[0]
    skywave = np.hstack((bluenoise[0], rednoise[0])) #in Angstrom
    noise = np.hstack((bluenoise[1], rednoise[1])) #in 10^-16 ergs/s/cm^2/A/spaxel   
    noise[noise > 100.] = 0. #replacing insanely high noise values
    noise = 1e-16 * np.multiply(noise, skywave)# to convert it to ergs/s/cm^2, as the flux values are
    factor = (np.pi*args.rad*1e2*logbook.final_pix_size/args.galsize)**2 * args.el_per_phot / (planck * nu) #skynoise for each cm^2 spread over entire collecting area of telescope (pi*args.rad*100^2) divided by total no. of cells
                                                                                                        #do we need to multiply with flux_ratio?? Think!
    noise *= factor #to transform into electrons/s from physical units, because we do not know the exposure time for the skynoise provided by Rob Sharp
    f = interp1d(skywave, noise, kind='cubic')
    wave = np.array(properties.dispsol)
    smallwave = wave[(wave >= np.min(skywave)) & (wave <= np.max(skywave))]
    interp_noise = f(smallwave)
    interp_noise = np.lib.pad(interp_noise, (len(np.where(wave<skywave[0])[0]), len(np.where(wave>skywave[-1])[0])), 'constant', constant_values=(0,0))
    return interp_noise
#-------------------------------------------------------------------------------------------
def plotmap(map, title, savetitle, cbtitle, args, logbook, islog=True):    
    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.1, left=0.1, right=0.9)
    ax = plt.subplot(111)
    if islog:
        map = np.ma.masked_where(map<0, map)
        cmin = np.min(np.log10(map)) if args.cmin is None else args.cmin
        cmax = np.max(np.log10(map)) if args.cmax is None else args.cmax 
        map = np.ma.masked_where(np.log10(map)<cmin, map)
        p = ax.imshow(np.log10(map), cmap='rainbow',vmin=cmin,vmax=cmax)
    else:
        if 'SNR' in title:
            cmin = 0. if args.cmin is None else args.cmin
            cmax = 50. if args.cmax is None else args.cmax  
        else:
            cmin = np.min(np.ma.masked_where(map<0, map)) if args.cmin is None else args.cmin
            cmax = np.max(np.ma.masked_where(map<0, map)) if args.cmax is None else args.cmax  
        map = np.ma.masked_where(map<cmin, map)
        p = ax.imshow(map, cmap='rainbow',vmin=cmin,vmax=cmax)
    ax.set_xticklabels([i*logbook.final_pix_size - args.galsize/2 for i in list(ax.get_xticks())])
    ax.set_yticklabels([i*logbook.final_pix_size - args.galsize/2 for i in list(ax.get_yticks())])
    plt.ylabel('y(kpc)')
    plt.xlabel('x(kpc)')
    plt.title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(p, cax=cax).set_label(cbtitle)    
    #circle1 = plt.Circle((galsize/(2*res), galsize/(2*res)), galsize/(2*res), color='k')#
    #ax.add_artist(circle1)#
    if args.saveplot: fig.savefig(args.path+savetitle+'.png')
    if args.hide: plt.close(fig)
    else: plt.show(block=False)
    return map
#-------------------------------------------------------------------------------------------
def getfn(args):
    return HOME+'/models/emissionlist'+args.outtag+'/emissionlist_'+args.file+'_Om'+str(args.Om)+'.txt'
#-------------------------------------------------------------------------------------------
def getsmoothparm(args, properties, logbook):
    if args.parm is None:
        args.pow, args.size = 4.7, 5 #power and sigma parameters for 2D Moffat kernal in pixel units
    else:
        args.pow, args.size = args.parm[0], int(args.parm[1])
    #------------------------------------------------------------------
    #-----Compute effective seeing i.e. FWHM as: the resolution on sky (res_arcsec) -> physical resolution on target frame (res_phys) -> pixel units (fwhm)
    fwhm = min(properties.res_phys/args.res, args.pix_per_beam) #choose the lesser of 10 pixels per beam or res/res_phys pixels per beam
    logbook.new_res = properties.res_phys/fwhm #need to re-bin the map such that each pixel is now of size 'new_res' => we have 'fwhm' pixels within every 'res_phys' physical resolution element
    dummy = np.zeros((int(args.galsize/args.res),int(args.galsize/args.res))) #dummy array to check what actual resolution we are left with after rebinning
    dummy = rebin(dummy, args.res, logbook.new_res) #mock rebin
    logbook.final_pix_size = args.galsize/np.shape(dummy)[0] #actual pixel size we will end up with
    logbook.fwhm = int(np.round(properties.res_phys/logbook.final_pix_size)) #actual no. of pixels per beam we will finally have
    #------------------------------------------------------------------
    if args.ker == 'gauss': logbook.sig = int(gf2s * logbook.fwhm)
    elif args.ker == 'moff': logbook.sig = int(np.round(logbook.fwhm/(2*np.sqrt(2**(1./args.pow)-1.))))
    logbook.size = logbook.sig*args.size
    if logbook.size%2 == 0: logbook.size += 1 #because kernels need odd integer as size
    return args, logbook
#-------------------------------------------------------------------------------------------
def getfitsname(args, properties):
    global info
    logbook = ap.Namespace()
    logbook.wlist, logbook.llist = readlist()
    logbook.wmin = logbook.wlist[0]-50. if args.wmin is None else args.wmin
    logbook.wmax = logbook.wlist[-1]+50. if args.wmax is None else args.wmax
    logbook.llist = logbook.llist[np.where(np.logical_and(logbook.wlist > logbook.wmin, logbook.wlist < logbook.wmax))] #truncate linelist as per wavelength range
    logbook.wlist = logbook.wlist[np.where(np.logical_and(logbook.wlist > logbook.wmin, logbook.wlist < logbook.wmax))]
    info = ''
    if args.spec_smear: info += '_specsmeared_'+str(int(args.vres))+'kmps'
    info1 = info
    if args.smooth:
        arc_string = ',arc='+str(args.res_arcsec)
        args, logbook = getsmoothparm(args, properties, logbook)
        info += '_smeared_'+args.ker+'_parm'+str(logbook.fwhm)+','+str(logbook.sig)+','+str(args.pow)+','+str(logbook.size)
    else:
        logbook.final_pix_size = args.res
        arc_string = ''
    info2 = info
    if args.addnoise: info += '_noisy'
    if args.fixed_SNR is not None: info += '_fixed_SNR'+str(args.fixed_SNR)
    if not args.maketheory:
        info+= '_obs'
        if args.exptime is not None:
            logbook.exptime = float(args.exptime)
        else:
            if args.scale_exptime: scalefactor = float(args.scale_exptime)
            elif args.scale_exp_SNR: scalefactor = 1200. * float(args.scale_exp_SNR)
            else: scalefactor = 1200. #sec
            lambda_spacing = args.vres * 6562./c if args.spec_smear else 10*args.vdel*6562./(args.nhr*c)
            logbook.exptime = float(scalefactor)*(0.04/logbook.final_pix_size)**2 * (0.6567/lambda_spacing) #increasing exposure time quadratically with finer resolution
                                                                                                            #scaled to <scale_factor> sec for 1" (=0.04kpc cell size) spatial and 30kmps spectral resolution

        info += '_exp'+str(int(logbook.exptime))+'s'
    if args.multi_realisation: info += '_real'+str(args.multi_realisation)
    
    logbook.H2R_filename = args.path + 'H2R_'+args.file+'Om='+str(args.Om)+'_'+str(logbook.wmin)+'-'+str(logbook.wmax)+'A' + info1+ args.gradtext+'.fits'
    logbook.skynoise_cubename = args.path + 'skycube_'+'pixsize_'+str(logbook.final_pix_size)+'_'+str(logbook.wmin)+'-'+str(logbook.wmax)+'A'+info1+'.fits'
    logbook.convolved_filename = args.path + 'convolved_'+args.file+'Om='+str(args.Om)+arc_string+'_'+str(logbook.wmin)+'-'+str(logbook.wmax)+'A' + info2+ args.gradtext +'.fits'
    logbook.fitsname = args.path + 'PPV_'+args.file+'Om='+str(args.Om)+arc_string+'_'+str(logbook.wmin)+'-'+str(logbook.wmax)+'A' + info+ args.gradtext +'.fits'
    if args.debug: print '\nDeb1023: logbook = ', logbook, '\n'
    
    return args, logbook
#-------------------------------------------------------------------------------------------
def write_fits(filename, data, args, fill_val=np.nan):
    hdu = fits.PrimaryHDU(np.ma.filled(data,fill_value=fill_val))
    hdulist = fits.HDUList([hdu])
    if filename[-5:] != '.fits':
        filename += '.fits'
    hdulist.writeto(filename, clobber=True)
    if not args.silent: myprint('Written file '+filename+'\n', args)    
#-------------------------------------------------------------------------------------------
def rebin(map, current_res, final_res):
    ratio = final_res/current_res
    new_s = min(factors(np.shape(map)[0]), key=lambda x:abs(x-np.shape(map)[0]/ratio)) #to get closest factor of initial size of array to re-bin into
    shape = (new_s, new_s)
    sh = shape[0],map.shape[0]/shape[0],shape[1],map.shape[1]/shape[1]
    return map.reshape(sh).sum(-1).sum(1)
#------to get fraction of non zero pixels in line map above given SNR within given radius------
def get_valid_frac(map, map_u, radius, args):
    x,y = np.ogrid[:np.shape(map)[0],:np.shape(map)[1]]
    dist = np.sqrt((x-np.shape(map)[0]/2)**2 + (y-np.shape(map)[1]/2)**2)
    radius = (radius/args.galsize) * np.shape(map)[0]
    map = np.ma.masked_where(dist > radius, map)
    ndatapoints = len(map.nonzero()[0])
    map = np.ma.masked_where(map/map_u < args.SNR_thresh, map)
    nvalid = len(map.nonzero()[0])
    return float(nvalid)/float(ndatapoints) #fraction
#-------------------------------------------------------------------------------------------
def rebin_old(map, shape):
    sh = shape[0],map.shape[0]/shape[0],shape[1],map.shape[1]/shape[1]
    return map.reshape(sh).sum(-1).sum(1)
#-------------------------------------------------------------------------------------------
def calc_dist(z, H0 = 70.):
    dist = z*c/H0 #Mpc
    return dist
#-------------------------------------------------------------------------------------------
def factors(n):    
    return list(reduce(list.__add__,([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
#-------------------------------------------------------------------------------------------
def masked_data(data):
    data = np.ma.masked_where(np.isnan(data), data)
    #data = np.ma.masked_where(data<=0, data)
    return data
#-------------------------------------------------------------------------------------------
def mydiag(title, data, args):
    myprint(title+': Mean, stdev, max, min= '+str(np.mean(masked_data(data)))+','+str(np.std(masked_data(data)))+','+\
    str(np.max(masked_data(data)))+','+str(np.min(masked_data(data)))+'\n', args)
#-------------------------------------------------------------------------------------------
def myprint(text, args):
    if args.toscreen or args.debug: print text
    else:
        ofile = open(args.outfile,'a')
        ofile.write(text)
        ofile.close()
#-------------------------------------------------------------------------------------------
def myexit(args, text=''):
    myprint(text+' Exiting by encountering sys.exit() in the code.', args)
    sys.exit()
#-------------------End of functions------------------------------------------------------------------------
#-------------------Begin main code------------------------------------------------------------------------
global info
col_ar=['m','blue','steelblue','aqua','lime','darkolivegreen','goldenrod','orangered','darkred','dimgray']
logOHsun = 8.77
c = 3e5 #km/s
H0 = 70. #km/s/Mpc Hubble's constant
planck = 6.626e-27 #ergs.sec Planck's constant
nu = 5e14 #Hz H-alpha frequency to compute photon energy approximately
f_esc = 0.0
f_dust = 0.0
const = 1e0 #to multiply with nebular flux to make it comparable with SB continuum
if __name__ == '__main__':
    properties = ap.Namespace()
    #-------------------arguments parsed-------------------------------------------------------
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
    parser.add_argument('--showplot', dest='showplot', action='store_true')
    parser.set_defaults(showplot=False) #to show spectrum fitting plot
    parser.add_argument('--nomap', dest='nomap', action='store_true')
    parser.set_defaults(nomap=False)

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
    parser.add_argument("--base_res")
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
    parser.add_argument("--cmin")
    parser.add_argument("--cmax")
    parser.add_argument("--rad")
    parser.add_argument("--gain")
    parser.add_argument("--exptime")
    parser.add_argument("--epp")
    parser.add_argument("--snr")
    parser.add_argument("--Zgrad")
    parser.add_argument("--fixed_SNR")
    parser.add_argument("--ncores")
    parser.add_argument("--galsize")
    parser.add_argument("--outtag")
    parser.add_argument("--oneHII")
    parser.add_argument("--vmask")
    args, leftovers = parser.parse_known_args()
    if args.debug: #debug mode over-rides 
        args.toscreen = True
        args.silent = False
        args.hide = False
        args.calcgradient = True
        args.nowrite = True
        args.saveplot = False

    if args.debug: myprint('Starting in debugging mode. Brace for storm of stdout statements and plots...\n', args)
    if args.outtag is None: args.outtag = '_sph_logT4.0_MADtemp_Z0.05,5.0_age0.0,5.0_lnII5.0,12.0_lU-4.0,-1.0_4D'

    if args.galsize is not None: 
        args.galsize = float(args.galsize)
    else:
        args.galsize = 26.0 #kpc 

    if args.path is None:
        args.path = HOME+'/Desktop/bpt/'
    subprocess.call(['mkdir -p '+args.path],shell=True) #create output directory if it doesn't exist

    if args.file is None:
        args.file = 'DD0600_lgf' #which simulation to use

    if args.Om is not None:
        args.Om = float(args.Om)
    else:
        args.Om = 0.5

    if args.ppb is not None:
        args.pix_per_beam = int(args.ppb)
    else:
        args.pix_per_beam = 10

    if args.gain is not None:
        args.gain = float(args.gain)
    else:
        args.gain = 1.5

    if args.epp is not None:
        args.el_per_phot = float(args.epp)
    else:
        args.el_per_phot = 0.5 #instrumental throughput, not all photons get converted to electrons

    if args.z is not None:
        args.z = float(args.z)
    else:
        args.z = 0.013

    if args.rad is not None:
        args.rad = float(args.rad)
    else:
        args.rad = 1. #metre

    properties.dist = calc_dist(args.z) #distance to object; in kpc
    properties.flux_ratio = (args.rad/(2*properties.dist*3.086e22))**2 #converting emitting luminosity to luminosity seen from earth, 3.08e19 factor to convert kpc to m
    if args.debug:
        myprint('Deb1272: Distance for z= '+str(args.z)+' is %.4F Mpc\n'%properties.dist, args)
        myprint('Deb1273: Flux ratio= (Radius= %.2F / (2 * dist= %.4E * 3.086e22))^2'%(args.rad, properties.dist), args)
        myprint('Deb1274: Flux ratio= %.4E'%properties.flux_ratio, args)

    if args.base_res is not None:
        args.base_res = float(args.base_res)
    else:
        args.base_res = 0.02 #kpc: simulation actual base resolution, for NG's Enzo sim base_res = 20pc
    
    if args.res is not None:
        args.res = float(args.res)
    else:
        args.res = args.base_res #kpc: input resolution to constructPPV cube, usually same as base resolution of simulation 
    
    if args.arc is not None:
        args.res_arcsec = float(args.arc)
    else:
        args.res_arcsec = 0.5 #arcsec
    
    properties.res_phys = args.res_arcsec*np.pi/(3600*180)*(properties.dist*1e3) #kpc

    if args.line is None:
        args.line = 'H6562'# #whose emission map to be made

    if args.cmin is not None:
        args.cmin = float(args.cmin)
    else:
        args.cmin = None

    if args.cmax is not None:
        args.cmax = float(args.cmax)
    else:
        args.cmax = None

    if args.nhr is not None:
        args.nhr = int(args.nhr)
    else:
        args.nhr = 100 # no. of bins used to resolve the range lamda +/- 5sigma around emission lines

    if args.nbin is not None:
        args.nbin = int(args.nbin)
    else:
        args.nbin = 50 #no. of bins used to bin the continuum into (without lines)

    if args.nres is not None:
        args.nres = int(args.nres)
    else:
        args.nres = 10 #no. of spectral resolution elements included on either side during fitting a line/group of lines

    if args.vdisp is not None:
        args.vdisp = float(args.vdisp)
    else:
        args.vdisp = 15. #km/s vel dispersion to be added to emission lines from MAPPINGS while making PPV

    if args.vdel is not None:
        args.vdel = float(args.vdel)
    else:
        args.vdel = 100. #km/s; vel range in which spectral resolution is higher is sig = 5*vdel/c
                    #so wavelength range of +/- sig around central wavelength of line is binned into further nhr bins

    if args.vres is not None:
        args.vres = float(args.vres)
    else:
        args.vres = 30. #km/s instrumental vel resolution to be considered while making PPV

    if args.wmin is not None:
        args.wmin = float(args.wmin)
    else:
        args.wmin = None #Angstrom; starting wavelength of PPV cube

    if args.wmax is not None:
        args.wmax = float(args.wmax)
    else:
        args.wmax = None #Angstrom; ending wavelength of PPV cube
    
    if args.snr is not None:
        args.SNR_thresh = float(args.snr)
    else:
        args.SNR_thresh = None

    if not args.keepprev:
        plt.close('all')

    if args.parm is not None:
        args.parm = [float(ar) for ar in args.parm.split(',')]
    else:
        args.parm = None # set of parameters i.e. telescope properties to be used for smearing cube/map
        
    if args.Zgrad is not None:
        args.logOHcen, args.logOHgrad = [float(ar) for ar in args.Zgrad.split(',')]
        args.gradtext = '_Zgrad'+str(args.logOHcen)+','+str(args.logOHgrad)
    else:
        args.logOHcen, args.logOHgrad = logOHsun, 0.
        args.gradtext = ''
    args.outtag = args.gradtext + args.outtag
    
    if args.ker is not None:
        args.ker = args.ker
    else:
        args.ker = 'moff' # convolution kernel to be used for smearing cube/map
        
    if args.fixed_SNR is not None:
        args.fixed_SNR = float(args.fixed_SNR) #fixed SNR used in makenoisy() function
    else:
        args.fixed_SNR = None 

    if args.ncores is not None:
        args.ncores = int(args.ncores) #number of cores used in parallel segments
    else:
        args.ncores = mp.cpu_count()/2

    if args.oneHII is not None:
        args.oneHII = int(args.oneHII) #serial no. of HII region to be used for the one HII region test case
    else:
        args.oneHII = None 

    if args.vmask is not None:
        args.vmask = float(args.vmask) #vel width to mask around nebular lines, before continuum fitting
    else:
        args.vmask = 100. #km/s 

    if args.useKD:
        args.num_arr = [['NII6584']]
        args.den_arr = [['OII3727']]
        args.ratio_lim = [(0,2)]
    else:
        args.num_arr = [['NII6584'], ['NII6584']]
        args.den_arr = [['SII6717', 'SII6730'], ['H6562']]
        args.ratio_lim = [(0,2), (0,0.5)]
    #-------------------------------------------------------------------------------------------
    args, logbook = getfitsname(args, properties) # name of fits file to be written into
    #-------------------------------------------------------------------------------------------
    if args.outfile is None:
        args.outfile = logbook.fitsname.replace('PPV','output_PPV')[:-5]+'.txt' # name of fits file to be written into
    #------------write starting conditions in output txt file or stdout-------------
    starting_text = ''
    if not len(sys.argv) > 1:
        starting_text += 'Insuffiecient information. Here is an example how to this routine might be called:\n'
        starting_text += 'run plotobservables.py --addnoise --smooth --keep --vres 600 --spec_smear --plotspec\n'
    else:
        starting_text += 'Path: '+args.path+' Use --path option to specify.'+'\n'
        starting_text += 'Outfile: '+args.outfile+' Use --outfile option to specify.'+'\n'
        starting_text += 'Simulation= '+args.file+'. Use --file option to specify.'+'\n'
        starting_text += 'Omega= '+str(args.Om)+'. Use --om option to specify Omega. You can supply , separated multiple omega values.'+'\n'
        starting_text += 'Maximum pix_per_beam of '+str(args.pix_per_beam)+'. Use --ppb option to specify pix_per_beam.'+'\n'
        starting_text += 'Gain= '+str(args.gain)+'. Use --gain option to specify gain.'+'\n'
        starting_text += 'Electrons per photon= '+str(args.el_per_phot)+'. Use --epp option to specify el_per_phot.'+'\n'
        starting_text += 'Redshift of '+str(args.z)+'. Use --z option to specify redshift.'+'\n'
        starting_text += 'Telescope mirror radius= '+str(args.rad)+' m. Use --rad option to specify radius in metres.'+'\n'
        starting_text += 'Simulation resolution= '+str(args.res)+' kpc. Use --res option to specify simulation resolution.'+'\n'
        starting_text += 'Telescope spatial resolution= '+str(args.res_arcsec)+'. Use --arc option to specify telescope resolution.'+'\n'
        starting_text += 'Resolution of telescope on object frame turns out to be res_phys~'+str(properties.res_phys)+' kpc.'+'\n'
        if args.exptime: starting_text += 'Exposure time set to '+str(logbook.exptime)+' seconds. Use --exptime option to specify absolute exposure time in seconds.'+'\n'
        else: starting_text += 'Exposure time scaled to '+str(logbook.exptime)+' seconds. Use --scale_exptime option to specify scale factor in seconds.'+'\n'
        starting_text += 'Line:'+args.line+'. Use --line option to specify line.'+'\n'
        starting_text += 'No. of bins used to resolve+/- 5sigma around emission lines= '+str(args.nhr)+'. Use --nhr to specify.'+'\n'
        starting_text += 'No. of bins used to bin the continuum into (without lines)= '+str(args.nbin)+'. Use --nbin to specify.'+'\n'
        starting_text += 'Velocity dispersion to be added to emission lines= '+str(args.vdisp)+' km/s.'+'. Use --vdisp to specify.'+'\n'
        starting_text += 'Velocity range in which spectral resolution is higher around central wavelength of line= '+str(args.vdel)+' km/s.'+'. Use --vdel to specify.'+'\n'
        starting_text += 'Instrumental velocity resolution to be considered while making PPV= '+str(args.vres)+' km/s.'+'. Use --vres to specify.'+'\n'
        if args.wmin: starting_text += 'Starting wavelength of PPV cube= '+str(args.wmin)+' A.'+'\n'
        else: starting_text += 'Starting wavelength of PPV cube at beginning of line list.'+'\n'
        if args.wmax: starting_text += 'Ending wavelength of PPV cube= '+str(args.wmax)+' A.'+'\n'
        else: starting_text += 'Ending wavelength of PPV cube at end of line list.'+'\n'
        if args.snr: starting_text += 'Applying SNR cut-off= '+str(args.SNR_thresh)+' on fitted lines.'+'\n'
        else: starting_text += 'No SNR cut-off will be applied.'+'\n'
        starting_text += 'Will run the parallel segments on '+str(args.ncores)+' cores.'+'\n'
        if args.smooth: 
            if args.parm: starting_text += 'Parameter for smoothing= '+str(args.parm[0])+', '+str(args.parm[1])+'\n'
            else: starting_text += 'Default smoothing parameter settings. Use --parm option to specify smearing parameters set.'+'\n'
            if args.ker: starting_text += 'Smoothing profile used: '+args.ker+'\n'
            else: starting_text += 'Default Moffat profile for smoothing.'+'\n'
        if args.Zgrad: starting_text += 'Using metallicity painted HII regions, with central logOH+12 = '+str(args.logOHcen)+', and gradient = '+str(args.logOHgrad)+' dex per kpc'+'\n'  
        else: starting_text += 'No additional metallicity gradient painted.'+'\n'
        starting_text += '\n'
        starting_text += 'Will be using/creating '+logbook.H2R_filename+' file.'+'\n'
        starting_text += 'Will be using/creating '+logbook.skynoise_cubename+' file.'+'\n'
        starting_text += 'Will be using/creating '+logbook.convolved_filename+' file.'+'\n'
        starting_text += 'Will be using/creating '+logbook.fitsname+' file.'+'\n'

    myprint(starting_text, args)
    #-------------------------------------------------------------------------------------------------------
    logbook.fittedcube = logbook.fitsname.replace('PPV','fitted-map-cube') # name of mapcube file to be read in       
    logbook.s = ascii.read(getfn(args), comment='#', guess=False)
    if args.toscreen and args.smooth: print 'Deb1432: res_phys, final pix per beam, final pix size, final shape=', properties.res_phys, logbook.fwhm, logbook.final_pix_size, args.galsize/logbook.final_pix_size, 'kpc' #
    if args.debug: mydiag('Deb1437: for H2R Ha luminosity: in ergs/s:', logbook.s['H6562'], args)
    #-----------------------jobs fetched--------------------------------------------------------------------
    if args.get_scale_length: properties.scale_length = get_scale_length(args, logbook)
    elif args.inspect and not args.met: axes = inspectmap(args, logbook, properties)
    elif args.ppv: properties.ppvcube = spec(args, logbook, properties)       
    else:
        if not os.path.exists(logbook.fitsname):
            if not args.silent: myprint('PPV file does not exist. Creating ppvcube..'+'\n', args)
            properties.ppvcube = spec(args, logbook, properties)       
        else:
            if not args.silent: myprint('Reading existing ppvcube from '+logbook.fitsname+'\n', args)
            properties.ppvcube = fits.open(logbook.fitsname)[0].data
        properties = get_disp_array(args, logbook, properties)
        
        if  (not args.plotintegmap) * (not args.plotspec) * (not args.fixfit) == 0:
            if args.X is not None:
                args.X = int(args.X)
            else:
                args.X = int(logbook.s['x'][args.oneHII if args.oneHII is not None else 0]*0.02/args.res) #p-p values at which point to extract spectrum from the ppv cube
            if args.plotspec and not args.silent: myprint('X position at which spectrum to be plotted= '+str(args.X)+'\n', args)

            if args.Y is not None:
                args.Y = int(args.Y)
            else:
                args.Y = int(logbook.s['y'][args.oneHII if args.oneHII is not None else 0]*0.02/args.res) #p-p values at which point to extract spectrum from the ppv cube
            if args.plotspec and not args.silent: myprint('Y position at which spectrum to be plotted= '+str(args.Y)+'\n', args)
            
            if args.plotintegmap:
                dummy = plotintegmap(args, logbook, properties)
            elif args.plotspec:
                dummy = spec_at_point(args, logbook, properties)
            elif args.fixfit:
                dummy, dummy_u = fixfit(args, logbook, properties)
    
        else:
            logbook.fittederror = logbook.fittedcube.replace('map','error')
            if os.path.exists(logbook.fittedcube) and not args.clobber:
                if not args.silent: myprint('Reading existing mapcube from '+logbook.fittedcube+'\n', args)
            else:
                if not args.silent: myprint('Mapfile does not exist. Creating mapcube..'+'\n', args)
                
                if args.spec_smear: smear = ' --spec_smear'
                else: smear = ''
                if args.silent: silent = ' --silent'
                else: silent = ''
                if args.toscreen: toscreen = ' --toscreen'
                else: toscreen = ''
                if args.debug: debug = ' --debug'
                else: debug = ''
                if args.showplot: showplot = ' --showplot'
                else: showplot = ''
                if args.oneHII is not None: oneHII = ' --oneHII '+str(args.oneHII)
                else: oneHII = ''
                if args.addnoise: addnoise = ' --addnoise'
                else: addnoise = ''
                
                funcname = HOME+'/Work/astro/ayan_codes/enzo_model_code/parallel_fitting_old.py'
                command = 'mpirun -np '+str(args.ncores)+' python '+funcname+' --fitsname '+logbook.fitsname+' --nbin '+str(args.nbin)+\
                ' --vdel '+str(args.vdel)+' --vdisp '+str(args.vdisp)+' --vres '+str(args.vres)+' --nhr '+str(args.nhr)+' --wmin '+\
                str(logbook.wmin)+' --wmax '+str(logbook.wmax)+' --fittedcube '+logbook.fittedcube+' --fittederror '+logbook.fittederror+\
                ' --outfile '+args.outfile+' --nres '+str(args.nres) + ' --vmask ' +str(args.vmask) + smear + silent + toscreen \
                + debug + showplot + oneHII + addnoise
                subprocess.call([command],shell=True)
                
            properties.mapcube = fits.open(logbook.fittedcube)[0].data
            if args.SNR_thresh is not None: properties.errorcube = fits.open(logbook.fittederror)[0].data
            else: properties.errorcube = None
            
            if args.bptpix: 
                bpt_pixelwise(args, logbook, properties)
            elif args.met: 
                if args.inspect: axes = inspectmap(args, logbook, properties)
                else: axes = None
                properties = metallicity(args, logbook, properties, axes=axes)
            elif args.map: 
                properties.map = emissionmap(args, logbook, properties)
            elif args.sfr: 
                properties.SFRmap_real, properties.SFRmapHa = SFRmaps(args, logbook, properties)
            else: 
                if not args.silent: myprint('Wrong choice. Choose from:\n --bptpix, --map, --sfr, --met, --ppv, --plotinteg, --plotspec'+'\n', args)
        
    #-------------------------------------------------------------------------------------------
    if args.hide: plt.close()
    else: plt.show(block=False)
    if not args.silent and args.saveplot: myprint('Saved plot here: '+path+'\n', args)
    myprint('Completed in %s minutes\n' % ((time.time() - start_time)/60), args)
