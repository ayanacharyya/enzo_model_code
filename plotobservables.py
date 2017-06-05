import time
start_time = time.time()
import numpy as np
import subprocess
from matplotlib import pyplot as plt
from astropy.io import ascii, fits
import os
HOME = os.getenv('HOME')
import sys
sys.path.append(HOME+'/Work/astro/mageproject/ayan/')
sys.path.append(HOME+'/models/enzo_model_code/')
import splot_util as su
import parallel_convolve as pc
import parallel_fitting as pf
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
def gauss(w, f, w0, f0, v, vz):
    w0 = w0*(1+vz/c) #stuff for v_z component of HII region
    sigma = w0*v/c #c=3e5 km/s
    g = (f0/np.sqrt(2*np.pi*sigma**2))*np.exp(-((w-w0)**2)/(2*sigma**2))    
    f += g
    return f
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
#------------Function to measure metallicity-------------------------------------------------------------------------------
def metallicity(args, logbook, properties):
    global info
    mapn2 = properties.mapcube[:,:,np.where(logbook.llist == 'NII6584')[0][0]]
    maps2a = properties.mapcube[:,:,np.where(logbook.llist == 'SII6717')[0][0]]
    maps2b = properties.mapcube[:,:,np.where(logbook.llist == 'SII6730')[0][0]]
    mapha = properties.mapcube[:,:,np.where(logbook.llist == 'H6562')[0][0]]
    #mapo2 = properties.mapcube[:,:,np.where(logbook.llist == 'OII3727')[0][0]]
    mapn2 = np.ma.masked_where(mapn2<=0., mapn2)
    maps2a = np.ma.masked_where(maps2a<=0., maps2a)
    maps2b = np.ma.masked_where(maps2b<=0., maps2b)
    mapha = np.ma.masked_where(mapha<=0., mapha)
    #mapo2 = np.ma.masked_where(mapo2<=0., mapo2)

    if args.SNR_thresh is not None:
        #loading all the flux uncertainties
        mapn2_u = properties.errorcube[:,:,np.where(logbook.llist == 'NII6584')[0][0]]
        maps2a_u = properties.errorcube[:,:,np.where(logbook.llist == 'SII6717')[0][0]]
        maps2b_u = properties.errorcube[:,:,np.where(logbook.llist == 'SII6730')[0][0]]
        mapha_u = properties.errorcube[:,:,np.where(logbook.llist == 'H6562')[0][0]]
        #mapo2_u = errorcube[:,:,np.where(logbook.llist == 'OII3727')[0][0]]
        mapn2_u = np.ma.masked_where(mapn2_u<=0., mapn2_u)
        maps2a_u = np.ma.masked_where(maps2a_u<=0., maps2a_u)
        maps2b_u = np.ma.masked_where(maps2b_u<=0., maps2b_u)
        mapha_u = np.ma.masked_where(mapha_u<=0., mapha_u)
        #mapo2_u = np.ma.masked_where(mapo2_u<=0., mapo2_u)        
        #imposing SNR cut
        mapn2 = np.ma.masked_where(mapn2/mapn2_u <args.SNR_thresh, mapn2)
        maps2a = np.ma.masked_where(maps2a/maps2a_u <args.SNR_thresh, maps2a)
        maps2b = np.ma.masked_where(maps2b/maps2b_u <args.SNR_thresh, maps2b)
        mapha = np.ma.masked_where(mapha/mapha_u <args.SNR_thresh, mapha)
        #mapo2 = np.ma.masked_where(mapo2/mapo2_u <args.SNR_thresh, mapo2)
        if args.toscreen:
            print 'Minimum (of all line maps used) fraction of non-zero cells above SNR_thresh= '+str(args.SNR_thresh)+' is %.2F\n'\
            %min(float(mapn2.count())/float(np.count_nonzero(mapn2)), float(maps2a.count())/float(np.count_nonzero(maps2a)), \
            float(maps2b.count())/float(np.count_nonzero(maps2b)), float(mapha.count())/float(np.count_nonzero(mapha)))
        
    g = np.shape(properties.mapcube)[0]
    b = np.linspace(-g/2 + 1,g/2,g)*(args.galsize)/g #in kpc
    d = np.sqrt(b[:,None]**2+b**2)
    t = args.file+':Met_Om'+str(args.Om)+'_arc'+str(args.res_arcsec)+'"'+'_vres='+str(args.vres)+'kmps_'+info+args.gradtext
    if args.SNR_thresh is not None: t += '_snr'+str(args.SNR_thresh)
    
    if args.useKD:
        #--------From Kewley 2002------------------
        logOHsol = 8.93 #log(O/H)+12 value used for Solar metallicity in earlier MAPPINGS in 2001
        log_ratio = np.log10(np.divide(mapn2,mapo2))
        logOHobj_map = np.log10(1.54020 + 1.26602*log_ratio + 0.167977*log_ratio**2) #+ 8.93
        myprint('log_ratio med, min '+str(np.median(log_ratio))+' '+str(np.min(log_ratio))+'\n', args)
        myprint('logOHobj_map before conversion med, min '+str(np.median(logOHobj_map))+' '+str(np.min(logOHobj_map))+'\n', args)
        t += '_KD02'
    else:
        #-------From Dopita 2016------------------------------
        logOHsol = 8.77 #log(O/H)+12 value used for Solar metallicity in MAPPINGS-V, Dopita 2016
        log_ratio = np.log10(np.divide(mapn2,np.add(maps2a,maps2b))) + 0.264*np.log10(np.divide(mapn2,mapha))
        logOHobj_map = log_ratio + 0.45*(log_ratio + 0.3)**5 # + 8.77
        myprint('log_ratio med, min '+str(np.median(log_ratio))+' '+str(np.min(log_ratio))+'\n', args)
        myprint('logOHobj_map before conversion med, min '+str(np.median(logOHobj_map))+' '+str(np.min(logOHobj_map))+'\n', args)
        t += '_D16'    
    #---------------------------------------------------
    Z_map = 10**(logOHobj_map) #converting to Z (in units of Z_sol) from log(O/H) + 12
    #Z_list = Z_map.flatten()
    Z_list =logOHobj_map.flatten()
    myprint('Z_list after conversion med, mean, min '+str(np.median(Z_list))+' '+str(np.mean(Z_list))+' '+str(np.min(Z_list))+'\n', args)
    '''
    #--to print metallicity histogram: debugging purpose------
    plt.hist(np.log10(Z_list), 100, range =(-1, 50)) #
    plt.title('Z_map after conv') #
    plt.xlabel('log Z/Z_sol') #
    plt.yscale('log') #
    '''
    cbarlab = 'log(Z/Z_sol)'
    if args.getmap:
        map = plotmap(logOHobj_map, t, 'Metallicity', cbarlab, args, logbook, islog=False)
    else:
        if not args.hide:
            if args.inspect: fig = plt.figure(figsize=(14,10))
            else: fig = plt.figure(figsize=(8,6))
            fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.1, right=0.95)
            ax = plt.subplot(111)
            plt.scatter(d.flatten(),Z_list, s=10, lw=0, c='b', label='pixels')
            if args.inspect:
                plt.scatter(d.flatten(),(np.divide(mapn2,np.add(maps2a,maps2b))).flatten(), s=10, lw=0, c='b', marker='s', label='pixels NII/SII')
                plt.scatter(d.flatten(),(np.divide(mapn2,mapha)).flatten(), s=10, lw=0, c='b', marker='^', label='pixels NII/Ha')
            #plt.axhline(10**(8.6-logOHsol),c='k',lw=0.5, label='trust') #line below which metallicity values should not be trusted for Kewley 2002 diag
            plt.plot(np.arange(args.galsize/2), np.poly1d((args.logOHgrad, args.logOHcen))(np.arange(args.galsize/2)) - logOHsol,c='r', label='True gradient')
            plt.axhline(0,c='k',linestyle='--',label='Zsol') #line for solar metallicity
            plt.xlabel('Galactocentric distance (kpc)')
            plt.ylabel(cbarlab)
            plt.xlim(0,args.galsize/2)
            plt.legend()
            plt.title(t)
            if args.inspect: plt.ylim(-1,3)
            else: plt.ylim(-1,1)
            if args.saveplot:
                fig.savefig(path+t+'.png')
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
                    if args.fixed_SNR is not None: gradfile += '_fixedSNR'+str(args.fixed_SNR)
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
                if not args.hide: plt.plot(x_arr, np.poly1d(linefit)(x_arr), c='b',label='Inferred gradient')
            except (TypeError, IndexError, ValueError):
                myprint('No data points for vres= '+str(args.vres)+' above given SNR_thresh of '+str(args.SNR_thresh)+'\n', args)
                pass
            
#-------------Fucntion for fitting multiple lines----------------------------
def fit_all_lines(args, logbook, properties, flam, pix_i, pix_j, nres=5, z=0, z_err=0.0001) :
    wave, flam = np.array(properties.dispsol), np.array(flam) #converting to numpy arrays
    kk, count, flux_array, flux_error_array = 1, 0, [], []
    ndlambda_left, ndlambda_right = [nres]*2 #how many delta-lambda wide will the window (for line fitting) be on either side of the central wavelength, default 5
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
            if args.debug: myprint('Trying to fit '+str(logbook.llist[kk-count:kk])+' line/s at once. Total '+str(count)+'\n', args)
            try: 
                popt, pcov = fitline(wave_short, flam_short, logbook.wlist[kk-count:kk], logbook.resoln, z=z, z_err=z_err)
                if args.showplot:
                    plt.axvline(leftlim, linestyle='--',c='g')
                    plt.axvline(rightlim, linestyle='--',c='g')
                ndlambda_left, ndlambda_right = [nres]*2
                if args.debug: myprint('Done this fitting!'+'\n', args)
            except TypeError, er:
                if args.debugt: myprint('Trying to re-do this fit with broadened wavelength window..\n', args)
                ndlambda_left+=1
                ndlambda_right+=1
                continue
            except (RuntimeError, ValueError), e:
                popt = np.zeros(count*3 + 1) #if could not fit the line/s fill popt with zeros so flux_array gets zeros
                pcov = np.zeros((count*3 + 1,count*3 + 1)) #if could not fit the line/s fill popt with zeros so flux_array gets zeros
                myprint('Could not fit lines '+str(logbook.llist[kk-count:kk])+' for pixel '+str(pix_i)+', '+str(pix_j)+'\n', args)
                pass
                
            for xx in range(0,count):
                #in popt for every bunch of lines, element 0 is the continuum(a)
                #and elements (1,2,3) or (4,5,6) etc. are the height(b), mean(c) and width(d)
                #so, for each line the elements (0,1,2,3) or (0,4,5,6) etc. make the full suite of (a,b,c,d) gaussian parameters
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
                flux = np.sqrt(2*np.pi)*(popt_single[1] - popt_single[0])*popt_single[3] #total flux = integral of guassian fit ; resulting flux in ergs/s/pc^2 units
                flux_array.append(flux)
                flux_error = np.sqrt(2*np.pi*(popt_single[3]**2*(pcov[0][0] + pcov[3*xx+1][3*xx+1])\
                + (popt_single[1]-popt_single[0])**2*pcov[3*(xx+1)][3*(xx+1)]\
                - 2*popt_single[3]**2*pcov[3*xx+1][0]\
                + 2*(popt_single[1] - popt_single[0])*popt_single[3]*(pcov[3*xx+1][3*(xx+1)] - pcov[0][3*(xx+1)])\
                )) # var_f = 3^2(00 + 11) + (1-0)^2*33 - (2)*3^2*10 + (2)*3*(1-0)*(13-03)
                flux_error_array.append(flux_error)
                if args.showplot:
                    leftlim = popt_single[2]*(1.-nres/logbook.resoln) 
                    rightlim = popt_single[2]*(1.+nres/logbook.resoln)
                    wave_short_single = wave[(leftlim < wave) & (wave < rightlim)]
                    plt.plot(wave_short_single, np.log10(su.gaus(wave_short_single,1, *popt_single)),lw=1, c='r')
            if args.showplot:
                if count >1: plt.plot(wave_short, np.log10(su.gaus(wave_short, count, *popt)),lw=2, c='g')                   
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
    flux_array[flux_array<1.] = 0. #filtering out obvious non-detections and setting those fluxes to 0
    return flux_array, flux_error_array
#-------------------------------------------------------------------------------------------
def fitline(wave, flam, wtofit, resoln, z=0, z_err=0.0001):
    v_maxwidth = 10*c/resoln #10*vres in km/s
    z_allow = 3*z_err #wavelengths are at restframe; assumed error in redshift
    p_init, lbound, ubound = [np.abs(flam[0])],[0.],[np.inf]
    for xx in range(0, len(wtofit)):
        fl = np.max(flam) #flam[(np.abs(wave - wtofit[xx])).argmin()] #flam[np.where(wave <= wtofit[xx])[0][0]]
        p_init = np.append(p_init, [fl-flam[0], wtofit[xx], wtofit[xx]*2.*gf2s/resoln])
        lbound = np.append(lbound,[0., wtofit[xx]*(1.-z_allow/(1.+z)), wtofit[xx]*1.*gf2s/resoln])
        ubound = np.append(ubound,[np.inf, wtofit[xx]*(1.+z_allow/(1.+z)), wtofit[xx]*v_maxwidth*gf2s/c])
    popt, pcov = curve_fit(lambda x, *p: su.gaus(x, len(wtofit), *p),wave,flam,p0= p_init, max_nfev=10000, bounds = (lbound, ubound))
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
                    cw.append(float(line.split()[1]))
                    cf.append(10**float(line.split()[2]))
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
    if args.changeunits: plt.ylim(29-40,37-40)
    else: plt.ylim(30,37)
    plt.xlim(logbook.wmin,logbook.wmax)
    if not args.hide:
        plt.show(block=False)
    if args.saveplot:
        fig.savefig(path+t+'.png')
#-------------------------------------------------------------------------------------------
def plotintegmap(args, logbook, properties):
    ppv = properties.ppvcube[:,:,(properties.dispsol >= logbook.wmin) & (properties.dispsol <= logbook.wmax)]
    cbarlab = 'Log surface brightness in erg/s/pc^2' #label of color bar
    if args.changeunits: 
        cbarlab = cbarlab[:cbarlab.find(' in ')+4] + 'erg/s/cm^2/A' #label would be ergs/s/pc^2/A if we choose to change units to flambda
    line = 'lambda-integrated wmin='+str(logbook.wmin)+', wmax='+str(logbook.wmax)+'\n'
    map = np.sum(ppv,axis=2)
    t = title(args.file)+line+' map for Omega = '+str(args.Om)+', res = '+str(logbook.final_pix_size)+' kpc'
    dummy = plotmap(map, t, line, cbarlab, args, logbook)
    return dummy
#-------------------------------------------------------------------------------------------
def spec_total(w, ppv, title, args, logbook):
    cbarlab = 'Log surface brightness in erg/s/pc^2' #label of color bar
    fig = plt.figure(figsize=(14,6))
    fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.1, right=0.95)
    ax = plt.subplot(111)
    for i in logbook.wlist:
        plt.axvline(i,ymin=0.9,c='black')    

    #-------------------------------------------------------------------------------------------
    y = np.log10(np.sum(ppv,axis=(0,1)))
    plt.plot(w, y,lw=1)
    t = title+', for '+title(args.file)+' Nebular+ stellar for Om = '+str(args.Om)+', res = '+str(logbook.final_pix_size)+' kpc' + info
    #-------------------------------------------------------------------------------------------
    plt.title(t)
    plt.ylabel(cbarlab)
    plt.xlabel('Wavelength (A)')
    if args.changeunits: plt.ylim(29-40,37-40)
    else: plt.ylim(np.min(y)*0.9, np.max(y)*1.1)
    plt.xlim(logbook.wmin,logbook.wmax)
    plt.show(block=False)
    if args.saveplot:
        fig.savefig(args.path+t+'.png')
#-------------------------------------------------------------------------------------------
def get_disp_array(args, logbook, properties):
    sig = 5*args.vdel/c 
    w = np.linspace(logbook.wmin, logbook.wmax, args.nbin)
    for ii in logbook.wlist:
        w1 = ii*(1-sig)
        w2 = ii*(1+sig)
        highres = np.linspace(w1, w2, args.nhr)
        w = np.insert(w, np.where(np.array(w)<w2)[0][-1]+1, highres)
    properties.w = np.sort(w)
    #-------------------------------------------------------------------------------------------
    if args.spec_smear:        
        properties.new_w = [np.min(properties.w)]
        while properties.new_w[-1] < np.max(properties.w):
            properties.new_w.append(properties.new_w[-1]*(1+args.vres/c))
        properties.nwbin = len(properties.new_w) #final no. of bins in wavelength dimension
        properties.bin_index = np.digitize(properties.w, properties.new_w)
    else:
        properties.nwbin = len(properties.w) + 1
        properties.new_w = properties.w
        properties.bin_index = -999 #dummy, not required if spec_smear is turned OFF
    
    properties.dispsol = np.array(properties.new_w[1:]) if args.spec_smear else np.array(properties.w)
    return properties
#-------------------------------------------------------------------------------------------
def spec(args, logbook, properties):
    global info
    properties = get_disp_array(args, logbook, properties)
    if args.changeunits: 
        cbarlab = cbarlab[:cbarlab.find(' in ')+4] + 'erg/s/cm^2/A' #label would be ergs/s/pc^2/A if we choose to change units to flambda
    if os.path.exists(logbook.H2R_filename):
        ppv = fits.open(logbook.H2R_filename)[0].data
        myprint('Reading existing H2R file cube from '+logbook.H2R_filename+'\n', args)
    else:
        #-------------------------------------------------------------------------------------------
        g,x,y = calcpos(logbook.s, args.galsize, args.res)
        ppv = np.zeros((g,g,properties.nwbin - 1))
        funcar = readSB(logbook.wmin, logbook.wmax)
        #-------------------------------------------------------------------------------------------
    
        for j in range(len(logbook.s)):
            myprint('Particle '+str(j+1)+' of '+str(len(s))+'\n', args)
            vz = float(logbook.s['vz'][j])
            a = int(round(logbook.s['age(MYr)'][j]))
            f = np.multiply(funcar[a](properties.w),(300./1e6)) ### to scale the continuum by 300Msun, as the ones produced by SB99 was for 1M Msun
            flist=[]
            for l in logbook.llist:
                try: flist.append(logbook.s[l][j])
                except: continue
            flist = np.multiply(np.array(flist), const) #to multiply with nebular flux to make it comparable with SB continuum
        
            for i, fli in enumerate(flist):
                f = gauss(properties.w, f, logbook.wlist[i], fli, args.vdisp, vz) #adding every line flux on top of continuum
            if args.changeunits:
                f /= (properties.w*3.086e18**2) #changing units for David Fisher: from ergs/s to ergs/s/A; the extra factor is to make it end up as /cm^2 insted of /pc^2
            if args.spec_smear: 
                f = [f[properties.bin_index == ii].sum() for ii in range(1, len(properties.new_w))]
            ppv[int(x[j]/args.res)][int(y[j]/args.res)][:] += f #ergs/s
        myprint('Done reading in all HII regions in '+str((time.time() - start_time)/60)+' minutes.\n', args)
        write_fits(H2R_filename, ppv, fill_val=np.nan, outputfile=outputfile)
        if args.debug:
            spec_total(properties.dispsol, ppv, 'Spectrum for only H2R after spec smear', args, logbook)
            mydiag('Deb701: for H2R cube',ppv, args)
    #-------------------------Now ideal PPV is ready: do whatever with it------------------------------------------------------------------
    
    if args.smooth and not args.maketheory:
        if os.path.exists(logbook.skynoise_cubename):
            properties.skynoise = fits.open(logbook.skynoise_cubename)[0].data
            myprint('Reading existing skynoise cube from '+logbook.skynoise_cubename+'\n', args)
        elif args.addnoise:
            myprint('Computing skynoise cube..\n', args)
            properties.skynoise = getskynoise(properties.dispsol, logbook.final_pix_size)            
            write_fits(logbook.skynoise_cubename, properties.skynoise, args, fill_val=np.nan)
        else: properties.skynoise = None
        if os.path.exists(logbook.convolved_filename): #read it in if the convolved cube already exists
            myprint('Reading existing convolved cube from '+logbook.convolved_filename+'\n', args)
        else: #make the convolved cube and save it if it doesn't exist already
            ppv_rebinned = np.zeros((int(args.galsize/logbook.final_pix_size), int(args.galsize/logbook.final_pix_size), np.shape(ppv)[2]))
            for ind in range(0,np.shape(ppv)[2]):
                ppv_rebinned[:,:,ind] = rebin(ppv[:,:,ind], args.res, logbook.new_res) #re-bin 2d array before convolving to make things faster(previously each pixel was of size res)    
            if args.debug: mydiag('Deb719: for just rebinned PPV cube', ppv_rebinned, args)
            myprint('Using '+args.ker+' kernel.\nUsing parameter set: sigma= '+str(logbook.sig)+', size= '+str(logbook.size,)+'\n', args)
            binned_cubename = args.path + 'temp_binned_cube.fits'
            write_fits(binned_cubename, ppv_rebinned, args, fill_val=np.nan)
            
            funcname = HOME+'/models/enzo_model_code/parallel_convolve.py'
            if args.silent: silent = ' --silent'
            else: silent = ''
            if args.toscreen: toscreen = ' --toscreen'
            else: toscreen = ''
                    
            command = 'mpirun -np '+str(args.ncores)+' python '+funcname+' --parallel --sig '+str(logbook.sig)+\
            ' --pow '+str(args.pow)+' --size '+str(logbook.size)+' --ker '+args.ker+' --convolved_filename '+\
            logbook.convolved_filename+' --outfile '+args.outfile+' --binned_cubename '+binned_cubename + silent + toscreen
            subprocess.call([command],shell=True)
            subprocess.call(['rm -r '+binned_cubename],shell=True)
            
        convolved_cube = fits.open(logbook.convolved_filename)[0].data #reading in convolved cube from file
        if args.debug: mydiag('Deb737: for convolved cube', convolved_cube, args)
        ppv = makeobservable(convolved_cube, args, logbook, properties) #add noise, clip saturated pixels, etc.
        if args.debug: mydiag('Deb739: for final PPV cube', ppv, args)
        
    myprint('Final pixel size on target frame = '+str(logbook.final_pix_size)+' kpc'+' and shape = ('+str(np.shape(ppv)[0])+','+str(np.shape(ppv)[1])+','+str(np.shape(ppv)[2])+') \n', args)
    #-------------------------Now realistic (smoothed, noisy) PPV is ready------------------------------------------------------------------
    if not args.hide: spec_total(properties.dispsol, ppv, 'Spectrum for total', args, logbook)
    myprint('Returning PPV as variable "ppvcube"'+'\n', args)
    return np.array(ppv)                
#-------------------------------------------------------------------------------------------
def makeobservable(cube, args, logbook, properties):
    nslice = np.shape(cube)[2]
    new_cube = np.zeros(np.shape(cube))
    for k in range(nslice):
        myprint('Making observable slice '+str(k+1)+' of '+str(nslice)+'\n', args)
        map = cube[:,:,k]
        skynoiseslice = properties.skynoise[k] if properties.skynoise is not None else None
        if args.debug: mydiag('Deb 754: before factor', map, args)
        map *= properties.factor #to get in counts from ergs/s
        if args.debug:
            mydiag('Deb 756: in counts: after multiplying by factor', map, args)
        if args.addnoise: map = makenoisy(map, args, logbook, properties, skynoiseslice=skynoiseslice, factor=args.gain) #factor=gain as it is already in counts (ADU), need to convert to electrons for Poisson statistics
        map = np.ma.masked_where(np.log10(map)<0., map) #clip all that have less than 1 count
        map = np.ma.masked_where(np.log10(map)>5., map) #clip all that have more than 100,000 count i.e. saturating
        map /= properties.factor #convert back to ergs/s from counts
        if args.debug: mydiag('Deb 762: in ergs/s: after dividing by factor', map, args)
        map /= (logbook.final_pix_size*1e3)**2 #convert to ergs/s/pc^2 from ergs/s
        if args.debug: mydiag('Deb 762: in ergs/s/pc^2: after dividing by pixel area', map, args)
        new_cube[:,:,k] = map
    return new_cube
#-------------------------------------------------------------------------------------------
def makenoisy(data, args, logbook, properties, skynoiseslice=None, factor=None):
    dummy = copy.copy(data)
    if args.debug:
        dummy = plotmap(dummy, 'before adding any noise', 'junk', 'counts', args, logbook, islog=False)
        mydiag('Deb 767: before adding any noise', dummy, args)
    size = args.galsize/np.shape(data)[0]
    if factor is None: factor = properties.factor
    data *= factor #to transform into counts (electrons) from physical units
    if args.debug: mydiag('Deb 771: after mutiplying gain factor', data, args)
    if args.fixed_SNR is not None: #adding only fixed amount of SNR to ALL spaxels
        noisydata = data + np.random.normal(loc=0., scale=np.abs(data/args.fixed_SNR), size=np.shape(data)) #drawing from normal distribution about a mean value of 0 and width =counts/SNR
        if args.debug:
            dummy = plotmap(noisydata, 'after fixed_SNR '+str(args.fixed_SNR)+' noise', 'junk', 'counts', args, logbook, islog=False)
            mydiag('Deb 775: after fixed_SNR '+str(args.fixed_SNR)+' noise', noisydata, args)
    else:
        noisydata = np.random.poisson(lam=data, size=None) #adding poisson noise to counts (electrons)
        noisydata = noisydata.astype(float)
        if args.debug:
            dummy = plotmap(noisydata, 'after poisson', 'junk', 'counts', args, logbook, islog=False)
            mydiag('Deb 783: after adding poisson noise', noisydata, args)
        readnoise = np.sqrt(2*7.) * np.random.normal(loc=0., scale=3.5, size=np.shape(noisydata)) #to draw gaussian random variables from distribution N(0,3.5) where 3.5 is width in electrons per pixel
                                    #sqrt(14) is to account for the fact that for SAMI each spectral fibre is 2 pix wide and there are 7 CCD frames for each obsv
        if args.debug: mydiag('Deb 781: only RDNoise', readnoise, args)
        noisydata += readnoise #adding readnoise
        if args.debug:
            dummy = plotmap(noisydata, 'after readnoise', 'junk', 'counts', args, logbook, islog=False)
            mydiag('Deb 783: after adding readnoise', noisydata, args)
        if skynoiseslice is not None and skynoiseslice != 0: 
            skynoise = np.random.normal(loc=0., scale=np.abs(skynoiseslice), size=np.shape(noisydata)) #drawing from normal distribution about a sky noise value at that particular wavelength
            noisydata /= logbook.exptime #converting to electrons/s just to add skynoise, bcz skynoise is also in el/s units
            noisydata += skynoise #adding sky noise
            noisydata *= logbook.exptime #converting back to electrons units
        if args.debug:
            dummy = plotmap(noisydata, 'after skynoise', 'junk', 'counts', args, logbook, islog=False)
            mydiag('Deb 795: after adding skynoise', noisydata, args)

    noisydata /= factor #converting back to physical units from counts (electrons)
    if args.debug: mydiag('Deb 803: after dividing gain factor', noisydata, args)
    if args.debug:
        noise = noisydata - dummy
        myprint('Net effect of all noise:\n'+\
        'makenoisy: array median std min max'+'\n'+\
        'makenoisy: data '+str(np.median(dummy))+' '+str(np.std(dummy))+' '+str(np.min(masked_data(dummy)))+' '+str(np.max(dummy))+'\n'+\
        'makenoisy: noisydata '+str(np.median(noisydata))+' '+str(np.std(noisydata))+' '+str(np.min(masked_data(noisydata)))+' '+str(np.max(noisydata))+'\n'+\
        'makenoisy: noise '+str(np.median(noise))+' '+str(np.std(noise))+' '+str(np.min(masked_data(noise)))+' '+str(np.max(noise))+'\n'\
        , args)
    return noisydata
#-------------------------------------------------------------------------------------------
def inspectmap(args, logbook, properties):
    g,x,y = calcpos(logbook.s, args.galsize, args.res)
    
    g2=np.shape(properties.ppvcube)[0]
    if args.plotmet:
        log_ratio = np.log10(np.divide(logbook.s['NII6584'],(logbook.s['SII6730']+logbook.s['SII6717']))) + 0.264*np.log10(np.divide(logbook.s['NII6584'],logbook.s['H6562']))
        logOHobj = log_ratio + 0.45*(log_ratio + 0.3)**5
        
        myprint('all HIIR n2, s2, ha medians '+str(np.median(logbook.s['NII6584']))+','+str(np.median(logbook.s['SII6730']))+','+str(np.median(logbook.s['H6562']))+'\n'+\
        'all HIIR n2, s2, ha integrated '+str(np.sum(logbook.s['NII6584']))+','+str(np.sum(logbook.s['SII6730']))+','+str(np.sum(logbook.s['H6562']))+'\n'+\
        'all HIIR Z/Zsol median '+str(np.median(10**logOHobj))+'\n', \
        outfile=outputfile, toscreen=args.toscreen)

        d = np.sqrt((x-args.galsize/2)**2 + (y-args.galsize/2)**2)
        plt.scatter(d,logOHobj,c='r',s=5,lw=0,label='indiv HII reg')
        plt.scatter(d,np.divide(logbook.s['NII6584'],(logbook.s['SII6730']+logbook.s['SII6717'])),c='r',s=5,lw=0,marker='s',label='indiv HII reg NII/SII')
        plt.scatter(d,np.divide(logbook.s['NII6584'],logbook.s['H6562']),c='r',s=5,lw=0,marker='^',label='indiv HII reg NII/Ha')
        '''
        #small check for DIG vs HIIR
        plt.scatter(d,logbook.s['SII6717']/logbook.s['H6562'],c='k',s=5,lw=0,label='indiv HII reg-SII/Ha') #
        plt.axhline(0.35, c='cyan',label='max allowed for HIIR') #
        print 'DIG analysis: SII6717/Halpha ratio for indiv HII regions: min, max, median',\
        np.min(logbook.s['SII6717']/logbook.s['H6562']), np.max(logbook.s['SII6717']/logbook.s['H6562']), np.median(logbook.s['SII6717']/logbook.s['H6562']) #
        '''
        tempn2 = make2Dmap(logbook.s['NII6584'], x, y, g2, args.galsize/g2)
        temps2a = make2Dmap(logbook.s['SII6717'], x, y, g2, args.galsize/g2)
        temps2b = make2Dmap(logbook.s['SII6730'], x, y, g2, args.galsize/g2)
        tempha = make2Dmap(logbook.s['H6562'], x, y, g2, args.galsize/g2)
        
        log_ratio = np.log10(np.divide(tempn2,(temps2a+temps2b))) + 0.264*np.log10(np.divide(tempn2,tempha))
        logOHobj = log_ratio + 0.45*(log_ratio + 0.3)**5
        
        myprint('summed up HIIR n2, s2a, s2b, ha medians '+str(np.median(tempn2))+','+str(np.median(temps2a))+','+str(np.median(temps2b))+','+str(np.median(tempha))+'\n'+\
        'summed up HIIR n2, s2a, s2b, ha integrated '+str(np.sum(tempn2))+','+str(np.sum(temps2a))+','+str(np.sum(temps2b))+','+str(np.sum(tempha))+'\n'+\
        'all HIIR Z/Zsol median '+str(np.median(10**logOHobj))+'\n',\
        outfile=outputfile, toscreen=args.toscreen)
        
        b = np.linspace(-g2/2 + 1,g2/2,g2)*(args.galsize)/g2 #in kpc
        d = np.sqrt(b[:,None]**2+b**2)
        plt.scatter(d.flatten(),logOHobj.flatten(),c='g',s=5,lw=0,label='summed up HII reg')
        plt.scatter(d.flatten(),(np.divide(tempn2,(temps2a+temps2b))).flatten(),c='g',s=5,lw=0,marker='s',label='summed up HII reg NII/SII')
        plt.scatter(d.flatten(),(np.divide(tempn2,tempha)).flatten(),c='g',s=5,lw=0,marker='^',label='summed up HII reg NII/Ha')
        plt.legend()
        #plt.show(block=False)
        
        map = plotmap(tempn2, 'NII6584'+': H2R summed up', 'trial', 'log flux(ergs/s)', args, logbook, islog=True)
        map = plotmap(temps2a, 'SII6717'+': H2R summed up', 'trial', 'log flux(ergs/s)', args, logbook, islog=True)
        map = plotmap(temps2b, 'SII6730'+': H2R summed up', 'trial', 'log flux(ergs/s)', args, logbook, islog=True)
        map = plotmap(tempha, 'H6562'+': H2R summed up', 'trial', 'log flux(ergs/s)', args, logbook, islog=True)
        
    else:        
        fig = plt.figure(figsize=(8,8))
        fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.1, left=0.1, right=0.9)
        ax = plt.subplot(111)
        pl=ax.scatter(x-15,y-15,c=np.log10(logbook.s[line]), lw=0,s=3,vmin=args.cmin,vmax=args.cmax) #x,y in kpc
        plt.title(line+': indiv H2R') 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(pl, cax=cax).set_label('log flux(ergs/s)')    
    
        temp = np.zeros((g2,g2))
        for j in range(len(s)):
            temp[int(x[j]*g2/galsize)][int(y[j]*g2/galsize)] += s[line][j]
        map = plotmap(temp, line+': H2R summed up', 'trial', 'log flux(ergs/s)', galsize, galsize/g2, cmin=cmin, cmax=cmax, hide = False, saveplot=saveplot, islog=True)
           
    if hasattr(properties, 'mapcube'):
        mapn2 = properties.mapcube[:,:,np.where(logbook.llist == 'NII6584')[0][0]]
        maps2a = properties.mapcube[:,:,np.where(logbook.llist == 'SII6717')[0][0]]
        maps2b = properties.mapcube[:,:,np.where(logbook.llist == 'SII6730')[0][0]]
        mapha = properties.mapcube[:,:,np.where(logbook.llist == 'H6562')[0][0]]
        mapn2 = np.ma.masked_where(mapn2<=0., mapn2)
        maps2a = np.ma.masked_where(maps2a<=0., maps2a)
        maps2b = np.ma.masked_where(maps2b<=0., maps2b)
        mapha = np.ma.masked_where(mapha<=0., mapha)

        if args.SNR_thresh is not None:
            #loading all the flux uncertainties
            mapn2_u = properties.errorcube[:,:,np.where(logbook.llist == 'NII6584')[0][0]]
            maps2a_u = properties.errorcube[:,:,np.where(logbook.llist == 'SII6717')[0][0]]
            maps2b_u = properties.errorcube[:,:,np.where(logbook.llist == 'SII6730')[0][0]]
            mapha_u = properties.errorcube[:,:,np.where(logbook.llist == 'H6562')[0][0]]
            mapn2_u = np.ma.masked_where(mapn2_u<=0., mapn2_u)
            maps2a_u = np.ma.masked_where(maps2a_u<=0., maps2a_u)
            maps2b_u = np.ma.masked_where(maps2b_u<=0., maps2b_u)
            mapha_u = np.ma.masked_where(mapha_u<=0., mapha_u)
        
            #imposing SNR cut
            mapn2 = np.ma.masked_where(mapn2/mapn2_u <args.SNR_thresh, mapn2)
            maps2a = np.ma.masked_where(maps2a/maps2a_u <args.SNR_thresh, maps2a)
            maps2b = np.ma.masked_where(maps2b/maps2b_u <args.SNR_thresh, maps2b)
            mapha = np.ma.masked_where(mapha/mapha_u <args.SNR_thresh, mapha)

        g = np.shape(properties.mapcube)[0]

        myprint('mapn2, s2a, s2b, ha max '+str(np.max(mapn2))+','+str(np.max(maps2a))+','+str(np.max(maps2b))+','+str(np.max(mapha))+'\n'+\
        'mapn2, s2a, s2b, ha min '+str(np.min(mapn2))+','+str(np.min(maps2a))+','+str(np.min(maps2b))+','+str(np.min(mapha))+'\n'+\
        'mapn2, s2a, s2b, ha integrated '+str(np.sum(mapn2*(galsize*1000./g)**2))+','+str(np.sum(maps2a*(galsize*1000./g)**2))+','+str(np.sum(maps2b*(galsize*1000./g)**2))+','+str(np.sum(mapha*(galsize*1000./g)**2))+','+'ergs/s'+'\n'+\
        '#cells= '+str(g)+' each cell= '+str(galsize*1000./g)+' '+'pc'+'\n',\
        outfile=outputfile, toscreen=args.toscreen)
            
        if plotmet:
            map = plotmap(mapn2, 'NII6584'+' map after fitting', 'Metallicity', 'log flux(ergs/s/pc^2)', args, logbook, islog=True)
            map = plotmap(maps2a, 'SII6717'+' map after fitting', 'Metallicity', 'log flux(ergs/s/pc^2)', args, logbook, islog=True)
            map = plotmap(maps2b, 'SII6730'+' map after fitting', 'Metallicity', 'log flux(ergs/s/pc^2)', args, logbook, islog=True)
            map = plotmap(mapha, 'H6562'+' map after fitting', 'Metallicity', 'log flux(ergs/s/pc^2)', args, logbook, islog=True)
        else:
            map = plotmap(properties.mapcube[:,:,np.where(logbook.llist == args.line)[0][0]]*(args.galsize*1000./g)**2, args.line+' map after fitting', 'Metallicity', 'log integ flux(ergs/s)', args, logbook, islog=True)
    
    return
#-------------------------------------------------------------------------------------------
def calcpos(s, galsize, res):
    g = int(np.ceil(galsize/res))
    x = (s['x']-1500/2)*res + galsize/2
    y = (s['y']-1500/2)*res + galsize/2
    return g, x, y
#-------------------------------------------------------------------------------------------
def make2Dmap(data, xi, yi, ngrid, res, domean=False, islog=False):
    map = np.zeros((ngrid,ngrid))
    if domean:
        count = np.zeros((ngrid,ngrid))
    for i in range(len(data)):
        x = int(xi[i]/res)
        y = int(yi[i]/res)
        if islog:
            map[x][y] += 10.**data[i]
        else:
            map[x][y] += data[i] 
        if domean:
            count[x][y] += 1
    if domean:
        map = np.divide(map,count)
        map[np.isnan(map)] = 0
    return map 
#-------------------------------------------------------------------------------------------
def getskynoise(wave, final_pix_size):
    #to read in sky noise files in physical units, convert to el/s, spectral-bin them and create map
    bluenoise = fits.open(HOME+'/models/Noise_model/NoiseData-99259_B.fits')[1].data[0]
    rednoise  = fits.open(HOME+'/models/Noise_model/NoiseData-99259_R.fits')[1].data[0]
    skywave = np.hstack((bluenoise[0], rednoise[0])) #in Angstrom
    noise = np.hstack((bluenoise[1], rednoise[1])) #in 10^-16 ergs/s/cm^2/A/spaxel    
    noise[noise > 100.] = 0. #replacing insanely high noise values
    noise = 1e-16 * (3.086e18)**2 * np.multiply(noise, skywave)# to convert it to ergs/s/pc^2, as the flux values are
    factor = (final_pix_size*1e3)**2 * flux_ratio * el_per_phot / (planck * nu) # do we need to multiply with flux_ratio?? Think!
    noise *= factor #to transform into counts/s (electrons/s) from physical units, because we do not know the exposure time for the skynoise provided by Rob Sharp
    f = interp1d(skywave, noise, kind='cubic')
    wave = np.array(wave)
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
        cmin = np.min(map) if args.cmin is None else args.cmin
        cmax = np.max(map) if args.cmax is None else args.cmax  
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
    dummy = np.zeros((1300,1300)) #dummy array to check what actual resolution we are left with after rebinning
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
        args, logbook = getsmoothparm(args, properties, logbook)
        info += '_smeared_'+args.ker+'_parm'+str(logbook.fwhm)+','+str(logbook.sig)+','+str(args.pow)+','+str(logbook.size)
    else: logbook.final_pix_size = args.res
    if args.changeunits: info += '_flambda'
    info2 = info
    if args.addnoise: info += '_noisy'
    if args.fixed_SNR is not None: info += '_fixedSNR'+str(args.fixed_SNR)
    if not args.maketheory: info+= '_obs'
    if args.exptime is not None:
        logbook.exptime = float(args.exptime)
    else:
        if args.scale_exptime: scalefactor = float(args.scale_exptime)
        else: scalefactor = 240000. #sec
        logbook.exptime = float(scalefactor)*(args.res/logbook.final_pix_size)**2 #increasing exposure time quadratically with finer resolution, with fiducial values of 600s for 0.5"

    info += '_exp'+str(logbook.exptime)+'s'
    if args.multi_realisation: info += '_real'+str(args.multi_realisation)
    
    logbook.H2R_filename = args.path + 'H2R_'+args.file+'Om='+str(args.Om)+'_'+str(logbook.wmin)+'-'+str(logbook.wmax)+'A' + info1+ args.gradtext+'.fits'
    logbook.skynoise_cubename = args.path + 'skycube_'+'pixsize_'+str(logbook.final_pix_size)+'_'+str(logbook.wmin)+'-'+str(logbook.wmax)+'A'+info1+'.fits'
    logbook.convolved_filename = args.path + 'convolved_'+args.file+'Om='+str(args.Om)+',arc='+str(args.res_arcsec)+'_'+str(logbook.wmin)+'-'+str(logbook.wmax)+'A' + info2+ args.gradtext +'.fits'
    logbook.fitsname = args.path + 'PPV_'+args.file+'Om='+str(args.Om)+',arc='+str(args.res_arcsec)+'_'+str(logbook.wmin)+'-'+str(logbook.wmax)+'A' + info+ args.gradtext +'.fits'
    
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
#-------------------------------------------------------------------------------------------
def rebin_old(map, shape):
    sh = shape[0],map.shape[0]/shape[0],shape[1],map.shape[1]/shape[1]
    return map.reshape(sh).sum(-1).sum(1)
#-------------------------------------------------------------------------------------------
def calc_dist(z, H0 = 70.):
    dist = z*c*1e3/H0 #kpc
    return dist
#-------------------------------------------------------------------------------------------
def factors(n):    
    return list(reduce(list.__add__,([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
#-------------------------------------------------------------------------------------------
def masked_data(data):
    return np.ma.masked_where(data<=0, data)
#-------------------------------------------------------------------------------------------
def mydiag(title, data, args):
    myprint(title+': Median, stdev, max, min= '+str(np.median(masked_data(data)))+','+str(np.std(masked_data(data)))+','+\
    str(np.max(masked_data(data)))+','+str(np.min(masked_data(data)))+'\n', args)
#-------------------------------------------------------------------------------------------
def myprint(text, args):
    if args.toscreen: print text
    else:
        ofile = open(args.outfile,'a')
        ofile.write(text)
        ofile.close()
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
    parser.add_argument('--changeunits', dest='changeunits', action='store_true')
    parser.set_defaults(changeunits=False)
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

    parser.add_argument('--scale_exptime')
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
    args, leftovers = parser.parse_known_args()
    if args.debug: #debug mode over-rides 
        args.toscreen = True
        args.silent = False
        args.hide = False
        args.inspect = True

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
        args.el_per_phot = 1.

    if args.z is not None:
        args.z = float(args.z)
    else:
        args.z = 0.013

    if args.rad is not None:
        args.rad = float(args.rad)
    else:
        args.rad = 1. #metre

    properties.dist = calc_dist(args.z) #distance to object; in kpc
    properties.flux_ratio = (args.rad/(2*properties.dist*3.086e19))**2 #converting emitting luminosity to luminosity seen from earth, 3.08e19 factor to convert kpc to m

    if args.res is not None:
        args.res = float(args.res)
    else:
        args.res = 0.02 #kpc: simulation actual resolution 
    
    if args.arc is not None:
        args.res_arcsec = float(args.arc)
    else:
        args.res_arcsec = 0.5 #arcsec
    
    properties.res_phys = args.res_arcsec*np.pi/(3600*180)*properties.dist #kpc

    if args.line is None:
        args.line = 'OIII5007'# #whose emission map to be made

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
        args.nbin = 1000 #no. of bins used to bin the continuum into (without lines)

    if args.vdisp is not None:
        args.vdisp = float(args.vdisp)
    else:
        args.vdisp = 15 #km/s vel dispersion to be added to emission lines from MAPPINGS while making PPV

    if args.vdel is not None:
        args.vdel = float(args.vdel)
    else:
        args.vdel = 100 #km/s; vel range in which spectral resolution is higher is sig = 5*vdel/c
                    #so wavelength range of +/- sig around central wavelength of line is binned into further nhr bins

    if args.vres is not None:
        args.vres = float(args.vres)
    else:
        args.vres = 30 #km/s instrumental vel resolution to be considered while making PPV

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

    #-------------------------------------------------------------------------------------------
    args, logbook = getfitsname(args, properties) # name of fits file to be written into
    #-------------------------------------------------------------------------------------------
    properties.factor = properties.flux_ratio * logbook.exptime * args.el_per_phot / (args.gain * planck * nu) #to bring it to units of photons, or rather, ADUs
    if args.changeunits: properties.factor *= 3.086e18**2 * c*1e3 / nu * 1e10 #in case the units are in ergs/s/cm^2/A instead of ergs/s
    if args.debug:
        myprint('Deb1345: Factor = flux ratio= %.4E * exptime= %.4E * el_per_phot= %d / (gain= %.2F * planck= %.4E * nu= %.4E)'%(properties.factor, logbook.exptime,\
        args.el_per_phot, args.gain, planck, nu), args)
        myprint('Deb1346: Factor= %.4E'%properties.factor, args)
    #-------------------------------------------------------------------------------------------
    if args.toscreen: print 'deb1322: res_phys, final pix per beam, final pix size, final shape=', properties.res_phys, logbook.fwhm, logbook.final_pix_size, args.galsize/logbook.final_pix_size, 'kpc' #
    if args.outfile is None:
        args.outfile = args.path + 'output_'+logbook.fitsname[:-5]+'.txt' # name of fits file to be written into
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
    
    if not args.silent: myprint(starting_text, args)
    #------------------------------------------------------
    #-----------------------jobs fetched--------------------------------------------------------------------
    logbook.fittedcube = logbook.fitsname.replace('PPV','fitted-map-cube') # name of mapcube file to be read in       
    logbook.s = ascii.read(getfn(args), comment='#', guess=False)
    if args.get_scale_length: properties.scale_length = get_scale_length(args, logbook)
    elif args.ppv: properties.ppvcube = spec(args, logbook, properties)       
    else:
        if not os.path.exists(logbook.fitsname):
            if not args.silent: myprint('PPV file does not exist. Creating ppvcube..'+'\n', args)
            properties.ppvcube = spec(args, logbook, properties)       
            write_fits(logbook.fitsname, properties.ppvcube, args, fill_val=np.nan)        
        else:
            if not args.silent: myprint('Reading existing ppvcube from '+logbook.fitsname+'\n', args)
            properties.ppvcube = fits.open(logbook.fitsname)[0].data
        properties = get_disp_array(args, logbook, properties)
    
        if args.plotintegmap or args.plotspec:
            if args.X is not None:
                args.X = float(args.X)
            else:
                args.X = np.shape(properties.ppvcube)[0]/2 #p-p values at which point to extract spectrum from the ppv cube
            if args.plotspec and not args.silent: myprint('X position at which spectrum to be plotted= '+str(X)+'\n', args)

            if args.Y is not None:
                args.Y = float(args.Y)
            else:
                args.Y = np.shape(properties.ppvcube)[0]/2 #p-p values at which point to extract spectrum from the ppv cube
            if args.plotspec and not args.silent: myprint('Y position at which spectrum to be plotted= '+str(Y)+'\n', args)
            
            if args.plotintegmap:
                dummy = plotintegmap(args, logbook, properties)
            elif args.plotspec:
                dummy = spec_at_point(args, logbook, properties)
    
        else:
            logbook.fittederror = logbook.fittedcube.replace('map','error')
            if os.path.exists(logbook.fittedcube) and not args.clobber:
                if not args.silent: myprint('Reading existing mapcube from '+logbook.fittedcube+'\n', args)
            else:
                if not args.silent: myprint('Mapfile does not exist. Creating mapcube..'+'\n', args)
                
                if args.spec_smear: smear = ' --spec_smear '
                else: smear = ''
                if args.silent: silent = ' --silent'
                else: silent = ''
                if args.toscreen: toscreen = ' --toscreen'
                else: toscreen = ''
                if args.debug: toscreen = ' --debug'
                else: debug = ''
                if args.showplot: showplot = ' --showplot'
                else: showplot = ''
                
                funcname = HOME+'/models/enzo_model_code/parallel_fitting.py'
                command = 'mpirun -np '+str(args.ncores)+' python '+funcname+' --parallel --fitsname '+logbook.fitsname+' --nbin '+str(args.nbin)+\
                ' --vdel '+str(args.vdel)+' --vdisp '+str(args.vdisp)+' --vres '+str(args.vres)+' --nhr '+str(args.nhr)+' --wmin '+\
                str(logbook.wmin)+' --wmax '+str(logbook.wmax)+' --fittedcube '+logbook.fittedcube+' --fittederror '+logbook.fittederror+\
                ' --outfile '+args.outfile + smear + silent + toscreen + debug + showplot
                subprocess.call([command],shell=True)
                
            properties.mapcube = fits.open(logbook.fittedcube)[0].data
            if args.SNR_thresh is not None: properties.errorcube = fits.open(logbook.fittederror)[0].data
            else: properties.errorcube = None
            
            if args.bptpix: 
                bpt_pixelwise(args, logbook, properties)
            elif args.met: 
                properties = metallicity(args, logbook, properties)
            elif args.map: 
                properties.map = emissionmap(args, logbook, properties)
            elif args.sfr: 
                properties.SFRmap_real, properties.SFRmapHa = SFRmaps(args, logbook, properties)
            else: 
                if not args.silent: myprint('Wrong choice. Choose from:\n --bptpix, --map, --sfr, --met, --ppv, --plotinteg, --plotspec'+'\n', args)
        
            if args.inspect:
                inspectmap(args, logbook, properties)
    #-------------------------------------------------------------------------------------------
    if args.hide: plt.close()
    else: plt.show(block=False)
    if not args.silent:
        if args.saveplot: myprint('Saved plot here: '+path+'\n', args)
        myprint('Completed in %s minutes\n' % ((time.time() - start_time)/60), args)
