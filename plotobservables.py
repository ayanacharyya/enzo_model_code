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
def bpt_pixelwise(mapcube, wlist, llist, saveplot=False):
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


    mapn2 = np.divide(mapn2,mapha)
    mapo3 = np.divide(mapo3,maphb)
    t = title(fn)+' Galactrocentric-distance color-coded, BPT of model \n\
for Omega = '+str(Om)+', resolution = '+str(res)+' kpc'+info
    plt.scatter((np.log10(mapn2)).flatten(),(np.log10(mapo3)).flatten(), s=4, c=d.flatten(), lw=0,vmin=0,vmax=galsize/2)
    plt.title(t)
    cb = plt.colorbar()
    #cb.ax.set_yticklabels(str(res*float(x.get_text())) for x in cb.ax.get_yticklabels())
    cb.set_label('Galactocentric distance (in kpc)')
    if saveplot:
        fig.savefig(path+t+'.png')

#-------------------------------------------------------------------------------------------
def meshplot2D(x,y):
    for i in range(0,np.shape(x)[0]):
        plt.plot(x[i,:],y[i,:], c='red')
    for i in range(0,np.shape(x)[1]):
        plt.plot(x[:,i],y[:,i], c='blue')
    plt.title(title(fn)+' BPT of models')

#-------------------------------------------------------------------------------------------
def meshplot3D(x,y, annotate=False):
    age_arr = np.linspace(0.,5.,6) #in Myr
    lnII = np.linspace(5., 12., 6) #nII in particles/m^3
    lU = np.linspace(-4.,-1., 4) #dimensionless
    n = 3
    for k in range(0,np.shape(x)[2]):
        for i in range(0,np.shape(x)[0]):
            plt.plot(x[i,:,k],y[i,:,k], c='red', lw=0.5)
            if annotate and k==n: ax.annotate(str(age_arr[i])+' Myr', xy=(x[i,-1,k],y[i,-1,k]), \
            xytext=(x[i,-1,k]-0.4,y[i,-1,k]-0), color='red', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3",color='red'))
            #plt.pause(1) #
        for i in range(0,np.shape(x)[1]):
            plt.plot(x[:,i,k],y[:,i,k], c='blue', lw=0.5)
            if annotate and k==n: ax.annotate('log nII= '+str(lnII[i]), xy=(x[-2,i,k],y[-2,i,k]), \
            xytext=(x[-2,i,k]+0.,y[-2,i,k]-0.4),color='blue',fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3",color='blue'))
            #plt.pause(1) #
        if annotate: ax.annotate('log U= '+str(lU[k]), xy=(x[5,-1,k],y[5,-1,k]), \
            xytext=(x[5,-1,k]-0.6,y[5,-1,k]-1.),color='black',fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3",color='black'))
    
    for i in range(0,np.shape(x)[0]):
        for j in range(0, np.shape(x)[1]):
            plt.plot(x[i,j,:],y[i,j,:], c='black', lw=0.5)
            #plt.pause(1) #
    
    plt.title('MAPPINGS grid of models')

#-------------------------------------------------------------------------------------------
def gridoverlay(annotate=False, saveplot=False):
    s = ascii.read(HOME+'/Mappings/lab/totalspec.txt',comment='#',guess=False)
    y = np.reshape(np.log10(np.divide(s['OIII5007'],s['HBeta'])),(6, 6, 4))
    x = np.reshape(np.log10(np.divide(s['NII6584'],s['H6562'])),(6, 6, 4))
    x[[3,4],:,:]=x[[4,3],:,:] #for clearer connecting lines between grid points
    y[[3,4],:,:]=y[[4,3],:,:] #
    plt.scatter(x,y, c='black', lw=0, s=2)
    meshplot3D(x,y, annotate=annotate)
    if saveplot:
        fig.savefig(path+fn+':BPT overlay')
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

def get_scale_length(s, args=None, outputfile='junk.txt'):
    start = time.time()
    g,x,y = calcpos(s, galsize, res)
    
    d_list = np.sqrt((x-galsize/2)**2 + (y-galsize/2)**2) #kpc
    ha_list = [x for (y,x) in sorted(zip(d_list,s['H6562']), key=lambda pair: pair[0])]
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
    ha_map = make2Dmap(s['H6562']/res**2, x, y, g, res)
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
        plt.xlim(0,galsize/2)
        plt.legend(bbox_to_anchor=(0.9, 0.42), bbox_transform=plt.gcf().transFigure)
        plt.title('Measuring star formation scale length')
        plt.show(block=False)
    
    ofile = open(outputfile,'a')
    output = 'Scale length from Halpha map = '+str(scale_length)+' kpc, in %.2F min\n'%((time.time()-start)/60.)
    ofile.write(output)
    if args.toscreen: print output #
    return scale_length
#------------Function to measure metallicity-------------------------------------------------------------------------------
def metallicity(mapcube, wlist, llist, final_pix_per_beam, errorcube=None, SNR_thresh=None, getmap=False, hide=False, cmin=None, cmax=None, saveplot=False,calcgradient=False,nowrite=False,scale_exptime=False,fixed_SNR=None,args=None, outputfile='junk.txt'):
    global info, gradtext
    ofile = open(outputfile,'a')
    mapn2 = mapcube[:,:,np.where(llist == 'NII6584')[0][0]]
    maps2a = mapcube[:,:,np.where(llist == 'SII6717')[0][0]]
    maps2b = mapcube[:,:,np.where(llist == 'SII6730')[0][0]]
    mapha = mapcube[:,:,np.where(llist == 'H6562')[0][0]]
    #mapo2 = mapcube[:,:,np.where(llist == 'OII3727')[0][0]]
    mapn2 = np.ma.masked_where(mapn2<=0., mapn2)
    maps2a = np.ma.masked_where(maps2a<=0., maps2a)
    maps2b = np.ma.masked_where(maps2b<=0., maps2b)
    mapha = np.ma.masked_where(mapha<=0., mapha)
    #mapo2 = np.ma.masked_where(mapo2<=0., mapo2)

    if errorcube is not None:
        #loading all the flux uncertainties
        mapn2_u = errorcube[:,:,np.where(llist == 'NII6584')[0][0]]
        maps2a_u = errorcube[:,:,np.where(llist == 'SII6717')[0][0]]
        maps2b_u = errorcube[:,:,np.where(llist == 'SII6730')[0][0]]
        mapha_u = errorcube[:,:,np.where(llist == 'H6562')[0][0]]
        #mapo2_u = errorcube[:,:,np.where(llist == 'OII3727')[0][0]]
        mapn2_u = np.ma.masked_where(mapn2_u<=0., mapn2_u)
        maps2a_u = np.ma.masked_where(maps2a_u<=0., maps2a_u)
        maps2b_u = np.ma.masked_where(maps2b_u<=0., maps2b_u)
        mapha_u = np.ma.masked_where(mapha_u<=0., mapha_u)
        #mapo2_u = np.ma.masked_where(mapo2_u<=0., mapo2_u)
        
        if SNR_thresh is not None:
            #imposing SNR cut
            mapn2 = np.ma.masked_where(mapn2/mapn2_u <SNR_thresh, mapn2)
            maps2a = np.ma.masked_where(maps2a/maps2a_u <SNR_thresh, maps2a)
            maps2b = np.ma.masked_where(maps2b/maps2b_u <SNR_thresh, maps2b)
            mapha = np.ma.masked_where(mapha/mapha_u <SNR_thresh, mapha)
            #mapo2 = np.ma.masked_where(mapo2/mapo2_u <SNR_thresh, mapo2)
        
    g = np.shape(mapcube)[0]
    b = np.linspace(-g/2 + 1,g/2,g)*(galsize)/g #in kpc
    d = np.sqrt(b[:,None]**2+b**2)
    t = fn+':Met_Om'+str(Om)+'_arc'+str(res_arcsec)+'"'+'_vres='+str(vres)+'kmps_'+info+gradtext
    if SNR_thresh is not None: t += '_snr'+str(SNR_thresh)
    
    if args.useKD:
        #--------From Kewley 2002------------------
        logOHsol = 8.93 #log(O/H)+12 value used for Solar metallicity in earlier MAPPINGS in 2001
        log_ratio = np.log10(np.divide(mapn2,mapo2))
        ofile.write('log_ratio med, min '+str(np.median(log_ratio))+' '+str(np.min(log_ratio))+'\n') #
        logOHobj_map = np.log10(1.54020 + 1.26602*log_ratio + 0.167977*log_ratio**2) #+ 8.93
        ofile.write('logOHobj_map before conversion med, min '+str(np.median(logOHobj_map))+' '+str(np.min(logOHobj_map))+'\n') #
        t += '_KD02'
    else:
        #-------From Dopita 2016------------------------------
        logOHsol = 8.77 #log(O/H)+12 value used for Solar metallicity in MAPPINGS-V, Dopita 2016
        log_ratio = np.log10(np.divide(mapn2,np.add(maps2a,maps2b))) + 0.264*np.log10(np.divide(mapn2,mapha))
        ofile.write('log_ratio med, min '+str(np.median(log_ratio))+' '+str(np.min(log_ratio))+'\n') #
        logOHobj_map = log_ratio + 0.45*(log_ratio + 0.3)**5 # + 8.77
        ofile.write('logOHobj_map before conversion med, min '+str(np.median(logOHobj_map))+' '+str(np.min(logOHobj_map))+'\n') #
        t += '_D16'    
    #---------------------------------------------------
    Z_map = 10**(logOHobj_map) #converting to Z (in units of Z_sol) from log(O/H) + 12
    #Z_list = Z_map.flatten()
    Z_list =logOHobj_map.flatten()
    ofile.write('Z_list after conversion med, mean, min '+str(np.median(Z_list))+' '+str(np.mean(Z_list))+' '+str(np.min(Z_list))+'\n') #
    ofile.close()
    '''
    #--to print metallicity histogram: debugging purpose------
    plt.hist(np.log10(Z_list), 100, range =(-1, 50)) #
    plt.title('Z_map after conv') #
    plt.xlabel('log Z/Z_sol') #
    plt.yscale('log') #
    '''
    cbarlab = 'log(Z/Z_sol)'
    if getmap:
        map = plotmap(logOHobj_map, t, 'Metallicity', cbarlab, galsize, galsize/np.shape(mapcube)[0], cmin = cmin, cmax =cmax, hide = hide, saveplot=saveplot, islog=False)
    else:
        if not hide:
            if args.inspect: fig = plt.figure(figsize=(14,10))
            else: fig = plt.figure(figsize=(8,6))
            fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.1, right=0.95)
            ax = plt.subplot(111)
            plt.scatter(d.flatten(),Z_list, s=10, lw=0, c='b', label='pixels')
            if args.inspect:
                plt.scatter(d.flatten(),(np.divide(mapn2,np.add(maps2a,maps2b))).flatten(), s=10, lw=0, c='b', marker='s', label='pixels NII/SII')
                plt.scatter(d.flatten(),(np.divide(mapn2,mapha)).flatten(), s=10, lw=0, c='b', marker='^', label='pixels NII/Ha')
            #plt.axhline(10**(8.6-logOHsol),c='k',lw=0.5, label='trust') #line below which metallicity values should not be trusted for Kewley 2002 diag
            plt.plot(np.arange(galsize/2), np.poly1d((logOHgrad, logOHcen))(np.arange(galsize/2)) - logOHsol,c='r', label='True gradient')
            plt.axhline(0,c='k',linestyle='--',label='Zsol') #line for solar metallicity
            plt.xlabel('Galactocentric distance (kpc)')
            plt.ylabel(cbarlab)
            plt.xlim(0,galsize/2)
            plt.legend()
            plt.title(t)
            if args.inspect: plt.ylim(-1,3)
            else: plt.ylim(-1,1)
            if saveplot:
                fig.savefig(path+t+'.png')
        #---to compute apparent gradient and scatter---
        if calcgradient:
            ofile = open(outputfile,'a')
            ofile.write('Fitting gradient..'+'\n')
            ofile.close()
            try:
                d_list = np.ma.masked_array(d.flatten(), Z_list.mask)
                Z_list = np.ma.compressed(Z_list)
                d_list = np.ma.compressed(d_list)            
                if len(Z_list) == 0: raise ValueError
                linefit, linecov = np.polyfit(d_list, Z_list, 1, cov=True)
                ofile = open(outputfile,'a')
                ofile.write('Fit paramters: '+str(linefit)+'\n')
                ofile.write('Fit errors: '+str(linecov)+'\n')
                ofile.close()
                if args.toscreen:
                    print 'Fit paramters: '+str(linefit)
                    print 'Fit errors: '+str(linecov)
                if not nowrite:
                    gradfile = 'met_grad_log_paint'
                    if scale_exptime: gradfile += '_exp'
                    if fixed_SNR is not None: gradfile += '_fixedSNR'+str(fixed_SNR)
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
simulation  res_arcsec   res_phys   vres        power   size    logOHcen        logOHgrad       SNR_thresh      slope       slope_u     intercept       intercept_u       scale_exptime       realisation\n'
                        open(gradfile,'w').write(head)
                    with open(gradfile,'a') as fout:
                        if args.parm is not None: output = '\n'+fn+'\t\t'+str(res_arcsec)+'\t\t'+str(final_pix_per_beam*galsize/np.shape(mapcube)[0])+'\t\t'+str(vres)+'\t\t'+str(parm[0])+'\t\t'+str(parm[1])+'\t\t'+\
                        str(logOHcen)+'\t\t'+'\t\t'+str(logOHgrad)+'\t\t'+'\t\t'+str(SNR_thresh)+'\t\t'+'\t\t'+str('%0.4F'%linefit[0])+'\t\t'+str('%0.4F'%np.sqrt(linecov[0][0]))+'\t\t'+\
                        str('%0.4F'%linefit[1])+'\t\t'+'\t\t'+str('%0.4F'%np.sqrt(linecov[1][1]))+'\t\t'+'\t\t'+str(float(args.scale_exptime))+'\t\t'+'\t\t'+str(args.multi_realisation)
                        else: output = '\n'+fn+'\t\t'+str(res_arcsec)+'\t\t'+str(final_pix_per_beam*galsize/np.shape(mapcube)[0])+'\t\t'+str(vres)+'\t\t'+str(4.7)+'\t\t'+str(10)+'\t\t'+\
                        str(logOHcen)+'\t\t'+'\t\t'+str(logOHgrad)+'\t\t'+'\t\t'+str(SNR_thresh)+'\t\t'+'\t\t'+str('%0.4F'%linefit[0])+'\t\t'+str('%0.4F'%np.sqrt(linecov[0][0]))+'\t\t'+\
                        str('%0.4F'%linefit[1])+'\t\t'+'\t\t'+str('%0.4F'%np.sqrt(linecov[1][1]))+'\t\t'+'\t\t'+str(float(args.scale_exptime))+'\t\t'+'\t\t'+str(args.multi_realisation)
                        fout.write(output)
                    
                x_arr = np.arange(0,10,0.1)
                if not hide: plt.plot(x_arr, np.poly1d(linefit)(x_arr), c='b',label='Inferred gradient')
            except (TypeError, IndexError, ValueError):
                ofile = open(outputfile,'a')
                ofile.write('No data points for vres= '+str(vres)+' above given SNR_thresh of '+str(SNR_thresh)+'\n')
                ofile.close()
                pass
            
#-------------Fucntion for fitting multiple lines----------------------------
def fit_all_lines(wlist, llist, wave, flam, resoln, pix_i, pix_j, nres=5, z=0, z_err=0.0001, silent=True, showplot=False, outputfile='junk.txt') :
    ofile = open(outputfile,'a')
    wave, flam = np.array(wave), np.array(flam) #converting to numpy arrays
    kk, count, flux_array, flux_error_array = 1, 0, [], []
    ndlambda_left, ndlambda_right = [nres]*2 #how many delta-lambda wide will the window (for line fitting) be on either side of the central wavelength, default 5
    try:
        count = 1
        first, last = [wlist[0]]*2
    except IndexError:
        pass
    while kk <= len(llist):
        center1 = last
        if kk == len(llist):
            center2 = 1e10 #insanely high number, required to plot last line
        else:
            center2 = wlist[kk]
        if center2*(1. - ndlambda_left/resoln) > center1*(1. + ndlambda_right/resoln):
            leftlim = first*(1.-ndlambda_left/resoln) 
            rightlim = last*(1.+ndlambda_right/resoln)
            wave_short = wave[(leftlim < wave) & (wave < rightlim)]
            flam_short = flam[(leftlim < wave) & (wave < rightlim)]
            if not silent: 
                ofile.write('Trying to fit '+str(llist[kk-count:kk])+' line/s at once. Total '+str(count)+'\n')
            try: 
                popt, pcov = fitline(wave_short, flam_short, wlist[kk-count:kk], resoln, z=z, z_err=z_err)
                if showplot:
                    plt.axvline(leftlim, linestyle='--',c='g')
                    plt.axvline(rightlim, linestyle='--',c='g')
                ndlambda_left, ndlambda_right = [nres]*2
                if not silent: ofile.write('Done this fitting!'+'\n')
            except TypeError, er:
                if not silent: ofile.write('Trying to re-do this fit with broadened wavelength window..\n')
                ndlambda_left+=1
                ndlambda_right+=1
                continue
            except (RuntimeError, ValueError), e:
                popt = np.zeros(count*3 + 1) #if could not fit the line/s fill popt with zeros so flux_array gets zeros
                pcov = np.zeros((count*3 + 1,count*3 + 1)) #if could not fit the line/s fill popt with zeros so flux_array gets zeros
                ofile.write('Could not fit lines '+str(llist[kk-count:kk])+' for pixel '+str(pix_i)+', '+str(pix_j)+'\n')
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
                if showplot:
                    leftlim = popt_single[2]*(1.-nres/resoln) 
                    rightlim = popt_single[2]*(1.+nres/resoln)
                    wave_short_single = wave[(leftlim < wave) & (wave < rightlim)]
                    plt.plot(wave_short_single, np.log10(su.gaus(wave_short_single,1, *popt_single)),lw=1, c='r')
            if showplot:
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
    ofile.close()
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
def makemapcube(ppv, dispsol, wlist, llist, vres, outputfile='junk.txt', args=None):
    if len(dispsol) != np.shape(ppv)[2]:
        ofile = open(outputfile,'a')
        ofile.write('Length of dispersion array mismatch.'+'\n')
        ofile.close()
        sys.exit()
    x = np.shape(ppv)[0]
    mapcube = np.zeros((x,x, len(wlist)))
    errorcube = np.zeros((x,x, len(wlist)))
    resoln = c/vres
    for i in range(x):
        for j in range(x):
            dummytime = time.time() #
            flam = np.array(ppv[i,j,:])/dispsol #converting flux to per wavelength unit (i.e. ergs/s/pc^2/A), before sending off to fitting routine
            idx = np.where(flam<0)[0]
            if len(idx) > 1:
                if idx[0] == 0: idx = idx[1:]
                if idx[-1] == len(flam)-1 : idx = idx[:-1]
                flam[idx]=(flam[idx-1] + flam[idx+1])/2. # replacing negative fluxes with average of nearest neighbours
            mapcube[i,j,:], errorcube[i,j,:] = fit_all_lines(wlist, llist, dispsol, flam, resoln, i, j, nres=5, z=0, z_err=0.0001, silent=True, outputfile=outputfile) #put fluxes fit by fitting routine in each cell of map
            if args.toscreen: print 'deb425: time taken to fit spectrum in cell',i,j,'=',(time.time() - dummytime)/60.
        ofile = open(outputfile,'a')
        ofile.write('Fit row '+str(i)+' of '+str(x-1)+'\n')
        ofile.close()
    return mapcube, errorcube
#-------------------------------------------------------------------------------------------
def emissionmap(mapcube, llist, line, errorcube=None, SNR_thresh=None, saveplot=False, hide=False, cmin=None, cmax=None, fitsname=''):
    map = mapcube[:,:,np.where(llist == line)[0][0]]
    map = np.ma.masked_where(map<0., map)
    if errorcube is not None:
        map_u = errorcube[:,:,np.where(llist == line)[0][0]]        
        if SNR_thresh is not None:
            map = np.ma.masked_where(map/map_u <SNR_thresh, map)
    t = line+'_map:\n'+fitsname
    map = plotmap(map, t, line, 'Log '+line+' surface brightness in erg/s/pc^2', galsize, galsize/np.shape(map)[0], cmin = cmin, cmax =cmax, hide = hide, saveplot=saveplot)
    return map
#-------------------------------------------------------------------------------------------
def SFRmaps(mapcube, wlist, llist, res, res_phys, getmap=True, saveplot=False, hide=False, cmin=None, cmax=None):
    global info
    ages = s['age(MYr)']
    masses = s['mass(Msun)']
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
    SFRmapHa = mapcube[:,:,np.where(llist == 'H6562')[0][0]]
    g,x,y = calcpos(s, galsize, res)
    SFRmap_real = make2Dmap(masses, x, y, g, res)/((res*1e3)**2)
    agemap = 1e6*make2Dmap(ages, x, y, g, res, domean=True)
    #SFRmap_real /= agemap #dividing by mean age in the box
    SFRmap_real /= 5e6 #dividing by straight 5M years
    SFRmap_real[np.isnan(SFRmap_real)]=0
    SFRmap_real = rebin_old(SFRmap_real,np.shape(SFRmapHa))
    SFRmap_real = np.ma.masked_where(SFRmap_real<=0., SFRmap_real)
    SFRmapHa *= (const/1.37e-12) #Msun/yr/pc^2
    
    res = galsize/np.shape(SFRmapHa)[0]
    t = title(fn)+'SFR map for Omega = '+str(Om)+', resolution = '+str(res)+' kpc'+info
    if getmap:
        #SFRmapQ0 = plotmap(SFRmapQ0, t, 'SFRmapQ0', 'Log SFR(Q0) density in Msun/yr/pc^2', galsize, res, cmin = cmin, cmax =cmax, saveplot = saveplot, hide = hide)
        SFRmap_real = plotmap(SFRmap_real, t, 'SFRmap_real', 'Log SFR(real) density in Msun/yr/pc^2', galsize, res, cmin = cmin, cmax =cmax, saveplot = saveplot, hide = hide)
        SFRmapHa = plotmap(SFRmapHa, t, 'SFRmapHa', 'Log SFR(Ha) density in Msun/yr/pc^2', galsize, res, cmin = cmin, cmax =cmax, saveplot = saveplot, hide = hide)
        #SFRmap_comp = plotmap(SFRmap_comp, t, 'Log SFR(Q0)/SFR(real) in Msun/yr/pc^2', galsize, res, islog=False, maketheory=maketheory)   
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
        t= 'SFR comparison for '+fn+', res '+str(res)+' kpc'+info
        plt.ylabel('Log (Predicted SFR density) in Msun/yr/pc^2')
        plt.xlabel('Log (Actual SFR density) in Msun/yr/pc^2')
        plt.title(t)
        #plt.colorbar().set_label('Galactocentric distance (in pix)')
        plt.legend(bbox_to_anchor=(0.35, 0.88), bbox_transform=plt.gcf().transFigure)  
        if saveplot:
            fig.savefig(path+title(fn)[:-2]+'_'+t+'.png')
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
def spec_at_point(ppv, dispsol, wlist, llist, X, Y, wmin=None ,wmax=None,  hide=False, saveplot=False, changeunits=False, filename=None):
    global info
    fig = plt.figure(figsize=(14,6))
    fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.05, right=0.95)
    for i in wlist:
        plt.axvline(i,ymin=0.9,c='black')    
    if wmin is None: wmin = wlist[0]-50.
    if wmax is None: wmax = wlist[-1]+50.
    cbarlab = 'Log surface brightness in erg/s/pc^2' #label of color bar
    plt.plot(dispsol, np.log10(ppv[X][Y][:]),lw=1, c='b')
    t = 'Spectrum at pp '+str(X)+','+str(Y)+' for '+filename#+title(fn)+' Nebular + stellar for Om = '+str(Om)+', res = '+str(res)+' kpc'# + info
    plt.title(t)
    plt.ylabel(cbarlab)
    plt.xlabel('Wavelength (A)')
    if changeunits: plt.ylim(29-40,37-40)
    else: plt.ylim(30,37)
    plt.xlim(wmin,wmax)
    if not hide:
        plt.show(block=False)
    if saveplot:
        fig.savefig(path+t+'.png')
#-------------------------------------------------------------------------------------------
def plotintegmap(ppv, dispsol, wlist, llist, wmin=None ,wmax=None, cmin=None, cmax=None, hide=False, saveplot=False, changeunits=False):
    if wmin is None: wmin = wlist[0]-50.
    if wmax is None: wmax = wlist[-1]+50.
    dispsol=np.array(dispsol)
    ppv = np.array(ppv)
    ppv = ppv[:,:,(dispsol >= wmin) & (dispsol <= wmax)]
    cbarlab = 'Log surface brightness in erg/s/pc^2' #label of color bar
    if changeunits: 
        cbarlab = cbarlab[:cbarlab.find(' in ')+4] + 'erg/s/cm^2/A' #label would be ergs/s/pc^2/A if we choose to change units to flambda
    line = 'lambda-integrated wmin='+str(wmin)+', wmax='+str(wmax)+'\n'
    map = np.sum(ppv,axis=2)
    t = title(fn)+line+' map for Omega = '+str(Om)+', res = '+str(res)+' kpc'
    map = plotmap(map, t, line, cbarlab, galsize, res, cmin = cmin, cmax =cmax, hide = hide, saveplot=saveplot)            
#-------------------------------------------------------------------------------------------
def get_disp_array(vdel, vdisp, vres, nhr, nbin=1000, c=3e5, wmin=None, wmax=None, spec_smear=False):
    wlist, llist = readlist()
    if wmin is None: wmin = wlist[0]-50.
    if wmax is None: wmax = wlist[-1]+50.
    sig = 5*vdel/c 
    w = np.linspace(wmin, wmax, nbin)
    llist = llist[np.where(np.logical_and(wlist > wmin, wlist < wmax))] #truncate linelist as per wavelength range
    wlist = wlist[np.where(np.logical_and(wlist > wmin, wlist < wmax))]
    for ii in wlist:
        w1 = ii*(1-sig)
        w2 = ii*(1+sig)
        highres = np.linspace(w1, w2, nhr)
        w = np.insert(w, np.where(np.array(w)<w2)[0][-1]+1, highres)
    w = np.sort(w)
    #-------------------------------------------------------------------------------------------
    if spec_smear:        
        new_w = [np.min(w)]
        while new_w[-1] < np.max(w):
            new_w.append(new_w[-1]*(1+vres/c))
        nwbin = len(new_w) #final no. of bins in wavelength dimension
        bin_index = np.digitize(w, new_w)
    else:
        nwbin = len(w) + 1
        new_w = w
        bin_index = -999 #dummy, not required if spec_smear is turned OFF
    return w, wmin, wmax, new_w, nwbin, wlist, llist, bin_index
#-------------------------------------------------------------------------------------------
def spec(s, Om, res, res_phys, final_pix_size, wmin=None, wmax=None, changeunits= False, spec_smear=False, addnoise = False, \
saveplot=False, smooth=False, ker = None, parm=None, hide=False, maketheory=False, scale_exptime=False, fixed_SNR=None, \
outputfile='junk.txt', H2R_filename=None, convolved_filename=None, skynoise_cube=None, args=None):
    global info, gradient_painted, gradtext
    w, wmin, wmax, new_w, nwbin, wlist, llist, bin_index = get_disp_array(vdel, vdisp, vres, nhr, wmin=wmin, wmax=wmax, spec_smear=spec_smear)
    cbarlab = 'Log surface brightness in erg/s/pc^2' #label of color bar
    if changeunits: 
        cbarlab = cbarlab[:cbarlab.find(' in ')+4] + 'erg/s/cm^2/A' #label would be ergs/s/pc^2/A if we choose to change units to flambda
    if H2R_filename and os.path.exists(H2R_filename):
        ppv = fits.open(H2R_filename)[0].data
        ofile = open(outputfile,'a')
        ofile.write('Reading existing H2R file cube from '+H2R_filename+'\n')
        ofile.close()
    else:
        #-------------------------------------------------------------------------------------------
        g,x,y = calcpos(s, galsize, res)
        ppv = np.zeros((g,g,nwbin - 1))
        funcar = readSB(wmin, wmax)
        #-------------------------------------------------------------------------------------------
    
        for j in range(len(s)):
            ofile = open(outputfile,'a')
            ofile.write('Particle '+str(j+1)+' of '+str(len(s))+'\n')
            vz = float(s['vz'][j])
            a = int(round(s['age(MYr)'][j]))
            f = np.multiply(funcar[a](w),(300./1e6)) ### to scale the continuum by 300Msun, as the ones produced by SB99 was for 1M Msun
            flist=[]
            for l in llist:
                try: flist.append(s[l][j])
                except: continue
            flist = np.multiply(np.array(flist), const)
        
            for i, fli in enumerate(flist):
                f = gauss(w, f, wlist[i], fli, vdisp, vz) #adding every line flux on top of continuum
            if changeunits:
                f /= (w*3.086e18**2) #changing units for David Fisher: from ergs/s to ergs/s/A; the extra factor is to make it end up as /cm^2 insted of /pc^2
            if spec_smear: 
                f = [f[bin_index == ii].sum() for ii in range(1, len(new_w))]
            ppv[int(x[j]/res)][int(y[j]/res)][:] += f #ergs/s
            ofile.close()
        ofile = open(outputfile,'a')
        ofile.write('Done reading in all HII regions in '+str((time.time() - start_time)/60)+' minutes.\n')
        ofile.close()
        write_fits(H2R_filename, ppv, fill_val=np.nan, outputfile=outputfile)
    #-------------------------Now ideal PPV is ready: do whatever with it------------------------------------------------------------------
    
    if spec_smear: w = new_w[1:]
    if smooth and not maketheory:
        if skynoise_cube and os.path.exists(skynoise_cube):
            skynoise = fits.open(skynoise_cube)[0].data
            ofile = open(outputfile,'a')
            ofile.write('Reading existing skynoise cube from '+skynoise_cube+'\n')
            ofile.close()
        elif addnoise:
            ofile = open(outputfile,'a')
            ofile.write('Computing skynoise cube..\n')
            ofile.close()
            skynoise = getskynoise(w, final_pix_size)            
            write_fits(skynoise_cube, skynoise, fill_val=np.nan, outputfile=outputfile)
        else: skynoise = None
        if os.path.exists(convolved_filename): #read it in if the convolved cube already exists
            ofile = open(outputfile,'a')
            ofile.write('Reading existing convolved cube from '+convolved_filename+'\n')
            ofile.close()
        else: #make the convolved cube and save it if it doesn't exist already
            ker, sig, pow, size, fwhm, new_res, dummmy = getsmoothparm(parm, ker, res, res_phys)
            map = rebin(ppv[:,:,0], res, new_res)
            ppv_rebinned = np.zeros((np.shape(map)[0],np.shape(map)[1],np.shape(ppv)[2]))
            ppv_rebinned[:,:,0] = map
            for ind in range(1,np.shape(ppv)[2]):
                ppv_rebinned[:,:,ind] = rebin(ppv[:,:,ind], res, new_res) #re-bin 2d array before convolving to make things faster(previously each pixel was of size res)    
            ofile = open(outputfile,'a')
            ofile.write('Using '+ker+' kernel.\nUsing parameter set: sigma= '+str(sig)+', size= '+str(size,)+'\n')
            ofile.close()
            if args.toscreen: print 'deb652: trying to parallely convolve..' #
            binned_cubename = path + 'temp_binned_cube.fits'
            write_fits(binned_cubename, ppv_rebinned, fill_val=np.nan, outputfile=outputfile, silent=True)
            funcname = HOME+'/models/enzo_model_code/parallel_convolve.py'
            if args.toscreen: silent = ''
            else: silent = ' --silent'
            command = 'mpirun -np '+str(ncores)+' python '+funcname+' --parallel --sig '+str(sig)+' --pow '+str(pow)+' --size '+str(size)+' --ker '+ker+\
            ' --convolved_filename '+convolved_filename+' --outputfile '+outputfile+' --binned_cubename '+binned_cubename + silent
            subprocess.call([command],shell=True)
            subprocess.call(['rm -r '+binned_cubename],shell=True)
            
        convolved_cube = fits.open(convolved_filename)[0].data #reading in convolved cube from file
        ppv= smoothcube(convolved_cube, res, res_phys, skynoise=skynoise, parm=parm, addnoise = addnoise, maketheory=maketheory, \
        changeunits=changeunits, fixed_SNR=fixed_SNR, outputfile=outputfile, silent=True, units_in_photon=False) #spatially smooth the PPV using certain parameter set
        
    res = np.round(galsize/np.shape(ppv)[0],2) #kpc
    ofile = open(outputfile,'a')
    ofile.write('Final pixel size on target frame = '+str(res)+' kpc'+' and shape = ('+str(np.shape(ppv)[0])+','+str(np.shape(ppv)[1])+','+str(np.shape(ppv)[2])+') \n')
    
    #-------------------------Now realistic (smoothed, noisy) PPV is ready------------------------------------------------------------------
    if not hide: 
        fig = plt.figure(figsize=(14,6))
        fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.1, right=0.95)
        ax = plt.subplot(111)
        for i in wlist:
            plt.axvline(i,ymin=0.9,c='black')    
    
        #-------------------------------------------------------------------------------------------
        plt.plot(w, np.log10(np.sum(ppv,axis=(0,1))),lw=1)
        t = 'Spectrum for total, for '+title(fn)+' Nebular+ stellar for Om = '+str(Om)+', res = '+str(res)+' kpc' + info
        #-------------------------------------------------------------------------------------------
        plt.title(t)
        plt.ylabel(cbarlab)
        plt.xlabel('Wavelength (A)')
        if changeunits: plt.ylim(29-40,37-40)
        else: plt.ylim(32,40)
        plt.xlim(wmin,wmax)
        plt.show(block=False)
        if saveplot:
            fig.savefig(path+t+'.png')
    ofile.write('Returning PPV as variable "ppvcube"'+'\n')
    ofile.close()
    return ppv                
#-------------------------------------------------------------------------------------------
def inspectmap(s, Om, res, res_phys, line='OIII5007', cmin=None, cmax=None, ppvcube=None, mapcube=None, errorcube=None, SNR_thresh=None, changeunits= False, saveplot=False, plotmet=False, hide=False, args=None):
    g,x,y = calcpos(s, galsize, res)
    
    g2=np.shape(ppvcube)[0]
    if plotmet:
        log_ratio = np.log10(np.divide(s['NII6584'],(s['SII6730']+s['SII6717']))) + 0.264*np.log10(np.divide(s['NII6584'],s['H6562']))
        logOHobj = log_ratio + 0.45*(log_ratio + 0.3)**5
        if args.toscreen: 
            print 'all HIIR n2, s2, ha medians', np.median(s['NII6584']), np.median(s['SII6730']), np.median(s['H6562'])
            print 'all HIIR n2, s2, ha integrated', np.sum(s['NII6584']), np.sum(s['SII6730']), np.sum(s['H6562'])
            print 'all HIIR Z/Zsol median', np.median(10**logOHobj)
        d = np.sqrt((x-galsize/2)**2 + (y-galsize/2)**2)
        plt.scatter(d,logOHobj,c='r',s=5,lw=0,label='indiv HII reg')
        plt.scatter(d,np.divide(s['NII6584'],(s['SII6730']+s['SII6717'])),c='r',s=5,lw=0,marker='s',label='indiv HII reg NII/SII')
        plt.scatter(d,np.divide(s['NII6584'],s['H6562']),c='r',s=5,lw=0,marker='^',label='indiv HII reg NII/Ha')
        '''
        #small check for DIG vs HIIR
        plt.scatter(d,s['SII6717']/s['H6562'],c='k',s=5,lw=0,label='indiv HII reg-SII/Ha') #
        plt.axhline(0.35, c='cyan',label='max allowed for HIIR') #
        print 'DIG analysis: SII6717/Halpha ratio for indiv HII regions: min, max, median',\
        np.min(s['SII6717']/s['H6562']), np.max(s['SII6717']/s['H6562']), np.median(s['SII6717']/s['H6562']) #
        '''
        tempn2 = make2Dmap(s['NII6584'], x, y, g2, galsize/g2)
        temps2a = make2Dmap(s['SII6717'], x, y, g2, galsize/g2)
        temps2b = make2Dmap(s['SII6730'], x, y, g2, galsize/g2)
        tempha = make2Dmap(s['H6562'], x, y, g2, galsize/g2)
        
        log_ratio = np.log10(np.divide(tempn2,(temps2a+temps2b))) + 0.264*np.log10(np.divide(tempn2,tempha))
        logOHobj = log_ratio + 0.45*(log_ratio + 0.3)**5
        if args.toscreen: 
            print 'summed up HIIR n2, s2a, s2b, ha medians', np.median(tempn2), np.median(temps2a), np.median(temps2b), np.median(tempha)
            print 'summed up HIIR n2, s2a, s2b, ha integrated', np.sum(tempn2), np.sum(temps2a), np.sum(temps2b), np.sum(tempha)
            print 'all HIIR Z/Zsol median', np.median(10**logOHobj)
        b = np.linspace(-g2/2 + 1,g2/2,g2)*(galsize)/g2 #in kpc
        d = np.sqrt(b[:,None]**2+b**2)
        plt.scatter(d.flatten(),logOHobj.flatten(),c='g',s=5,lw=0,label='summed up HII reg')
        plt.scatter(d.flatten(),(np.divide(tempn2,(temps2a+temps2b))).flatten(),c='g',s=5,lw=0,marker='s',label='summed up HII reg NII/SII')
        plt.scatter(d.flatten(),(np.divide(tempn2,tempha)).flatten(),c='g',s=5,lw=0,marker='^',label='summed up HII reg NII/Ha')
        plt.legend()
        #plt.show(block=False)
        '''
        map = plotmap(tempn2, 'NII6584'+': H2R summed up', 'trial', 'log flux(ergs/s)', galsize, galsize/g2, cmin=cmin, cmax=cmax, hide = False, saveplot=saveplot, islog=True)
        map = plotmap(temps2a, 'SII6717'+': H2R summed up', 'trial', 'log flux(ergs/s)', galsize, galsize/g2, cmin=cmin, cmax=cmax, hide = False, saveplot=saveplot, islog=True)
        map = plotmap(temps2b, 'SII6730'+': H2R summed up', 'trial', 'log flux(ergs/s)', galsize, galsize/g2, cmin=cmin, cmax=cmax, hide = False, saveplot=saveplot, islog=True)
        map = plotmap(tempha, 'H6562'+': H2R summed up', 'trial', 'log flux(ergs/s)', galsize, galsize/g2, cmin=cmin, cmax=cmax, hide = False, saveplot=saveplot, islog=True)
        '''
    else:        
        fig = plt.figure(figsize=(8,8))
        fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.1, left=0.1, right=0.9)
        ax = plt.subplot(111)
        pl=ax.scatter(x-15,y-15,c=np.log10(s[line]), lw=0,s=3,vmin=cmin,vmax=cmax) #x,y in kpc
        plt.title(line+': indiv H2R') 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(pl, cax=cax).set_label('log flux(ergs/s)')    
    
        temp = np.zeros((g2,g2))
        for j in range(len(s)):
            temp[int(x[j]*g2/galsize)][int(y[j]*g2/galsize)] += s[line][j]
        map = plotmap(temp, line+': H2R summed up', 'trial', 'log flux(ergs/s)', galsize, galsize/g2, cmin=cmin, cmax=cmax, hide = False, saveplot=saveplot, islog=True)
           
    if mapcube is not None:
        mapn2 = mapcube[:,:,np.where(llist == 'NII6584')[0][0]]
        maps2a = mapcube[:,:,np.where(llist == 'SII6717')[0][0]]
        maps2b = mapcube[:,:,np.where(llist == 'SII6730')[0][0]]
        mapha = mapcube[:,:,np.where(llist == 'H6562')[0][0]]
        mapn2 = np.ma.masked_where(mapn2<=0., mapn2)
        maps2a = np.ma.masked_where(maps2a<=0., maps2a)
        maps2b = np.ma.masked_where(maps2b<=0., maps2b)
        mapha = np.ma.masked_where(mapha<=0., mapha)

        if errorcube is not None:
            #loading all the flux uncertainties
            mapn2_u = errorcube[:,:,np.where(llist == 'NII6584')[0][0]]
            maps2a_u = errorcube[:,:,np.where(llist == 'SII6717')[0][0]]
            maps2b_u = errorcube[:,:,np.where(llist == 'SII6730')[0][0]]
            mapha_u = errorcube[:,:,np.where(llist == 'H6562')[0][0]]
            mapn2_u = np.ma.masked_where(mapn2_u<=0., mapn2_u)
            maps2a_u = np.ma.masked_where(maps2a_u<=0., maps2a_u)
            maps2b_u = np.ma.masked_where(maps2b_u<=0., maps2b_u)
            mapha_u = np.ma.masked_where(mapha_u<=0., mapha_u)
        
            if SNR_thresh is not None:
                #imposing SNR cut
                mapn2 = np.ma.masked_where(mapn2/mapn2_u <SNR_thresh, mapn2)
                maps2a = np.ma.masked_where(maps2a/maps2a_u <SNR_thresh, maps2a)
                maps2b = np.ma.masked_where(maps2b/maps2b_u <SNR_thresh, maps2b)
                mapha = np.ma.masked_where(mapha/mapha_u <SNR_thresh, mapha)

        g = np.shape(mapcube)[0]
        if args.toscreen:
            print 'mapn2, s2a, s2b, ha max',np.max(mapn2), np.max(maps2a), np.max(maps2b), np.max(mapha)
            print 'mapn2, s2a, s2b, ha min',np.min(mapn2), np.min(maps2a), np.min(maps2b), np.min(mapha)
            print 'mapn2, s2a, s2b, ha integrated',np.sum(mapn2*(galsize*1000./g)**2), np.sum(maps2a*(galsize*1000./g)**2), np.sum(maps2b*(galsize*1000./g)**2), np.sum(mapha*(galsize*1000./g)**2), 'ergs/s' #
            print '#cells=',g,'each cell=',galsize*1000./g,'pc'
        if plotmet:
            map = plotmap(mapn2, 'NII6584'+' map after fitting', 'Metallicity', 'log flux(ergs/s/pc^2)', galsize, galsize/g, cmin = cmin, cmax =cmax, hide = hide, saveplot=saveplot, islog=True)
            map = plotmap(maps2a, 'SII6717'+' map after fitting', 'Metallicity', 'log flux(ergs/s/pc^2)', galsize, galsize/g, cmin = cmin, cmax =cmax, hide = hide, saveplot=saveplot, islog=True)
            map = plotmap(maps2b, 'SII6730'+' map after fitting', 'Metallicity', 'log flux(ergs/s/pc^2)', galsize, galsize/g, cmin = cmin, cmax =cmax, hide = hide, saveplot=saveplot, islog=True)
            map = plotmap(mapha, 'H6562'+' map after fitting', 'Metallicity', 'log flux(ergs/s/pc^2)', galsize, galsize/g, cmin = cmin, cmax =cmax, hide = hide, saveplot=saveplot, islog=True)
        else:
            map = plotmap(mapcube[:,:,np.where(llist == line)[0][0]]*(galsize*1000./g)**2, line+' map after fitting', 'Metallicity', 'log integ flux(ergs/s)', galsize, galsize/g, cmin = cmin, cmax =cmax, hide = False, saveplot=saveplot, islog=True)
    
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
def smoothcube(cube, res, res_phys, skynoise=None, addnoise = False, maketheory = False, parm=None, changeunits=False, \
fixed_SNR=None, outputfile='junk.txt', silent=True, units_in_photon=False):
    nslice = np.shape(cube)[2]
    new_cube = np.zeros(np.shape(cube))
    
    for k in range(nslice):
        ofile = open(outputfile,'a')
        ofile.write('Making observable slice '+str(k+1)+' of '+str(nslice)+'\n')
        ofile.close()   
        skynoiseslice = skynoise[k] if skynoise is not None else None
        new_cube[:,:,k] = makeobservable(cube[:,:,k], skynoiseslice=skynoiseslice, addnoise =addnoise, changeunits = changeunits, silent=silent, units_in_photon=units_in_photon, fixed_SNR=fixed_SNR, outputfile=outputfile)
   
    return new_cube
#-------------------------------------------------------------------------------------------
def makeobservable(map, skynoiseslice=None, addnoise=False, changeunits=False, silent=False, units_in_photon=False, fixed_SNR=None, outputfile='junk.txt'):
    global exptime
    size = galsize/np.shape(map)[0]
    factor = flux_ratio * exptime * el_per_phot / gain #
    if not units_in_photon: factor /= (planck * nu) #to bring it to units of photons, or rather, ADUs
    if changeunits: factor *= 3.086e18**2 * c*1e3 / nu * 1e10 #in case the units are in ergs/s/cm^2/A instead of ergs/s
    map *= factor #to get in counts from ergs/s
    if addnoise: map = makenoisy(map, skynoiseslice=skynoiseslice, factor=gain, silent=silent, fixed_SNR=fixed_SNR, outputfile=outputfile) #factor=gain as it is already in counts (ADU), need to convert to electrons for Poisson statistics
    map = np.ma.masked_where(np.log10(map)<0., map) #clip all that have less than 1 count
    map = np.ma.masked_where(np.log10(map)>5., map) #clip all that have more than 100,000 count i.e. saturating
    map /= factor #convert back to ergs/s from counts
    map /= (size*1e3)**2 #convert to ergs/s/pc^2 from ergs/s
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
def makenoisy(data, skynoiseslice=None, factor=None, silent=False, fixed_SNR=None, outputfile='junk.txt'):
    global exptime
    ofile = open(outputfile,'a')
    dummy = copy.copy(data)
    size = galsize/np.shape(data)[0]
    if factor is None:
        factor = flux_ratio * exptime * el_per_phot / (planck * nu)
    data *= factor #to transform into counts (electrons) from physical units
    if fixed_SNR is not None: #adding only fixed amount of SNR to ALL spaxels
        noisydata = data + np.random.normal(loc=0., scale=np.abs(data/fixed_SNR), size=np.shape(data)) #drawing from normal distribution about a mean value of 0 and width =counts/SNR
    else:
        noisydata = np.random.poisson(lam=data, size=None) #adding poisson noise to counts (electrons)
        noisydata = noisydata.astype(float)
        #dummy = plotmap(noisydata, 'after poisson', 'junk', 'counts', galsize, galsize/np.shape(noisydata)[0], islog=False) #
        readnoise = np.sqrt(2*7) * np.random.normal(loc=0., scale=3.5, size=np.shape(noisydata)) #to draw gaussian random variables from distribution N(0,3.5) where 3.5 is width in electrons per pixel
                                    #sqrt(14) is to account for the fact that for SAMI each spectral fibre is 2 pix wide and there are 7 CCD frames for each obsv
        noisydata += readnoise #adding readnoise
        #dummy = plotmap(noisydata, 'after readnoise', 'junk', 'counts', galsize, galsize/np.shape(noisydata)[0], islog=False) #
        if skynoiseslice is not None and skynoiseslice != 0: 
            skynoise = np.random.normal(loc=0., scale=np.abs(skynoiseslice), size=np.shape(noisydata)) #drawing from normal distribution about a sky noise value at that particular wavelength
            noisydata /= exptime #converting to electrons/s just to add skynoise, bcz skynoise is also in el/s units
            noisydata += skynoise #adding sky noise
            noisydata *= exptime #converting back to electrons units
        #dummy = plotmap(noisydata, 'after skynoise', 'junk', 'counts', galsize, galsize/np.shape(noisydata)[0], islog=False) #

    noisydata /= factor #converting back to physical units from counts (electrons)
    if not silent:
        noise = noisydata - dummy    
        ofile.write('makenoisy: array median std min max'+'\n')
        ofile.write('makenoisy: data '+str(np.median(dummy))+' '+str(np.std(dummy))+' '+str(np.min(np.ma.masked_where(dummy<=0, dummy)))+' '+str(np.max(dummy))+'\n')
        ofile.write('makenoisy: noisydata '+str(np.median(noisydata))+' '+str(np.std(noisydata))+' '+str(np.min(np.ma.masked_where(noisydata<=0, noisydata)))+' '+str(np.max(noisydata))+'\n')
        ofile.write('makenoisy: noise '+str(np.median(noise))+' '+str(np.std(noise))+' '+str(np.min(np.ma.masked_where(noise<=0, noise)))+' '+str(np.max(noise))+'\n')
    ofile.close()
    return noisydata
#-------------------------------------------------------------------------------------------
def plotmap(map, title, savetitle, cbtitle, galsize, res, cmin = None, cmax = None, islog=True, saveplot=False, hide=False, maketheory=False):    
    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.1, left=0.1, right=0.9)
    ax = plt.subplot(111)
    if islog:
        map = np.ma.masked_where(map<0, map)
        if cmin is None: cmin = np.min(np.log10(map))
        if cmax is None: cmax = np.max(np.log10(map))     
        map = np.ma.masked_where(np.log10(map)<cmin, map)
        p = ax.imshow(np.log10(map), cmap='rainbow',vmin=cmin,vmax=cmax)
    else:
        if cmin is None: cmin = np.min(map)
        if cmax is None: cmax = np.max(map)     
        map = np.ma.masked_where(map<cmin, map)
        p = ax.imshow(map, cmap='rainbow',vmin=cmin,vmax=cmax)
    ax.set_xticklabels([i*res - galsize/2 for i in list(ax.get_xticks())])
    ax.set_yticklabels([i*res - galsize/2 for i in list(ax.get_yticks())])
    plt.ylabel('y(kpc)')
    plt.xlabel('x(kpc)')
    plt.title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(p, cax=cax).set_label(cbtitle)    
    #circle1 = plt.Circle((galsize/(2*res), galsize/(2*res)), galsize/(2*res), color='k')#
    #ax.add_artist(circle1)#
    if saveplot: fig.savefig(path+title+'.png')
    if hide: plt.close(fig)
    else: plt.show(block=False)
    return map
#-------------------------------------------------------------------------------------------
def getfn(outtag,fn,Om):
    return HOME+'/models/emissionlist'+outtag+'/emissionlist_'+fn+'_Om'+str(Om)+'.txt'
#-------------------------------------------------------------------------------------------
def getsmoothparm(parm, ker, res, res_phys):
    if parm is None:
        pow, size = 4.7, 5 #power and sigma parameters for 2D Moffat kernal in pixel units
    else:
        pow, size = parm[0], int(parm[1])
    #------------------------------------------------------------------
    #-----Compute effective seeing i.e. FWHM as: the resolution on sky (res_arcsec) -> physical resolution on target frame (res_phys) -> pixel units (fwhm)
    fwhm = min(res_phys/res, pix_per_beam) #choose the lesser of 10 pixels per beam or res/res_phys pixels per beam
    new_res = res_phys/fwhm #need to re-bin the map such that each pixel is now of size 'new_res' => we have 'fwhm' pixels within every 'res_phys' physical resolution element
    dummy = np.zeros((1300,1300)) #dummy array to check what actual resolution we are left with after rebinning
    dummy = rebin(dummy, res, new_res) #mock rebin
    final_pix_size = 26./np.shape(dummy)[0] #actual pixel size we will end up with
    fwhm = int(np.round(res_phys/final_pix_size)) #actual no. of pixels per beam we will finally have
    #------------------------------------------------------------------
    if ker == 'gauss':
        sig = int(gf2s * fwhm)
        size = sig*size if sig*size%2 == 1 else sig*size + 1 #because kernels need odd integer as size
    elif ker == 'moff':
        sig = int(np.round(fwhm/(2*np.sqrt(2**(1./pow)-1.))))
        size = sig*size if sig*size%2 == 1 else sig*size + 1 #because kernels need odd integer as size
    return ker, sig, pow, size, fwhm, new_res, final_pix_size
#-------------------------------------------------------------------------------------------
def getfitsname(parm, ker, res, res_phys, Om, wmin, wmax, args):
    global info, gradient_painted, gradtext, exptime
    wlist, llist = readlist()
    if wmin is None: wmin = wlist[0]-50.
    if wmax is None: wmax = wlist[-1]+50.
    info = ''
    if args.spec_smear: info += '_specsmeared_'+str(int(vres))+'kmps'
    info1 = info
    if args.smooth:
        ker, sig, pow, size, fwhm, new_res, final_pix_size = getsmoothparm(parm, ker, res, res_phys)
        info += '_smeared_'+ker+'_parm'+str(fwhm)+','+str(sig)+','+str(pow)+','+str(size)
    else: final_pix_size = res
    if args.changeunits: info += '_flambda'
    info2 = info
    if args.addnoise: info += '_noisy'
    if args.fixed_SNR is not None: info += '_fixedSNR'+str(fixed_SNR)
    if not args.maketheory: info+= '_obs'
    if args.exp is not None:
        exptime = float(args.exp)
    else:
        if args.scale_exptime:
            exptime = float(args.scale_exptime)*(res/final_pix_size)**2 #increasing exposure time quadratically with finer resolution, with fiducial values of 600s for 0.5"
        else:
            exptime = float(240000)*(res/final_pix_size)**2 #sec
    info += '_exp'+str(exptime)+'s'
    if args.multi_realisation: info += '_real'+str(args.multi_realisation)
    
    H2R_filename = 'H2R_'+fn+'Om='+str(Om)+'_'+str(wmin)+'-'+str(wmax)+'A' + info1+ gradtext+'.fits'
    skynoise_cubename = 'skycube_'+'pixsize_'+str(final_pix_size)+'_'+str(wmin)+'-'+str(wmax)+'A'+info1+'.fits'
    convolved_filename = 'convolved_'+fn+'Om='+str(Om)+',arc='+str(res_arcsec)+'_'+str(wmin)+'-'+str(wmax)+'A' + info2+ gradtext +'.fits'
    fitsname = 'PPV_'+fn+'Om='+str(Om)+',arc='+str(res_arcsec)+'_'+str(wmin)+'-'+str(wmax)+'A' + info+ gradtext +'.fits'
    
    return fitsname, final_pix_size, H2R_filename, convolved_filename, skynoise_cubename
#-------------------------------------------------------------------------------------------
def write_fits(filename, data, fill_val=np.nan, silent=False, outputfile='junk.txt'):
    ofile = open(outputfile,'a')
    hdu = fits.PrimaryHDU(np.ma.filled(data,fill_value=fill_val))
    hdulist = fits.HDUList([hdu])
    if filename[-5:] != '.fits':
        filename += '.fits'
    hdulist.writeto(filename, clobber=True)
    if not silent: ofile.write('Written file '+filename+'\n')    
    ofile.close()
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
#-------------------End of functions------------------------------------------------------------------------
#-------------------Begin main code------------------------------------------------------------------------
global info, gradient_painted, gradtext, exptime, ncores
col_ar=['m','blue','steelblue','aqua','lime','darkolivegreen','goldenrod','orangered','darkred','dimgray']
logOHsun = 8.77
outtag = '_sph_logT4.0_MADtemp_Z0.05,5.0_age0.0,5.0_lnII5.0,12.0_lU-4.0,-1.0_4D'
galsize = 26.0 #kpc 
c = 3e5 #km/s
H0 = 70. #km/s/Mpc Hubble's constant
planck = 6.626e-27 #ergs.sec Planck's constant
nu = 5e14 #Hz H-alpha frequency to compute photon energy approximately
f_esc = 0.0
f_dust = 0.0
const = 1e0 #to multiply with nebular flux to make it comparable with SB continuum
if __name__ == '__main__':
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

    parser.add_argument('--scale_exptime')
    parser.add_argument('--multi_realisation')
    parser.add_argument("--path")
    parser.add_argument("--outfile")
    parser.add_argument("--file")
    parser.add_argument("--om")
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
    parser.add_argument("--exp")
    parser.add_argument("--epp")
    parser.add_argument("--snr")
    parser.add_argument("--Zgrad")
    parser.add_argument("--fixed_SNR")
    parser.add_argument("--ncores")
    args, leftovers = parser.parse_known_args()


    if args.path is not None:
        path = args.path
    else:
        path = HOME+'/Desktop/bpt/'
    subprocess.call(['mkdir -p '+path],shell=True) #create output durectory if it doesn't exist

    if args.file is not None:
        fn = args.file
    else:
        fn = 'DD0600_lgf' #which simulation to use

    if args.om is not None:
        Om_ar = [float(ar) for ar in args.om.split(',')]
    else:
        Om_ar = [0.5]

    if args.ppb is not None:
        pix_per_beam = float(args.ppb)
    else:
        pix_per_beam = 10

    if args.gain is not None:
        gain = float(args.gain)
    else:
        gain = 1.5

    if args.epp is not None:
        el_per_phot = float(args.epp)
    else:
        el_per_phot = 1.

    if args.z is not None:
        z = float(args.z)
    else:
        z = 0.013

    if args.rad is not None:
        rad = float(args.rad)
    else:
        rad = 1. #metre

    dist = calc_dist(z) #distance to object; in kpc
    flux_ratio = (rad/(2*dist*3.086e19))**2 #converting emitting luminosity to luminosity seen from earth, 3.08e19 factor to convert kpc to m

    if args.res is not None:
        res = float(args.res)
    else:
        res = 0.02 #kpc: simulation actual resolution 
    
    if args.arc is not None:
        res_arcsec = float(args.arc)
    else:
        res_arcsec = 0.5 #arcsec
    
    res_phys = res_arcsec*np.pi/(3600*180)*dist #kpc

    if args.line is not None:
        line = args.line
    else:
        line = 'OIII5007'# #whose emission map to be made

    if args.cmin is not None:
        cmin = float(args.cmin)
    else:
        cmin = None

    if args.cmax is not None:
        cmax = float(args.cmax)
    else:
        cmax = None

    if args.nhr is not None:
        nhr = int(args.nhr)
    else:
        nhr = 100 # no. of bins used to resolve the range lamda +/- 5sigma around emission lines

    if args.nbin is not None:
        nbin = int(args.nbin)
    else:
        nbin = 1000 #no. of bins used to bin the continuum into (without lines)

    if args.vdisp is not None:
        vdisp = float(args.vdisp)
    else:
        vdisp = 15 #km/s vel dispersion to be added to emission lines from MAPPINGS while making PPV

    if args.vdel is not None:
        vdel = float(args.vdel)
    else:
        vdel = 100 #km/s; vel range in which spectral resolution is higher is sig = 5*vdel/c
                    #so wavelength range of +/- sig around central wavelength of line is binned into further nhr bins

    if args.vres is not None:
        vres = float(args.vres)
    else:
        vres = 30 #km/s instrumental vel resolution to be considered while making PPV

    if args.wmin is not None:
        wmin = float(args.wmin)
    else:
        wmin = None #Angstrom; starting wavelength of PPV cube

    if args.wmax is not None:
        wmax = float(args.wmax)
    else:
        wmax = None #Angstrom; ending wavelength of PPV cube
    
    if args.snr is not None:
        SNR_thresh = float(args.snr)
    else:
        SNR_thresh = None

    if not args.keepprev:
        plt.close('all')

    if args.parm is not None:
        parm = [float(ar) for ar in args.parm.split(',')]
    else:
        parm = None # set of parameters i.e. telescope properties to be used for smearing cube/map
        
    if args.Zgrad is not None:
        logOHcen, logOHgrad = [float(ar) for ar in args.Zgrad.split(',')]
        gradtext = '_Zgrad'+str(logOHcen)+','+str(logOHgrad)
    else:
        logOHcen, logOHgrad = logOHsun, 0. # set of parameters i.e. telescope properties to be used for smearing cube/map
        gradtext = ''
    outtag = gradtext+outtag
    
    if args.ker is not None:
        ker = args.ker
    else:
        ker = 'moff' # convolution kernel to be used for smearing cube/map
        
    if args.fixed_SNR is not None:
        fixed_SNR = float(args.fixed_SNR) #fixed SNR used in makenoisy() function
    else:
        fixed_SNR = None 

    if args.ncores is not None:
        ncores = int(args.ncores) #number of cores used in parallel segments
    else:
        ncores = mp.cpu_count()/2

    #-----------------------jobs fetched--------------------------------------------------------------------
    for i, Om in enumerate(Om_ar):
        fitsname, final_pix_size, H2R_filename, convolved_filename, skynoise_cubename = getfitsname(parm, ker, res, res_phys, Om, wmin, wmax, args) # name of fits file to be written into
        final_pix_per_beam = int(fitsname[fitsname.find('parm')+4:fitsname.find('parm')+4+(fitsname[fitsname.find('parm')+4:]).find(',')]) #the actual FWHM used while generating the ppv cube, extracted from its name
        if args.toscreen: 
            print 'deb1322: res_phys, final pix per beam, final pix size, final shape=', res_phys, final_pix_per_beam, final_pix_size, galsize/final_pix_size, 'kpc' #
            print path+fitsname #
            if not os.path.exists(path+fitsname): print 'ppv does not exist' #
        #sys.exit() #
        if args.outfile is not None:
            outfile = args.outfile
        else:
            outfile = path + 'output_'+getfitsname(parm, ker, res, res_phys, Om, wmin, wmax, args)[0][:-5]+'.txt' # name of fits file to be written into
        #------------write output txt file-------------
        ofile = open(outfile,'w')
        if not args.silent:
            if not len(sys.argv) > 1:
                ofile.write('Insuffiecient information. Here is an example how to this routine might be called:\n')
                ofile.write('run plotobservables.py --addnoise --smooth --keep --vres 600 --spec_smear --plotspec\n')
                ofile.close()
                sys.exit()
            if args.path: ofile.write('Path: '+path+'\n')
            else: ofile.write('Default path: '+path+' Use --path option to specify.'+'\n')
            if args.outfile: ofile.write('Outfile: '+outfile+'\n')       
            else: ofile.write('Default outfile: '+outfile+' Use --outfile option to specify.'+'\n') 
            if args.file: ofile.write('Simulation= '+ fn+'\n')
            else: ofile.write('Default simulation= '+ fn+'. Use --file option to specify.'+'\n')
            if args.om: ofile.write('Omega= '+str(Om)+'\n')
            else: ofile.write('Default omega= '+str(Om)+'. Use --om option to specify Omega. You can supply , separated multiple omega values.'+'\n')
            if args.ppb: ofile.write('Minimum pix_per_beam= '+str(pix_per_beam)+'\n')
            else: ofile.write('Default minimum pix_per_beam of '+str(pix_per_beam)+'. Use --ppb option to specify pix_per_beam.'+'\n')
            if args.gain: ofile.write('Instrumental gain= '+str(gain)+'\n')
            else: ofile.write('Default gain= '+str(gain)+'. Use --gain option to specify gain.'+'\n')
            if args.epp: ofile.write('Electrons per photon= '+str(el_per_phot)+'\n')
            else: ofile.write('Default electrons per photon= '+str(el_per_phot)+'. Use --epp option to specify el_per_phot.'+'\n')
            if args.z: ofile.write('Redshift= '+str(z)+'\n')
            else: ofile.write('Default redshift of '+str(z)+'. Use --z option to specify redshift.'+'\n')
            if args.rad: ofile.write('Telescope radius chosen= '+str(rad)+' m'+'\n')
            else: ofile.write('Default telescope mirror rad= '+str(rad)+' m. Use --rad option to specify radius in metres.'+'\n')
            if args.res: ofile.write('Simulation resoltion forced to be res= '+str(res)+'\n')
            else: ofile.write('Default simulation res= '+str(res)+' kpc. Use --res option to specify simulation resolution.'+'\n')
            if args.arc: ofile.write('Telescope resoltion set to res='+str(res_arcsec)+'\n')
            else: ofile.write('Default telescope resolution= '+str(res_arcsec)+'. Use --arc option to specify telescope resolution.'+'\n')
            ofile.write('Resolution of telescope on object frame turns out to be res_phys~'+str(res_phys)+' kpc.'+'\n')
            if args.exp: ofile.write('Exposure time set to '+str(exptime)+' seconds.'+'\n')
            elif args.scale_exptime: ofile.write('Exposure time scaled to '+str(exptime)+' seconds.'+'\n')
            else: ofile.write('Default exptime= '+str(exptime)+' s. Use --exp option to specify exposure time in seconds.'+'\n')
            if args.line: ofile.write('line= '+line+'\n')
            elif args.map: ofile.write('Default line:'+line+'. Use --line option to specify line.'+'\n')
            if args.nhr: ofile.write('No. of bins used to resolve+/- 5sigma around emission lines= '+str(nhr)+'\n')
            else: ofile.write('Default No. of bins used to resolve+/- 5sigma around emission lines= '+str(nhr)+'. Use --nhr to specify.'+'\n')
            if args.nbin: ofile.write('No. of bins used to bin the continuum into (without lines)= '+str(nbin)+'\n')
            else: ofile.write('Default No. of bins used to bin the continuum into (without lines)= '+str(nbin)+'. Use --nbin to specify.'+'\n')
            if args.vdisp: ofile.write('Vel dispersion to be added to emission lines= '+str(vdisp)+' km/s.'+'\n')
            else: ofile.write('Default Vel dispersion to be added to emission lines= '+str(vdisp)+' km/s.'+'. Use --vdisp to specify.'+'\n')
            if args.vdel: ofile.write('Vel range in which spectral resolution is higher around central wavelength of line= '+str(vdel)+' km/s.'+'\n')
            else: ofile.write('Default Vel range in which spectral resolution is higher around central wavelength of line= '+str(vdel)+' km/s.'+'. Use --vdel to specify.'+'\n')
            if args.vres: ofile.write('Instrumental vel resolution to be considered while making PPV= '+str(vres)+' km/s.'+'\n')
            else: ofile.write('Default Instrumental vel resolution to be considered while making PPV= '+str(vres)+' km/s.'+'. Use --vres to specify.'+'\n')
            if args.wmin: ofile.write('Starting wavelength of PPV cube= '+str(wmin)+' A.'+'\n')
            else: ofile.write('Starting wavelength of PPV cube at beginning of line list.'+'\n')
            if args.wmax: ofile.write('Ending wavelength of PPV cube= '+str(wmax)+' A.'+'\n')
            else: ofile.write('Ending wavelength of PPV cube at end of line list.'+'\n')
            if args.snr: ofile.write('Applying SNR cut-off= '+str(SNR_thresh)+' on fitted lines.'+'\n')
            else: ofile.write('No SNR cut-off will be applied.'+'\n')
            ofile.write('Will run the parallel segments on '+str(ncores)+' cores.'+'\n')
            if args.smooth: 
                if args.parm: ofile.write('Parameter for smoothing= '+str(parm[0])+', '+str(parm[1])+'\n')
                else: ofile.write('Default smoothing parameter settings. Use --parm option to specify smearing parameters set.'+'\n')
                if args.ker: ofile.write('Smoothing profile used: '+ker+'\n')
                else: ofile.write('Default Moffat profile for smoothing.'+'\n')
            if args.Zgrad: ofile.write('Using metallicity painted HII regions, with central logOH+12 = '+str(logOHcen)+', and gradient = '+str(logOHgrad)+' dex per kpc'+'\n')        
            else: ofile.write('No additional metallicity gradient painted.'+'\n')
            ofile.write('Will be using/creating '+H2R_filename+' file.'+'\n')
            ofile.write('Will be using/creating '+skynoise_cubename+' file.'+'\n')
            ofile.write('Will be using/creating '+convolved_filename+' file.'+'\n')
            ofile.write('Will be using/creating '+fitsname+' file.'+'\n')
        ofile.close()
        #------------------------------------------------------

        fittedcube = fitsname.replace('PPV','fitted-map-cube') # name of mapcube file to be read in       
        s = ascii.read(getfn(outtag,fn,Om), comment='#', guess=False)
        if args.get_scale_length:
            scale_length = get_scale_length(s, args=args, outputfile=outfile)
        elif args.ppv: 
            ppvcube = spec(s, Om, res, res_phys, final_pix_size, wmin=wmin ,wmax=wmax, changeunits= args.changeunits, \
            spec_smear = args.spec_smear, saveplot = args.saveplot, smooth=args.smooth, parm = parm, ker = ker, hide=args.hide, \
            addnoise=args.addnoise, maketheory=args.maketheory, scale_exptime=args.scale_exptime, fixed_SNR=fixed_SNR, \
            outputfile=outfile, H2R_filename=path+H2R_filename, convolved_filename=path+convolved_filename, skynoise_cube=path+skynoise_cubename, args=args)
            write_fits(path+fitsname, ppvcube, fill_val=np.nan, outputfile=outfile)        
        else:
            if not os.path.exists(path+fitsname):
                if not args.silent:
                    ofile = open(outfile, 'a')
                    ofile.write('ppv file does not exist. Creating ppvcube..'+'\n')
                    ofile.close()
                ppvcube = spec(s, Om, res, res_phys, final_pix_size, wmin=wmin ,wmax=wmax, changeunits= args.changeunits, \
                spec_smear = args.spec_smear, saveplot = args.saveplot, smooth=args.smooth, parm = parm, ker = ker, hide=args.hide, \
                addnoise=args.addnoise, maketheory=args.maketheory, scale_exptime=args.scale_exptime, fixed_SNR=fixed_SNR, \
                outputfile=outfile, H2R_filename=path+H2R_filename, convolved_filename=path+convolved_filename, skynoise_cube=path+skynoise_cubename, args=args)
                write_fits(path+fitsname, ppvcube, fill_val=np.nan, outputfile=outfile)        
                fittedcube = fitsname.replace('PPV','fitted-map-cube')
            else:
                if not args.silent:
                    ofile = open(outfile, 'a')
                    ofile.write('Reading existing ppvcube from '+path+fitsname+'\n')
                    ofile.close()
                ppvcube = fits.open(path+fitsname)[0].data
            w, dummy2, dummy3, new_w, dummy4, wlist, llist, dummy7 = get_disp_array(vdel, vdisp, vres, nhr, wmin=wmin ,wmax=wmax, spec_smear=args.spec_smear)
            if args.spec_smear: dispsol = new_w[1:]
            else: dispsol = w
        
            if args.plotintegmap or args.plotspec:
                ofile = open(outfile,'a')
                if args.X is not None:
                    X = float(args.X)
                else:
                    X = np.shape(ppvcube)[0]/2 #p-p values at which point to extract spectrum from the ppv cube
                if args.plotspec and not args.silent: ofile.write('X position at which spectrum to be plotted= '+str(X)+'\n')

                if args.Y is not None:
                    Y = float(args.Y)
                else:
                    Y = np.shape(ppvcube)[0]/2 #p-p values at which point to extract spectrum from the ppv cube
                if args.plotspec and not args.silent: ofile.write('Y position at which spectrum to be plotted= '+str(Y)+'\n')
                ofile.close()
                
                if args.plotintegmap:
                    plotintegmap(ppvcube, dispsol, wlist, llist, wmin=wmin ,wmax=wmax, cmin=cmin, cmax=cmax, hide=args.hide, saveplot=args.saveplot, changeunits=args.changeunits)
                elif args.plotspec:
                    spec_at_point(ppvcube, dispsol, wlist, llist, X, Y, wmin=wmin ,wmax=wmax,  hide=args.hide, saveplot=args.saveplot, changeunits=args.changeunits, filename=fitsname)
        
            elif args.inspect and fittedcube == 'junk.fits':
                inspectmap(s, Om, res, res_phys, line=line, cmin=cmin, cmax=cmax, ppvcube=ppvcube, mapcube=None, errorcube=None, SNR_thresh=None, changeunits= args.changeunits, saveplot=args.saveplot, args=args)
            
            else:
                fittederror =fittedcube.replace('map','error')
                if os.path.exists(path+fittedcube) and not args.clobber:
                    mapcube = fits.open(path+fittedcube)[0].data
                    if SNR_thresh is not None: errorcube = fits.open(path+fittederror)[0].data
                    else: errorcube = None
                    if not args.silent: 
                        ofile = open(outfile, 'a')
                        ofile.write('Reading existing mapcube from '+path+fittedcube+'\n')
                        ofile.close()
                else:
                    if not args.silent:
                        ofile = open(outfile, 'a')
                        ofile.write('Mapfile does not exist. Creating mapcube..'+'\n')
                        ofile.close()
                    if args.spec_smear: smear = ' --spec_smear '
                    else: smear = ''
                    if args.toscreen: silent = ''
                    else: silent = ' --silent'
                    funcname = HOME+'/models/enzo_model_code/parallel_fitting.py'
                    command = 'mpirun -np '+str(ncores)+' python '+funcname+' --parallel --fitsname '+path+fitsname+' --vdel '+str(vdel)+\
                    ' --vdisp '+str(vdisp)+' --vres '+str(vres)+' --nhr '+str(nhr)+' --wmin '+str(wmin)+' --wmax '+str(wmax)+\
                    ' --fittedcube '+path+fittedcube+' --fittederror '+path+fittederror+' --outputfile '+outfile+smear+silent
                    subprocess.call([command],shell=True)
                    mapcube = fits.open(path+fittedcube)[0].data
                    errorcube = fits.open(path+fittederror)[0].data
                if args.bptpix: 
                    bpt_pixelwise(mapcube, wlist, llist, saveplot = args.saveplot)
                elif args.met: 
                    metallicity(mapcube, wlist, llist, final_pix_per_beam, errorcube=errorcube, SNR_thresh=SNR_thresh, getmap=args.getmap, hide=args.hide, saveplot = args.saveplot, cmin=cmin, cmax=cmax, calcgradient=args.calcgradient, nowrite=args.nowrite, scale_exptime=args.scale_exptime, fixed_SNR=fixed_SNR,args=args, outputfile=outfile)
                elif args.map: 
                    map = emissionmap(mapcube, llist, line, errorcube=errorcube, SNR_thresh=SNR_thresh, saveplot = args.saveplot, cmin=cmin, cmax=cmax, hide=args.hide, fitsname = fittedcube)
                elif args.sfr: 
                    SFRmap_real, SFRmapHa = SFRmaps(mapcube,wlist, llist, res, res_phys, getmap=args.getmap, cmin=cmin, cmax=cmax, saveplot = args.saveplot, hide=args.hide)
                else: 
                    if not args.silent: ofile.write('Wrong choice. Choose from:\n --bptpix, --map, --sfr, --met, --ppv, --plotinteg, --plotspec'+'\n')
            
                if args.inspect:
                    inspectmap(s, Om, res, res_phys, line=line, cmin=cmin, cmax=cmax, ppvcube=ppvcube, mapcube=mapcube, errorcube=errorcube, SNR_thresh=SNR_thresh, changeunits= args.changeunits, saveplot=args.saveplot, plotmet=args.met, hide=args.hide, args=args)
    #-------------------------------------------------------------------------------------------
    if args.hide: plt.close()
    else: plt.show(block=False)
    if not args.silent:
        ofile = open(outfile,'a')
        if args.saveplot: ofile.write('Saved here: '+path+'\n')
        ofile.write('Completed in %s minutes\n' % ((time.time() - start_time)/60))
        ofile.close()
