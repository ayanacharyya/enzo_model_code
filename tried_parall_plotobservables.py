import time
start_time = time.time()
import numpy as np
import subprocess
from matplotlib import pyplot as plt
from astropy.io import ascii, fits
import sys
from operator import itemgetter
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse as ap
parser = ap.ArgumentParser(description="observables generating tool")
import astropy.convolution as con
import copy
import multiprocessing as mp
#------------Reading pre-defined line list-----------------------------
def readlist():
    target = []
    finp = open('/Users/acharyya/Mappings/lab/targetlines.txt','r')
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
    plt.legend(bbox_to_anchor=(0.45, 0.35), bbox_transform=plt.gcf().transFigure)
#-------------------------------------------------------------------------------------------------
def gauss(w, f, w0, f0, v, vz):
    w0 = w0*(1+vz/c) #stuff for v_z component of HII region
    sigma = w0*v/c #c=3e5 km/s
    g = (f0/np.sqrt(2*np.pi*sigma**2))*np.exp(-((w-w0)**2)/(2*sigma**2))    
    f += g
    return f
#-------------------------------------------------------------------------------------------
def bpt_pixelwise(s, Om, res, saveplot=False, smooth=False, ker = None, parm=None, addnoise=False, maketheory=False):
    g,x,y = calcpos(s, galsize, res)
    b=np.linspace(-g/2,g/2,g)
    d = np.sqrt(b[:,None]**2+b**2)
    t = title(fn)+' Galactrocentric-distance color-coded, BPT of model \n\
for Omega = '+str(Om)+', resolution = '+str(res)+' kpc'
    mapn2 = make2Dmap(s['NII6584'], x, y, g, res)
    mapha = make2Dmap(s['H6562'], x, y, g, res)
    mapo3 = make2Dmap(s['OIII5007'], x, y, g, res)
    maphb = make2Dmap(s['HBeta'], x, y, g, res)
    
    if smooth:
        mapn2, info = smoothmap(mapn2, parm=parm, ker=ker, maskzero=True, addnoise = addnoise, maketheory = maketheory)
        mapha, info = smoothmap(mapha, parm=parm, ker=ker, maskzero=True, addnoise = addnoise, maketheory = maketheory)
        mapo3, info = smoothmap(mapo3, parm=parm, ker=ker, maskzero=True, addnoise = addnoise, maketheory = maketheory)
        maphb, info = smoothmap(maphb, parm=parm, ker=ker, maskzero=True, addnoise = addnoise, maketheory = maketheory)
        t += info

    mapn2 = np.divide(mapn2,mapha)
    mapo3 = np.divide(mapo3,maphb)
    plt.scatter((np.log10(mapn2)).flatten(),(np.log10(mapo3)).flatten(), s=4, c=d.flatten(), lw=0,vmin=0/res,vmax=13./res)
    plt.title(t)
    cb = plt.colorbar()
    cb.ax.set_yticklabels(str(res*float(x.get_text())) for x in cb.ax.get_yticklabels())
    cb.set_label('Galactocentric distance (in kpc)')
    if saveplot:
        fig.savefig(path+t+'.png')
#-------------------------------------------------------------------------------------------
def bpt_vs_radius(s, Om, saveplot=False):
    x = (s['x']-1500/2)*0.02 + galsize/2
    y = (s['y']-1500/2)*0.02 + galsize/2
    d = np.sqrt(x**2 + y**2)
    n2ha = np.divide(s['NII6584'],s['H6562'])
    o3hb = np.divide(s['OIII5007'],s['HBeta'])
    plt.scatter(np.log10(n2ha),np.log10(o3hb), s=2, c=d, lw=0,vmin=0,vmax=26)
    plt.title(title(fn)+' Galactocentric radii color-coded BPT of model for Omega = '+str(Om))
    cb = plt.colorbar()
    cb.ax.set_yticklabels(str(res*float(x.get_text())) for x in cb.ax.get_yticklabels())
    cb.set_label('Galactocentric distance (in kpc)')
    if saveplot:
        fig.savefig(path+fn+': BPT_vs_radius for Omega= '+str(Om)+'.png')

#-------------------------------------------------------------------------------------------
def addbpt(s,Om,col, saveplot=False):
    n2ha = np.divide(s['NII6584'],s['H6562'])
    o3hb = np.divide(s['OIII5007'],s['HBeta'])
    plt.scatter(np.log10(n2ha),np.log10(o3hb), s=2, c=col, lw=0, label='Omega = '+str(Om))
    lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=10)
    for ii in range(len(lgnd.legendHandles)): lgnd.legendHandles[ii]._sizes = [30]
    plt.title(title(fn)+' BPT of HII regions for different Omega value')
    if saveplot: 
        fig.savefig(path+fn+':Omega color coded BPT')

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
    s = ascii.read('/Users/acharyya/Mappings/lab/totalspec.txt',comment='#',guess=False)
    y = np.reshape(np.log10(np.divide(s['OIII5007'],s['HBeta'])),(6, 6, 4))
    x = np.reshape(np.log10(np.divide(s['NII6584'],s['H6562'])),(6, 6, 4))
    x[[3,4],:,:]=x[[4,3],:,:] #for clearer connecting lines between grid points
    y[[3,4],:,:]=y[[4,3],:,:] #
    plt.scatter(x,y, c='black', lw=0, s=2)
    meshplot3D(x,y, annotate=annotate)
    if saveplot:
        fig.savefig(path+fn+':BPT overlay')
#-------------------------------------------------------------------------------------------
def emissionmap(s, Om, res, line, saveplot=False, smooth=False, off=None, ker = None, parm=None, hide=False, addnoise = False, cmin=None, cmax=None, maketheory=False):
    g,x,y = calcpos(s, galsize, res)
    flux = s[line]/((res*1e3)**2) #ergs/s/pc^2
    t = title(fn)+line+' map for Omega = '+str(Om)+', resolution = '+str(res)+' kpc'
    map = make2Dmap(flux, x, y, g, res)
    if smooth:
        map, info = smoothmap(map, parm=parm, ker=ker, addnoise = addnoise, maketheory = maketheory)
        t += info
    map = plotmap(map, t, line, 'Log '+line+' surface brightness in erg/s/pc^2', galsize, res, cmin = cmin, cmax =cmax, hide = hide, saveplot=saveplot)
    return map
#-------------------------------------------------------------------------------------------
def SFRmaps(s, Om, reso, getmap=True, saveplot=False, smooth=False, ker = None, off=None, parm=None, hide=False, addnoise = False, cmin=None, cmax=None, maketheory=False):
    flux = s['H6562'] #ergs/s
    lum = s['logQ0'] #log(Q0) in photons/s
    ages = s['age(MYr)']
    masses = s['mass(Msun)']
    #----------to get correct conversion rate for lum to SFR----------------------------------------------
    SBmodel = 'starburst08' #this has continuous 1Msun/yr SFR 
    input_quanta = '/Users/acharyya/SB99-v8-02/output/'+SBmodel+'/'+SBmodel+'.quanta'
    SB99_age = np.array([float(x.split()[0]) for x in open(input_quanta).readlines()[6:]])
    SB99_logQ = np.array([float(x.split()[1]) for x in open(input_quanta).readlines()[6:]])
    const = 1./np.power(10,SB99_logQ)
    #-------------------------------------------------------------------------------------------
    const = 7.9e-42*1.37e-12 #factor to convert Q0 to SFR, value from literature
    #const = 1.0*const[-1] #factor to convert Q0 to SFR, value from corresponding SB99 file (07)
    #mean=[]
    try:
        dummy = iter(reso)
    except TypeError:
        reso = [reso]
    for i, res in enumerate(reso):
        g,x,y = calcpos(s, galsize, res)
        b=np.linspace(-g/2,g/2,g)
        #d = np.sqrt(b[:,None]**2+b**2)

        SFRmapHa = make2Dmap(flux, x, y, g, res)/((res*1e3)**2) #Just the H-alpha map in ergs/s/pc^2
        SFRmapQ0 = make2Dmap(lum, x, y, g, res, islog=True)/((res*1e3)**2) #Just the ionizing photon map in phot/s/pc^2
        SFRmap_real = make2Dmap(masses, x, y, g, res)/((res*1e3)**2)
        agemap = 1e6*make2Dmap(ages, x, y, g, res, domean=True)
        #SFRmap_real /= agemap #dividing by mean age in the box
        SFRmap_real /= 5e6 #dividing by straight 5M years
        SFRmap_real[np.isnan(SFRmap_real)]=0
        #SFRmap_comp = SFRmapQ0/SFRmap_real
        #mean.append(np.log10(np.mean(SFRmap_real)))
        t = title(fn)+'SFR map for Omega = '+str(Om)+', resolution = '+str(res)+' kpc'
        info = ''
        if smooth:
            SFRmapQ0, info = smoothmap(SFRmapQ0, parm=parm, ker=ker, maskzero=True, addnoise = addnoise, maketheory = maketheory, units_in_photon=True)
            SFRmapHa, info = smoothmap(SFRmapHa, parm=parm, ker=ker, maskzero=True, addnoise = addnoise, maketheory = maketheory)
            t += info
        else: 
            SFRmapHa = np.ma.masked_where(SFRmapHa<=0., SFRmapHa)
            SFRmapQ0 = np.ma.masked_where(SFRmapQ0<=0., SFRmapQ0)
        SFRmap_real = np.ma.masked_where(SFRmap_real<=0., SFRmap_real)
        SFRmapHa *= (const/1.37e-12) #Msun/yr/pc^2
        SFRmapQ0 *= const*(1-f_esc)*(1-f_dust) #Msun/yr/pc^2
        if getmap:
            SFRmapQ0 = plotmap(SFRmapQ0, t, 'SFRmapQ0', 'Log SFR(Q0) density in Msun/yr/pc^2', galsize, res, cmin = cmin, cmax =cmax, saveplot = saveplot, hide = hide)
            SFRmap_real = plotmap(SFRmap_real, t, 'SFRmap_real', 'Log SFR(real) density in Msun/yr/pc^2', galsize, res, cmin = cmin, cmax =cmax, saveplot = saveplot, hide = hide)
            SFRmapHa = plotmap(SFRmapHa, t, 'SFRmapHa', 'Log SFR(Ha) density in Msun/yr/pc^2', galsize, res, cmin = cmin, cmax =cmax, saveplot = saveplot, hide = hide)
            #SFRmap_comp = plotmap(SFRmap_comp, t, 'Log SFR(Q0)/SFR(real) in Msun/yr/pc^2', galsize, res, islog=False, maketheory=maketheory)   
        else:
            fig = plt.figure(figsize=(8,6))
            fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.1, left=0.1, right=0.95)
            ax = plt.subplot(111)
            ax.scatter((np.log10(SFRmap_real)).flatten(),(np.log10(SFRmapQ0)).flatten(), s=4, c='r', lw=0, label='SFR(Q0)')
            ax.scatter((np.log10(SFRmap_real)).flatten(),(np.log10(SFRmapHa)).flatten(), s=4, c='b', lw=0, label='SFR(Ha)')
            #ax.scatter((np.log10(SFRmap_real)).flatten(),(np.log10(SFRmapHa)).flatten(), s=4, c=col_ar[i], lw=0, label=str(res))
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
    return SFRmap_real, SFRmapQ0 
#-------------------------------------------------------------------------------------------------
def readSB(wmin, wmax):
    inpSB=open('/Users/acharyya/SB99-v8-02/output/starburst03/starburst03.spectrum','r')
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
#---------------parallelisation stuff----------------------------------------------------------------------------
def put_star_on_ppv(j, s, funcar, w, new_w, llist, const, vdisp, x, y, res, bin_index, changeunits, spec_smear):
    global ppv
    print 'Particle', j, 'of', len(s) #
    vz = float(s['vz'][j])
    a = int(round(s['age(MYr)'][j]))
    f = np.multiply(funcar[a](w),(300./1e6)) ### to scale the continuum by 300Msun, as the ones produced by SB99 was for 1M Msun
    flist=[]
    for l in llist:
        flist.append(s[l][j])
    flist = np.multiply(np.array(flist), const)
    for i, fli in enumerate(flist):
        f = gauss(w, f, wlist[i], fli, vdisp, vz) #adding every line flux on top of continuum
    if changeunits:
        f /= (w*3.086e18**2) #changing units for David Fisher: from ergs/s to ergs/s/A; the extra factor is to make it end up as /cm^2 insted of /pc^2
    if spec_smear: 
        f = [f[bin_index == ii].sum() for ii in range(1, len(new_w))]
    ppv[int(x[j]/res)][int(y[j]/res)][:] += np.divide(f,(res*1e3)**2) #ergs/s/pc^2
#-------------------------------------------------------------------------------------------
def spec(s, Om, res, col, wmin = None, wmax = None, changeunits= False, off=None, getcube=False, X=None, Y=None, spec_smear=False, plotspec=False, plotintegmap=False, addnoise = False, savecube=False, saveplot=False, smooth=False, ker = None, parm=None, hide=False, cmin=None, cmax=None, maketheory=False):
    global cube, info, ppv
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
    #-------------------------------------------------------------------------------------------
    global ppv
    g,x,y = calcpos(s, galsize, res)
    ppv = np.zeros((g,g,nwbin - 1))
    funcar = readSB(wmin, wmax)
    cbarlab = 'Log surface brightness in erg/s/pc^2' #label of color bar
    info = ''
    if changeunits: 
        cbarlab = cbarlab[:cbarlab.find(' in ')+4] + 'erg/s/cm^2/A' #label would be ergs/s/pc^2/A if we choose to change units to flambda
    #----------------------------parallelisation stuff to get ppv---------------------------------------------------------------
    p = mp.Pool(mp.cpu_count())
    for j in range(len(s)):
        p.apply_async(put_star_on_ppv, (j, s, funcar, w, new_w, llist, const, vdisp, x, y, res, bin_index, changeunits, spec_smear))              
    p.close()
    p.join()
    #-------------------------Now PPV is ready: do whatever with it------------------------------------------------------------------
    if spec_smear: 
        w = new_w[1:]
        if '_specsmeared' not in info: info += '_specsmeared'    
        if smooth and not plotintegmap:
            cube = copy.copy(ppv)
            smoothcube(ppv, parm=parm, ker=ker, addnoise = addnoise, maketheory=maketheory, changeunits=changeunits) #spatially smooth the PPV using certain parameter set
    if savecube:
        p = mp.Pool(mp.cpu_count())
        for k in range(np.shape(ppv)[2]):
            p.apply_async(plotmap_caller, (k, 'w slice %.2f A' %w[k], str(w[k]), cbarlab, galsize, res), dict(cmin = cmax, cmax =cmax, hide = True, saveplot=False, maketheory=maketheory))              
        p.close()
        p.join()
    #-------------------------------------------------------------------------------------------
    if plotintegmap:
        line = 'lambda-integrated wmin='+str(wmin)+', wmax='+str(wmax)+'\n'
        t = title(fn)+line+' map for Omega = '+str(Om)+', res = '+str(res)+' kpc'
        map = np.sum(ppv,axis=2)
        if smooth: map, info = smoothmap(map, parm=parm, ker=ker, addnoise = addnoise, maketheory=maketheory, changeunits=changeunits, info=info) #should not need this after smoothcube is working
        map = plotmap(map, t+info, line, cbarlab, galsize, res, cmin = cmin, cmax =cmax, hide = hide, saveplot=saveplot, maketheory=maketheory)            
        print 'Returning integrated map as variable "ppvcube"'
        return map
    else:
        fig = plt.figure(figsize=(8,6))
        fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.1, right=0.95)
        ax = plt.subplot(111)
        for i in wlist:
            plt.axvline(i,ymin=0.9,c='black')    
        #-------------------------------------------------------------------------------------------
        if plotspec:
            plt.plot(w, np.log10(ppv[X][Y][:]),lw=1, c=col)
            t = 'Spectrum at pp '+str(X)+','+str(Y)+' for '+title(fn)+' Nebular + stellar for Om = '+str(Om)+', res = '+str(res)+' kpc' + info
        else:
            plt.plot(w, np.log10(np.sum(ppv,axis=(0,1))),lw=1, c=col)
            t = 'Spectrum for total, for '+title(fn)+' Nebular+ stellar for Om = '+str(Om)+', res = '+str(res)+' kpc' + info
        #-------------------------------------------------------------------------------------------
        plt.title(t)
        plt.ylabel(cbarlab)
        plt.xlabel('Wavelength (A)')
        if changeunits: plt.ylim(29-40,37-40)
        else: plt.ylim(29,37)
        plt.xlim(wmin,wmax)
        if not hide:
            plt.show(block=False)
        if saveplot:
            fig.savefig(path+t+'.png')
        print 'Returning PPV as variable "ppvcube"'
        return ppv                
#-------------------------------------------------------------------------------------------
def calcpos(s, galsize, res):
    g = int(np.ceil(galsize/res))
    x = (s['x']-1500/2)*0.02 + galsize/2
    y = (s['y']-1500/2)*0.02 + galsize/2
    return g, x, y
#-------------------------------------------------------------------------------------------
def make2Dmap(data, xi, yi, gridsize, res, domean=False, islog=False):
    map = np.zeros((gridsize,gridsize))
    if domean:
        count = np.zeros((gridsize,gridsize))
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
def smoothmap(map, parm=None, ker='moff', maskzero=False, addnoise = False, maketheory=False, info='', silent = False, changeunits=False, units_in_photon=False):
    if parm is None:
        sig, pow, size = 1, 4, 10 #sigma and truncation length of 2D gaussian kernal in pixel units
    else:
        sig, pow, size = parm[0], parm[1], int(parm[2]*parm[0])
    if size%2 == 0:
        size += 1 #because kernels need odd integer as size
    if ker == 'gauss':
        kernel = con.Gaussian2DKernel(sig, x_size = size, y_size = size)
        if not silent: print 'Using Gaussian kernel.\nUsing parameter set: FWHM=', sig*2*np.sqrt(2*np.log(2)), ', size=', size, ' pixels.'
    elif ker == 'moff':
        kernel = con.Moffat2DKernel(sig, pow, x_size = size, y_size = size)
        if not silent: print 'Using Moffat kernel.\nUsing parameter set: FWHM=', sig*2*np.sqrt(2**(1./pow)-1.), ', pow=', pow, ', size=', size, ' pixels.'
    else:
        if not silent: print 'Kernel not identified. Using moffat. Use --ker <option> to specify kernel, where <option>=gauss OR moff'   
        sys.exit()
    map = con.convolve(map, kernel, boundary = 'fill', fill_value = 0.0, normalize_kernel=True)
    if '_smeared_' not in info: info += '\n_smeared_'+ker+'_parm'+str(parm)
    if maskzero:
        map = np.ma.masked_where(map<=0., map)
    if not maketheory: map, info = makeobservable(map, addnoise =addnoise, changeunits = changeunits, info = info, silent=silent, units_in_photon=units_in_photon)
    return map, info
#----------------------same as smoothmap but to be used by smoothcube in parallel---------------------------------------------------------------------
def smoothslice(k, parm=None, ker='moff', addnoise = False, maketheory=False, silent = False, changeunits=False, units_in_photon=False):
    global cube, info
    cube[:,:,k] = con.convolve(cube[:,:,k], kernel, boundary = 'fill', fill_value = 0.0, normalize_kernel=True)
    if not maketheory: cube[:,:,k], info = makeobservable(cube[:,:,k], addnoise =addnoise, changeunits = changeunits, info = info, silent=silent, units_in_photon=units_in_photon)
#-------------------------------------------------------------------------------------------
def smoothcube(parm=None, ker='moff', changeunits=False, addnoise = False, maketheory=False, silent = False):
    global cube, info
    if parm is None:
        sig, pow, size = 1, 4, 10 #sigma and truncation length of 2D gaussian kernal in pixel units
    else:
        sig, pow, size = parm[0], parm[1], int(parm[2]*parm[0])
    if size%2 == 0:
        size += 1 #because kernels need odd integer as size
    if ker == 'gauss':
        kernel = con.Gaussian2DKernel(sig, x_size = size, y_size = size)
        if not silent: print 'Using Gaussian kernel.\nUsing parameter set: FWHM=', sig*2*np.sqrt(2*np.log(2)), ', size=', size, ' pixels.'
    elif ker == 'moff':
        kernel = con.Moffat2DKernel(sig, pow, x_size = size, y_size = size)
        if not silent: print 'Using Moffat kernel.\nUsing parameter set: FWHM=', sig*2*np.sqrt(2**(1./pow)-1.), ', pow=', pow, ', size=', size, ' pixels.'
    else:
        if not silent: print 'Kernel not identified. Using moffat. Use --ker <option> to specify kernel, where <option>=gauss OR moff'   
        sys.exit()
    if '_smeared_' not in info: info += '\n_smeared_'+ker+'_parm'+str(parm)
    p = mp.Pool(mp.cpu_count())
    for k in range(np.shape(cube)[2]):
        p.apply_async(smoothslice, (k), dict(addnoise = addnoise, maketheory = maketheory, parm=parm, info=info, silent = True, changeunits=changeunits))              
        print 'smoothed slice', k, 'of', np.shape(cube)[2] #
    p.close()
    p.join()
#-------------------------------------------------------------------------------------------
def makeobservable(map, addnoise=False, changeunits=False, info='',silent=False, units_in_photon=False):
    factor = (res*1e3)**2 * flux_ratio * exptime * el_per_phot / gain
    if not units_in_photon: factor /= (planck * nu) #to bring it to units of photons, or rather, ADUs
    if changeunits: 
        factor *= 3.086e18**2 * c*1e3 / nu * 1e10 #in case the units are in ergs/s/cm^2/A instead of ergs/s/pc^2
        if '_flambda' not in info: info += '_flambda'
    map *= factor #to get in counts
    if addnoise: 
        map = makenoisy(map, factor=1., silent=silent)
        if '_noisy' not in info: info += '_noisy'
    map = np.ma.masked_where(np.log10(map)<0., map) #clip all that have less than 1 count
    map = np.ma.masked_where(np.log10(map)>5., map) #clip all that have more than 100,000 count i.e. saturating
    map /= factor
    if '_obs' not in info: info += '_obs'
    return map, info
#-------------------------------------------------------------------------------------------
def makenoisy(data, factor=None, silent=False):
    dummy = copy.copy(data)
    if factor is None:
        factor = (res*1e3)**2 * flux_ratio * exptime * el_per_phot / (planck * nu)
    data *= factor
    noisydata = np.random.poisson(lam=data, size=None)/factor
    noisydata = noisydata.astype(float)
    if not silent:
        noise = noisydata - dummy    
        print 'makenoisy: array mean std min max'
        print 'makenoisy: data', np.mean(dummy), np.std(dummy), np.min(np.ma.masked_where(dummy<=0, dummy)), np.max(dummy)
        print 'makenoisy: noisydata', np.mean(noisydata), np.std(noisydata), np.min(np.ma.masked_where(noisydata<=0, noisydata)), np.max(noisydata)
        print 'makenoisy: noise', np.mean(noise), np.std(noise), np.min(np.ma.masked_where(noise<=0, noise)), np.max(noise) 
    return noisydata
#-------------------------------------------------------------------------------------------
def plotmap_caller(k, t1, t2, cbarlab, galsize, res, cmin = None, cmax =None, hide = True, saveplot=False, maketheory=False):
    global ppv
    ppv[:,:,k] = plotmap(ppv[:,:,k], t1, t2, cbarlab, galsize, res, cmin = cmax, cmax =cmax, hide = True, saveplot=False, maketheory=maketheory)
    fig = plt.gcf() #get handle of current figure
    fig.savefig(path+fn+'_cube/map_for_Om='+str(Om_ar[0])+'_slice'+str(k)+info+'.png')
    plt.close()
#-------------------------------------------------------------------------------------------
def plotmap(map, title, savetitle, cbtitle, galsize, res, cmin = None, cmax = None, islog=True, saveplot=False, hide=False, maketheory=False):    
    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.1, left=0.1, right=0.9)
    ax = plt.subplot(111)
    map = np.ma.masked_where(map<0, map)
    if cmin is None: cmin = np.min(np.log10(map))
    if cmax is None: cmax = np.max(np.log10(map))     
    map = np.ma.masked_where(np.log10(map)<cmin, map)
    if islog:
        p = ax.imshow(np.log10(map), cmap='rainbow',vmin=cmin,vmax=cmax)
    else:
        p = ax.imshow(map, cmap='rainbow')
    ax.set_xticklabels([i*res - galsize/2 for i in list(ax.get_xticks())])
    ax.set_yticklabels([i*res - galsize/2 for i in list(ax.get_yticks())])
    plt.ylabel('y(kpc)')
    plt.xlabel('x(kpc)')
    plt.title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(p, cax=cax).set_label(cbtitle)
    if not hide:
        plt.show(block=False)
    if saveplot:
        fig.savefig(path+title+'.png')
    return map
#-------------------------------------------------------------------------------------------
def getfn(outtag,fn,Om):
    return '/Users/acharyya/models/emissionlist'+outtag+'/emissionlist_'+fn+'_Om'+str(Om)+'.txt'
#-------------------------------------------------------------------------------------------
def write_fits(filename, data, fill_val=np.nan):
    hdu = fits.PrimaryHDU(data.filled(fill_val))
    hdulist = fits.HDUList([hdu])
    if filename[-5:] != '.fits':
        filename += '.fits'
    hdulist.writeto(filename, clobber=True)
    print 'Written file', filename    
#-------------------------------------------------------------------------------------------
def calc_dist(z, H0 = 70.):
    dist = z*c*1e3/H0 #kpc
    return dist
#-------------------End of functions------------------------------------------------------------------------
#-------------------Begin main code------------------------------------------------------------------------
global ppv
col_ar=['m','blue','steelblue','aqua','lime','darkolivegreen','goldenrod','orangered','darkred','dimgray']
path = '/Users/acharyya/Desktop/bpt/'
outtag = '_logT4'
galsize = 26 #kpc 
c = 3e5 #km/s
H0 = 70. #km/s/Mpc Hubble's constant
planck = 6.626e-27 #ergs.sec Planck's constant
nu = 5e14 #Hz H-alpha frequency to compute photon energy approximately
f_esc = 0.0
f_dust = 0.0
const = 1e0 #to multiply with nebular flux to make it comparable with SB continuum
#plt.ion()
#-------------------arguments parsed-------------------------------------------------------
parser.add_argument('--addbpt', dest='addbpt', action='store_true')
parser.set_defaults(addbpt=False)
parser.add_argument('--bptpix', dest='bptpix', action='store_true')
parser.set_defaults(bptpix=False)
parser.add_argument('--bptrad', dest='bptrad', action='store_true')
parser.set_defaults(bptrad=False)
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
parser.add_argument('--getmap', dest='getmap', action='store_true')
parser.set_defaults(getmap=False)
parser.add_argument('--ppv', dest='ppv', action='store_true')
parser.set_defaults(ppv=False)
parser.add_argument('--plotmap', dest='plotmap', action='store_true')
parser.set_defaults(plotmap=False)
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

parser.add_argument("--file")
parser.add_argument("--om")
parser.add_argument("--line")
parser.add_argument("--arc")
parser.add_argument("--z")
parser.add_argument("--res")
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
parser.add_argument("--wfits")
parser.add_argument("--off")
parser.add_argument("--rad")
parser.add_argument("--gain")
parser.add_argument("--exp")
parser.add_argument("--epp")
args, leftovers = parser.parse_known_args()

if args.file is not None:
    fn = args.file
    print 'Simulation=', fn
else:
    fn = 'DD0600_lgf' #which simulation to use
    print 'Simulation not specified. Using default', fn, '. Use --file option to specify simulation.'

if args.om is not None:
    Om_ar = [float(ar) for ar in args.om.split(',')]
    print 'Omega=', Om_ar
else:
    Om_ar = [0.5]
    print 'Omega not specified. Using default Omega', Om_ar, '. Use --om option to specify Omega. \
You can supply , separated multiple omega values.'

if args.arc is not None:
    arcsec_per_pix = float(args.arc)
    print 'arcsec_per_pix=', arcsec_per_pix
else:
    arcsec_per_pix = 0.05
    print 'Arcsec per pixel not specified. Using default arcsec_per_pix of', arcsec_per_pix, '. Use --arc option to specify arcsec_per_pix.'

if args.gain is not None:
    gain = float(args.gain)
    print 'Instrumental gain=', gain
else:
    gain = 1.5
    print 'Instrumental gain not specified. Using default gain=', gain, '. Use --gain option to specify gain.'

if args.epp is not None:
    el_per_phot = float(args.epp)
    print 'electrons per photon=', el_per_phot
else:
    el_per_phot = 1.
    print 'el_per_phot not specified. Using default epp=', el_per_phot, '. Use --epp option to specify el_per_phot.'

if args.exp is not None:
    exptime = float(args.exp)
    print 'Exposure time=', exptime, 'seconds'
else:
    exptime = 600 #sec = 10min exposure
    print 'Exposure time not specified. Using default exptime=', exptime, '. Use --exp option to specify exposure time in seconds.'

if args.z is not None:
    z = float(args.z)
    print 'Redshift=', z
else:
    z = 0.13
    print 'Redshift not specified. Using default redshift of', z, '. Use --z option to specify redshift.'

if args.rad is not None:
    rad = float(args.rad)
    print 'Telescope radius chosen=', rad, 'm'
else:
    rad = 1. #metre
    print 'Telescope radius not specified. Using default rad=', rad, 'm. Use --rad option to specify radius in metres.'

dist = calc_dist(z) #distance to object; in kpc
flux_ratio = (rad/(2*dist*3.086e19))**2 #converting emitting luminosity to luminosity seen from earth, 3.08e19 factor to convert kpc to m

if args.res is not None:
    res = float(args.res)
    print 'resoltion forced to be res=', res
else:
    res = round(arcsec_per_pix*np.pi/(3600*180)*dist, 1) #kpc
    print 'Resolution turns out to be res~', res, ' kpc'

if args.map:
    if args.line is not None:
        line = args.line
        print 'line=', line
    else:
        line = 'OIII5007'# #whose emission map to be made
        print 'Line not specified. Using default', line, '. Use --line option to specify line.'

if args.cmin is not None:
    cmin = float(args.cmin)
else:
    cmin = None

if args.cmax is not None:
    cmax = float(args.cmax)
else:
    cmax = None

if args.ppv:
    if args.nhr is not None:
        nhr = int(args.nhr)
    else:
        nhr = 100 # no. of bins used to resolve the range lamda +/- 5sigma around emission lines
    print 'No. of bins used to resolve+/- 5sigma around emission lines=', nhr

    if args.nbin is not None:
        nbin = int(args.nbin)
    else:
        nbin = 1000 #no. of bins used to bin the continuum into (without lines)
    print 'No. of bins used to bin the continuum into (without lines)=', nbin

    if args.vdisp is not None:
        vdisp = float(args.vdisp)
    else:
        vdisp = 15 #km/s vel dispersion to be added to emission lines from MAPPINGS while making PPV
    print 'Vel dispersion to be added to emission lines=', vdisp, 'km/s'

    if args.vdel is not None:
        vdel = float(args.vdel)
    else:
        vdel = 100 #km/s; vel range in which spectral resolution is higher is sig = 5*vdel/c
                    #so wavelength range of +/- sig around central wavelength of line is binned into further nhr bins
    print 'Vel range in which spectral resolution is higher around central wavelength of line=', vdel, 'km/s'

    if args.vres is not None:
        vres = float(args.vres)
    else:
        vres = 30 #km/s instrumental vel resolution to be considered while making PPV
    print 'Instrumental vel resolution to be considered while making PPV=', vres, 'km/s'

    if args.wmin is not None:
        wmin = float(args.wmin)
        print 'Starting wavelength of PPV cube=', wmin, 'A'
    else:
        wmin = None #Angstrom; starting wavelength of PPV cube
        print 'Starting wavelength of PPV cube at beginning of line list'
    
    if args.wmax is not None:
        wmax = float(args.wmax)
        print 'Ending wavelength of PPV cube=', wmax, 'A'
    else:
        wmax = None #Angstrom; ending wavelength of PPV cube
        print 'Ending wavelength of PPV cube at end of line list'
    
    if args.X is not None:
        X = float(args.X)
    else:
        X = int(galsize/res/2) - 1 #p-p values at which point to extract spectrum from the ppv cube
    if args.plotspec: print 'X position at which spectrum to be plotted=', X

    if args.Y is not None:
        Y = float(args.Y)
    else:
        Y = int(galsize/res/2) - 1 #p-p values at which point to extract spectrum from the ppv cube
    if args.plotspec: print 'Y position at which spectrum to be plotted=', Y

    
if not args.keepprev:
    plt.close('all')

if args.addbpt or args.bptpix or args.bptrad:
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

if args.parm is not None:
    parm = [float(ar) for ar in args.parm.split(',')]
else:
    parm = None # set of parameters i.e. telescope properties to be used for smearing cube/map
    if args.smooth: print 'Parameter set for smearing not specified. Using default settings. Use --parm option to specify smearing parameters set.'
        
if args.ker is not None:
    ker = args.ker
else:
    ker = 'moff' # convolution kernel to be used for smearing cube/map
    if args.smooth: print 'Kernel not specified. Will be using default Moffat profile for smoothing'

if args.off is not None:
    off = float(args.off)
else:
    off = 10. #offset used in makenoisy() function
        
if args.wfits is not None:
    wfits = args.wfits
    print 'Will be saving fits file.'
else:
    wfits = None # name of fits file to be written into
        
#-----------------------jobs fetched--------------------------------------------------------------------
for i, Om in enumerate(Om_ar):
    s = ascii.read(getfn(outtag,fn,Om), comment='#', guess=False)
    if args.addbpt: 
        addbpt(s,Om,col_ar[i], saveplot = args.saveplot)
    elif args.bptrad: 
        bpt_vs_radius(s,Om, saveplot = args.saveplot)
    elif args.bptpix: 
        bpt_pixelwise(s, Om, res, saveplot = args.saveplot, smooth=args.smooth, parm = parm, ker = ker, addnoise=args.addnoise, maketheory=args.maketheory)
    elif args.map: 
        map = emissionmap(s, Om, res, line, saveplot = args.saveplot, cmin=cmin, cmax=cmax, off = off, smooth=args.smooth, \
        parm = parm, ker = ker, hide=args.hide, addnoise=args.addnoise, maketheory=args.maketheory)
    elif args.sfr: 
        SFRmap_real, SFRmapQ0 = SFRmaps(s, Om, res, getmap=args.getmap, cmin=cmin, cmax=cmax, off = off, \
        saveplot = args.saveplot, smooth=args.smooth, parm = parm, ker = ker, hide=args.hide, addnoise=args.addnoise, maketheory=args.maketheory)
    elif args.ppv: 
        ppvcube = spec(s, Om, res, col_ar[i], wmin=wmin, wmax=wmax, cmin=cmin, cmax=cmax, off = off, \
        changeunits= args.changeunits, X=X, Y=Y, spec_smear = args.spec_smear, plotspec = args.plotspec, \
        plotintegmap = args.plotmap, savecube=args.savecube, saveplot = args.saveplot, smooth=args.smooth, parm = parm, \
        ker = ker, hide=args.hide, addnoise=args.addnoise, maketheory=args.maketheory)
    else: 
        print 'Wrong choice. Choose from:\n --addbpt, --bptpix, --bptrad, --map, --sfr, --ppv'
if args.saveplot:
    print 'Saved here:', path
if wfits is not None:
    filename = title(fn)[:-2]+'_'+wfits
    if args.smooth: filename += '_smeared_'+ker+'_parm'+args.parm
    if args.addnoise: filename += '_noisy'
    if args.changeunits: filename += '_flambda'
    if 'map' in locals(): write_fits(path+filename, map, fill_val=np.nan)
    if 'ppvcube' in locals(): write_fits(path+filename, ppvcube, fill_val=np.nan)
#-------------------------------------------------------------------------------------------
print('Done in %s minutes' % ((time.time() - start_time)/60))
if not args.hide:
    plt.show(block=False)
