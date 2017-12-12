import time
start_time = time.time()
import numpy as np
import sys
sys.path.append('/Users/acharyya/Work/astro/ayan_codes/HIIgrid/')
import rungrid as r
import subprocess
from astropy.io import ascii
from operator import itemgetter
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
import warnings
warnings.filterwarnings("ignore")
#-----------Function to check if float------------------------------
def isfloat(str):
    try: 
        float(str)
    except ValueError: 
        return False
    return True
#-----------------------------------------------
def num(s):
    if s[-1].isdigit():
        return str(format(float(s),'0.2e'))
    else:
        return str(format(float(s[:-1]),'0.2e'))
#--------------------------------------------------------
def calc_n2(Om, r_m, Q_H0):
    return (np.sqrt(3*Q_H0*(1+Om)/(4*np.pi*alpha_B*r_m**3)))
#-----------------------------------------------
def calc_U(Om, nII, Q_H0):
    return r.func(nII, np.log10(Q_H0))*((1+Om)**(4/3.) - (4./3. + Om)*(Om**(1/3.)))
#---------function to paint metallicity gradient-----------------------------------------------
def calc_Z(x,y, logOHcen, logOHgrad):
    res, galsize = 0.02, 26. #kpc
    x *= res #converting x,y from pixels to kpc
    y *= res
    #r = np.sqrt((x-galsize/2)**2 + (y-galsize/2)**2) #calculate galactocentric distance (kpc)
    r = np.sqrt((x-galsize/2 - 2)**2 + (y-galsize/2 - 2)**2) #calculate galactocentric distance (kpc), with the galactic center being at an offset of (2,2)kpc w.r.t center of FoV
    Z = 10**(logOHcen-logOHsun + logOHgrad*r) #assuming central met logOHcen and gradient logOHgrad dex/kpc, logOHsun = 8.77 (Dopita 16) 
    return Z
#-----------------------------------------------
#--------------------------------------------------------
fn=sys.argv[1] #filename for radlist (which simulation output)
Om_ar = [0.5] #np.linspace(0.1,1.,10) #
alpha_B = 3.46e-19 #m^3/s OR 3.46e-13 cc/s, Krumholz Matzner (2009)
c = 3e8 # m/s
ltemp = 4. # assumed 1e4 K temp
logOHsun = 8.77 #Dopita 16
logOHcen = logOHsun #9.2 #(Ho 15)
logOHgrad_arr = [-0.4] #[-0.1,-0.05,-0.025,-0.01] #dex per kpc (Ho 15)
#-----------------------------------------------
for logOHgrad in logOHgrad_arr:
    fin = open('/Users/acharyya/models/rad_list/rad_list_newton_'+fn, 'r')
    r_m, Q_H0, age, x, y, vz, m= [], [], [], [], [], [], []
    for line in fin.readlines():
        r_m.append(float(line.split()[5])*3.06e16) #pc to SI
        Q_H0.append(float(line.split()[6])) #photons/s
        age.append(float(line.split()[10])) #Myr
        x.append(float(line.split()[1])) #cell units
        y.append(float(line.split()[2]))
        vz.append(float(line.split()[11]))
        m.append(float(line.split()[12])) #Msun
    fin.close()
    #------------Reading pre-defined line list-----------------------------
    target = []
    finp = open('/Users/acharyya/Mappings/lab/targetlines.txt','r')
    l = finp.readlines()[3:]
    for lin in l:
        if len(lin.split())>1 and lin[0] != '#':
            target.append([float(lin.split()[0]),lin.split()[1]])
    finp.close()
    target = sorted(target, key=itemgetter(0))
    lines = np.array(target)[:,1]
    #-----------------------------------------------
    ifunc=[]
    path = '/Users/acharyya/models/emissionlist'+'_Zgrad'+str(logOHcen)+','+str(logOHgrad)+r.outtag
    subprocess.call(['mkdir -p '+path],shell=True)

    s = ascii.read('/Users/acharyya/Mappings/lab/totalspec'+r.outtag+'.txt',comment='#',guess=False)
    '''
    #-------Not part of main code; only for testing: From Dopita 2016------------------------------
    print 'outtag =', r.outtag
    logOHsol = 8.77 #log(O/H)+12 value used for Solar metallicity in MAPPINGS-V, Dopita 2016
    log_ratio = np.log10(np.divide(s['NII6584'],(s['SII6730']+s['SII6717']))) + 0.264*np.log10(np.divide(s['NII6584'],s['H6562']))
    print 'log_ratio med, min', np.median(log_ratio), np.min(log_ratio) #
    logOHobj_map = log_ratio #+ 0.45*(log_ratio + 0.3)**5 # + 8.77
    print 'logOHobj_map before conversion med, min', np.median(logOHobj_map), np.min(logOHobj_map) #

    Z_list = 10**(logOHobj_map) #converting to Z (in units of Z_sol) from log(O/H) + 12
    print 'Z_list after conversion med, mean, min',np.median(Z_list), np.mean(Z_list), np.min(Z_list) #

    plt.hist(Z_list, 100, range =(-1,6)) #
    plt.title('Z_map for grids') #

    plt.xlabel('Z/Z_sol') #
    plt.show(block=False) #
    print 'analysis for outtag=', r.outtag #
    print 'This was just a temporary test. Exiting lookup.py without doing any looking up. If required, turn this off at line 69-84.' #
    print '\nZ\tage\tlnII\tlU\tHa\tNII\tSIIa\tSIIb'
    for i in range(len(s)):
        Z_out = 10**(np.log10(np.divide(s['NII6584'][i],(s['SII6730'][i]+s['SII6717'][i]))) + 0.264*np.log10(np.divide(s['NII6584'][i],s['H6562'][i])))
        print s['Z'][i], Z_out, s['age'][i]/1e6, np.log10(s['nII'][i]), np.log10(s['<U>'][i]), s['H6562'][i], s['NII6584'][i], s['SII6717'][i], s['SII6730'][i]
    sys.exit() #
    #-----------------------------------------------
    '''
    for line in lines:
        l = np.reshape(np.log10(s[line]),(len(r.Z_arr), len(r.age_arr), len(r.lognII_arr), len(r.logU_arr)))
        iff = RGI((r.Z_arr, r.age_arr, r.lognII_arr, r.logU_arr), l)
        ifunc.append(iff)
    #-----------------------------------------------
    for Om in Om_ar:
        fout = path+'/emissionlist_'+fn+'_Om'+str(Om)+'.txt'
        Z, n, U, Rs, ri, lpok, ind, coord, fl = [], [], [], [], [], [], [], [], 0
        for i in range(len(r_m)):
            Z.append(calc_Z(x[i],y[i],logOHcen,logOHgrad))
            n.append(calc_n2(Om, r_m[i], Q_H0[i]))
            U.append(calc_U(Om, n[i], Q_H0[i]))
            Rs.append(float(r_m[i]/((1+Om)**(1/3.))))
            ri.append(float(Rs[i]*(Om**(1/3.))))
            lpok.append(float(np.log10(n[i]) + ltemp - 6.))
            ind.append(i)
            coord.append([Z[i], age[i], np.log10(n[i]), np.log10(U[i])])
        coord = np.array(coord)
        print 'lognII range', np.min(coord[:,1]), np.max(coord[:,1]) #
        print 'lpok range', np.min(coord[:,1])-2, np.max(coord[:,1])-2 #
        print 'logU range', np.min(coord[:,2]), np.max(coord[:,2]) #
        #-----------------------------------------------
        print 'Sanity check for Omega=',Om
        print 'age grid in Myr', r.age_arr
        print 'required age range', np.min(age), np.max(age)
        if np.min(age) < r.age_arr[0] or np.max(age) > r.age_arr[-1]: fl = 1
        print 'Log nII grid', r.lognII_arr
        print 'required lognII range', np.min(np.log10(n)), np.max(np.log10(n))
        if np.min(np.log10(n)) < r.lognII_arr[0] or np.max(np.log10(n)) > r.lognII_arr[-1]: fl = 1
        print 'Log U grid', r.logU_arr
        print 'required logU range', np.min(np.log10(U)), np.max(np.log10(U))
        if np.min(np.log10(U)) < r.logU_arr[0] or np.max(np.log10(U)) > r.logU_arr[-1]: fl = 1
        if fl == 1:
            print 'Required range out of grid range for this Omega value. Redo your grids.\n'
            continue
        else:
            print 'All good for Om=',Om,'. Continuing...\n'
        #-----------------------------------------------
        fluxes=[]
        for i, line in enumerate(lines):
            flux = np.power(10,ifunc[i](coord))
            fluxes.append(flux)
        #-----------------------------------------------
        
        #---for testing only: to check final distribution of Z after looking up the grid---
        x, y, Z, fluxes, n = np.array(x), np.array(y), np.array(Z), np.array(fluxes), np.array(n)
        wo = ''
        '''
        x = x[(n >= 10**(5.2-4+6)) & (n <= 10**(6.7-4+6))] #D16 models have 5.2 < lpok < 6.7
        y = y[(n >= 10**(5.2-4+6)) & (n <= 10**(6.7-4+6))] #D16 models have 5.2 < lpok < 6.7
        Z = Z[(n >= 10**(5.2-4+6)) & (n <= 10**(6.7-4+6))] #D16 models have 5.2 < lpok < 6.7
        fluxes = [fluxes[ind][(n >= 10**(5.2-4+6)) & (n <= 10**(6.7-4+6))] for ind in range(np.shape(fluxes)[0])] #D16 models have 5.2 < lpok < 6.7
        print 'For logOHgrad=', logOHgrad, ',Om=', Om, ':', len(x), 'out of', len(n), 'H2Rs meet D16 criteria'
        n = n[(n >= 10**(5.2-4+6)) & (n <= 10**(6.7-4+6))] #D16 models have 5.2 < lpok < 6.7
        wo = '_wo'
        '''
        res, galsize = 0.02, 26. #
        print np.min(x), np.max(x) #
        x *= res
        y *= res
        print np.min(x), np.max(x) #
        dist = np.sqrt((x-galsize/2 -2)**2 + (y-galsize/2 -2)**2) #
        print np.min(dist), np.max(dist) #
        Zout = 10**(np.log10(np.divide(fluxes[41],(fluxes[43]+fluxes[42]))) + 0.264*np.log10(np.divide(fluxes[41],fluxes[40]))) #D16 method
        #Zout = 1.54020 + 1.26602*np.log10(np.divide(fluxes[41],(fluxes[28]+fluxes[29]))) + 0.167977*np.log10(np.divide(fluxes[41],(fluxes[28]+fluxes[29])))**2 #KD02 method
        
        plt.figure()
        plt.plot(np.arange(galsize/2), np.poly1d((logOHgrad, logOHcen))(np.arange(galsize/2)) - logOHsun, c='brown', ls='dotted', label='Target input gradslope='+str('%.4F'%logOHgrad)) #
        
        col ='k'
        plt.scatter(dist,np.log10(Z), c=col, lw=0, alpha=0.5)
        linefit, linecov = np.polyfit(dist, np.log10(Z), 1, cov=True)
        x_arr = np.arange(galsize/2)
        plt.plot(x_arr, np.poly1d(linefit)(x_arr), c=col,label='Fitted input Z slope='+str('%.4F'%linefit[0]))
        
        col = 'b'
        plt.scatter(dist,np.log10(Zout), c=np.log10(n)-6, lw=0, alpha=0.2)
        linefit, linecov = np.polyfit(dist, np.log10(Zout), 1, cov=True)
        x_arr = np.arange(galsize/2)
        plt.plot(x_arr, np.poly1d(linefit)(x_arr), c=col,label='Fitted output D16 Z slope='+str('%.4F'%linefit[0]))

        plt.xlabel('Galactocentric distance (kpc)')
        plt.ylabel('Z/Zsun')
        plt.legend(loc='best', fontsize=10)
        cb = plt.colorbar()
        cb.set_label('density log(cm^-3)')#'Halpha luminosity log(ergs/s)')#     
        plt.title('logOHgrad='+str(logOHgrad)+', Om='+str(Om))
        fig = plt.gcf()
        #fig.savefig('/Users/acharyya/Desktop/bpt/Zgrad_D16_col_density_logOHgrad='+str(logOHgrad)+',Om='+str(Om)+wo+'.eps')
        plt.show(block=False)
        '''
        #-----to plot Zin vs Zout--------
        plt.figure()
        plt.scatter(np.log10(Z), np.log10(Zout), c=fluxes[40], lw=0, alpha=0.5)
        plt.plot(np.log10(Z), np.log10(Z), c='k') #1:1 line
        plt.xlabel('Zin')
        plt.ylabel('Zout')
        cb = plt.colorbar()
        cb.set_label('Halpha luminosity log(ergs/s)')#'dist(kpc)')#'density log(cm^-3)')#
        plt.title('logOHgrad='+str(logOHgrad)+', Om='+str(Om))
        fig = plt.gcf()
        #fig.savefig('/Users/acharyya/Desktop/bpt/Zout_D16_vs_Zin_col_Ha_logOHgrad='+str(logOHgrad)+',Om='+str(Om)+wo+'.eps')
        plt.show(block=False) #
        #------------------
        '''
        continue #
        #-----------------------------------------------
        
        head = 'Sl. x   y   vz  Z(Zsun)  age(MYr)  mass(Msun) nII <U> logQ0   lpok    Rs(pc)  r_i(pc) r_m(pc) '
        for name in lines:
            head += '  '+name
        outarr = np.row_stack((ind, x, y, vz, Z, age, m, n, U, np.log10(Q_H0), lpok, np.divide(Rs, 3.086e16), np.divide(ri, 3.086e16), np.divide(r_m, 3.086e16), fluxes))
        np.savetxt(fout, np.transpose([outarr]), "%d  \
        %.1F  %.1F  %.1F  %0.2F  %.0E  %.2F  %.0E  %.0E  %.3F  %.3F  %.1E  %.1E  %.1E "+" %.2E"*len(lines), \
        header=head, comments='')
        #subprocess.call(['open '+fout+' -a "TextWrangler"'],shell=True)
print('Done in %s minutes' % ((time.time() - start_time)/60))
