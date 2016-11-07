import time
start_time = time.time()
import numpy as np
import sys
sys.path.append('/Users/acharyya/Mappings/lab')
import rungrid as r
import subprocess
from astropy.io import ascii
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
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
#--------------------------------------------------------
fn=sys.argv[1] #filename for radlist (which simulation output)
Om_ar = np.linspace(0.1,1.,10) #
alpha_B = 3.46e-19 #m^3/s OR 3.46e-13 cc/s, Krumholz Matzner (2009)
c = 3e8 # m/s
ltemp = 4. # assumed 1e4 K temp
#--------------------------------------------------------
#Z = np.linspace()
age_arr = np.linspace(0.,5.,6) #in Myr
lnII = np.linspace(5., 12., 6) #nII in particles/m^3
lU = np.linspace(-4.,-1., 4) #dimensionless
#age_arr = np.linspace(0.,5.,2) #in Myr
#lnII = np.linspace(5., 12., 2) #nII in particles/m^3
#lU = np.linspace(-4.,-1., 3) #dimensionless
#-----------------------------------------------
fin = open('/Users/acharyya/models/rad_list/rad_list_newton_'+fn, 'r')
r_m, Q_H0, age, x, y, fl = [], [], [], [], [], 0
for line in fin.readlines():
    r_m.append(float(line.split()[5])*3.06e16) #pc to SI
    Q_H0.append(float(line.split()[6])) #photons/s
    age.append(float(line.split()[10])) #Myr
    x.append(float(line.split()[1]))
    y.append(float(line.split()[2]))
fin.close()
#-----------------------------------------------    
s = ascii.read('/Users/acharyya/Mappings/lab/totalspec.txt',comment='#',guess=False)
Hb = np.reshape(np.log10(s['HBeta']),(len(age_arr), len(lnII), len(lU)))
OII = np.reshape(np.log10(s['OII3727']),(len(age_arr), len(lnII), len(lU)))
OIII4 = np.reshape(np.log10(s['OIII4363']),(len(age_arr), len(lnII), len(lU)))
OIII5 = np.reshape(np.log10(s['OIII5007']),(len(age_arr), len(lnII), len(lU)))
OI = np.reshape(np.log10(s['OI6300']),(len(age_arr), len(lnII), len(lU)))
Ha = np.reshape(np.log10(s['H6562']),(len(age_arr), len(lnII), len(lU)))
NII = np.reshape(np.log10(s['NII6584']),(len(age_arr), len(lnII), len(lU)))
SII = np.reshape(np.log10(s['SII6730']),(len(age_arr), len(lnII), len(lU)))
#-----------------------------------------------
ifHb = RGI((age_arr, lnII, lU), Hb)
ifOII = RGI((age_arr, lnII, lU), OII)
ifOIII4 = RGI((age_arr, lnII, lU), OIII4)
ifOIII5 = RGI((age_arr, lnII, lU), OIII5)
ifOI = RGI((age_arr, lnII, lU), OI)
ifHa = RGI((age_arr, lnII, lU), Ha)
ifNII = RGI((age_arr, lnII, lU), NII)
ifSII = RGI((age_arr, lnII, lU), SII)
#-----------------------------------------------
for Om in Om_ar:
    n, U, Rs, ri, lpok, ind, coord = [], [], [], [], [], [], []
    for i in range(len(r_m)):
        n.append(calc_n2(Om, r_m[i], Q_H0[i]))
        U.append(calc_U(Om, n[i], Q_H0[i]))
        Rs.append(float(r_m[i]/((1+Om)**(1/3.))))
        ri.append(float(Rs[i]*(Om**(1/3.))))
        lpok.append(float(np.log10(n[i]) + ltemp - 6.))
        ind.append(i)
        coord.append([age[i], np.log10(n[i]), np.log10(U[i])])
    coord = np.array(coord)
    #-----------------------------------------------

    print 'Sanity check for Omega=',Om
    print 'age grid in Myr', age_arr
    print 'required age range', np.min(age), np.max(age)
    if np.min(age) < age_arr[0] or np.max(age) > age_arr[-1]: fl = 1
    print 'Log nII grid', lnII
    print 'required lognII range', np.min(np.log10(n)), np.max(np.log10(n))
    if np.min(np.log10(n)) < lnII[0] or np.max(np.log10(n)) > lnII[-1]: fl = 1
    print 'Log U grid', lU
    print 'required logU range', np.min(np.log10(U)), np.max(np.log10(U))
    if np.min(np.log10(U)) < lU[0] or np.max(np.log10(U)) > lU[-1]: fl = 1
    if fl == 1:
        print 'Required range out of grid range for this Omega value. Redo your grids.\n'
        #sys.exit()
        continue
    else:
        print 'All good for Om=',Om,'. Continuing...\n'
    #-----------------------------------------------

    Hb = np.power(10,ifHb(coord))
    OII = np.power(10,ifOII(coord))
    OIII4 = np.power(10,ifOIII4(coord))
    OIII5 = np.power(10,ifOIII5(coord))
    OI = np.power(10,ifOI(coord))
    Ha = np.power(10,ifHa(coord))
    NII = np.power(10,ifNII(coord))
    SII = np.power(10,ifSII(coord))
    #-----------------------------------------------
    fout = 'emissionlist_'+fn+'_Om'+str(Om)+'.txt'
    head = 'Sl. x   y   age nII <U> logQ0   lpok    Rs(pc)  r_i(pc) r_m(pc)    HBeta   \
    OII3727 OIII4363    OIII5007    OI6300  H6562   NII6584 SII6730'
    np.savetxt(fout, np.transpose([ind, x, y, age, n, U, np.log10(Q_H0), lpok, np.divide(Rs, 3.086e16), \
    np.divide(ri, 3.086e16), np.divide(r_m, 3.086e16), Hb, OII, OIII4, OIII5, OI, Ha, NII, SII]), "%d  \
    %.1F  %.1F  %.0E  %.0E  %.0E  %.3F  %.3F  %.1E  %.1E  %.1E  %.2E  %.2E  %.2E  %.2E  %.2E  %.2E  %.2E  %.2E", \
    header=head, comments='')
    #subprocess.call(['open '+fout+' -a "TextWrangler"'],shell=True)
print('Done in %s minutes' % ((time.time() - start_time)/60))
