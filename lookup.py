import time
start_time = time.time()
import numpy as np
import sys
sys.path.append('/Users/acharyya/Mappings/lab')
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
r_m, Q_H0, age, x, y, vz, m= [], [], [], [], [], [], []
for line in fin.readlines():
    r_m.append(float(line.split()[5])*3.06e16) #pc to SI
    Q_H0.append(float(line.split()[6])) #photons/s
    age.append(float(line.split()[10])) #Myr
    x.append(float(line.split()[1]))
    y.append(float(line.split()[2]))
    vz.append(float(line.split()[11]))
    m.append(float(line.split()[12])) #Msun
fin.close()
#------------Reading pre-defined line list-----------------------------
target = []
finp = open('/Users/acharyya/Mappings/lab/targetlines.txt','r')
l = finp.readlines()[3:]
for lin in l:
    if len(lin.split())>1:
        target.append([float(lin.split()[0]),lin.split()[1]])
finp.close()
target = sorted(target, key=itemgetter(0))
lines = np.array(target)[:,1]
#-----------------------------------------------
ifunc=[]
outtag = '_logT4'
path = '/Users/acharyya/models/emissionlist'+outtag
subprocess.call(['mkdir -p '+path],shell=True)

s = ascii.read('/Users/acharyya/Mappings/lab/totalspec'+outtag+'.txt',comment='#',guess=False)
for line in lines:
    l = np.reshape(np.log10(s[line]),(len(age_arr), len(lnII), len(lU)))
    iff = RGI((age_arr, lnII, lU), l)
    ifunc.append(iff)
#-----------------------------------------------
for Om in Om_ar:
    n, U, Rs, ri, lpok, ind, coord, fl = [], [], [], [], [], [], [], 0
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
        continue
    else:
        print 'All good for Om=',Om,'. Continuing...\n'
    #-----------------------------------------------
    fluxes=[]
    for i, line in enumerate(lines):
        flux = np.power(10,ifunc[i](coord)) #*m*1e-6 #flux normalised to mass of each cluster as MAPPINGS was run for 1M Msun clusters
        fluxes.append(flux)
    #-----------------------------------------------
    fout = path+'/emissionlist_'+fn+'_Om'+str(Om)+'.txt'
    head = 'Sl. x   y   vz  age(MYr)  mass(Msun) nII <U> logQ0   lpok    Rs(pc)  r_i(pc) r_m(pc) '
    for name in lines:
        head += '  '+name
    outarr = np.row_stack((ind, x, y, vz, age, m, n, U, np.log10(Q_H0), lpok, np.divide(Rs, 3.086e16), np.divide(ri, 3.086e16), np.divide(r_m, 3.086e16), fluxes))
    np.savetxt(fout, np.transpose([outarr]), "%d  \
    %.1F  %.1F  %.1F  %.0E  %.2F  %.0E  %.0E  %.3F  %.3F  %.1E  %.1E  %.1E "+" %.2E"*len(lines), \
    header=head, comments='')
    #subprocess.call(['open '+fout+' -a "TextWrangler"'],shell=True)
print('Done in %s minutes' % ((time.time() - start_time)/60))
