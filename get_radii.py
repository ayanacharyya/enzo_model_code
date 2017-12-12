#!/usr/bin/env python

import time

start_time2 = time.time()
import math
import numpy as np
from scipy.interpolate import interp1d
import sys
from scipy.optimize import newton, fmin_powell

# import multiprocessing as mp
global count


def func(i):
    # --------calculating characteristic radius-----------#
    r_ch = (alpha_B * eps ** 2 * f_trap ** 2 * psi ** 2 * Q_H0[i]) / (12 * np.pi * phi * k_B ** 2 * TII ** 2 * c ** 2)
    # --------calculating characteristic time-----------#
    m_SI = m[i] * 1.989e30  # converting Msun to kg
    age_SI = age[i] * 3.1536e13  # converting Myr to sec
    r_0 = (m_SI ** 2 * (3 - k_rho) ** 2 * G / (Pamb[i] * 8 * np.pi)) ** (1 / 4.)
    rho_0 = (2 / np.pi) ** (1 / 4.) * (Pamb[i] / G) ** (3 / 4.) / np.sqrt(m_SI * (3 - k_rho))
    t_ch = np.sqrt(
        4 * np.pi * rho_0 * r_0 ** k_rho * c * r_ch ** (4 - k_rho) / ((3 - k_rho) * f_trap * psi * eps * Q_H0[i]))
    tau = age_SI / t_ch
    # --------calculating radiation pressure radius-----------#
    xII_rad = ((tau ** 2) * (4 - k_rho) / 2) ** (1 / (4 - k_rho))
    # --------calculating gas pressure radius-----------#
    xII_gas = ((7 - 2 * k_rho) ** 2 * tau ** 2 / (4 * (9 - 2 * k_rho))) ** (2 / (7 - 2 * k_rho))
    # --------calculating approximate instantaneous radius-----------#
    xII_apr = (xII_rad ** ((7 - k_rho) / 2) + xII_gas ** ((7 - k_rho) / 2)) ** (2 / (7 - k_rho))
    rII = xII_apr * r_ch
    # --------calculating stall radius-----------#
    Prad = psi * eps * f_trap * Q_H0[i] / (4 * np.pi * c)
    Pgas = (3 * phi * Q_H0[i] / (4 * np.pi * alpha_B * (1 + Y / (4 * X)))) * (mu_H * m_H * cII ** 2) ** 2
    r0 = (Pgas / (Pamb[i] ** 2)) ** (1 / 3.0)
    r_stall = 10 ** (newton(Flog, math.log10(r0), args=(Prad, Pgas, Pamb[i]), maxiter=100))
    # --------determining minimum of the two-----------#
    r = min(rII, r_stall)
    nII = np.sqrt(3 * phi * Q_H0[i] / (4 * (r ** 3) * np.pi * alpha_B * (1 + Y / (4 * X))))
    IP = Q_H0[i] / (4 * np.pi * r ** 2 * c * nII)
    P_over_k = np.log10(nII * TII)
    s = str(i) + ' ' + str(x[i]) + ' ' + str(y[i]) + ' ' + str(float(r_stall / 3.06e16)) + ' ' + str(
        float(rII / 3.06e16)) + ' ' \
        + str(float(r / 3.06e16)) + ' ' + str(Q_H0[i]) + ' ' + str(nII) + ' ' + str(IP) + ' ' + str(
        P_over_k) + ' ' + str(float(age[i])) + ' ' \
        + str(float(vz[i])) + '  ' + str(float(m[i])) + '\n'  # now prints mass as well
    f.write(s)
    # --------checking if the HII region is a stalled one-----------#
    global stalled, merged
    if r_stall < rII:
        stalled += 1
    if r / 3.06e16 > 20.:
        merged += 1
    print i
    # print i, m_SI, Pamb[i], rho_0, r_0, tau, Q_H0[i]/1e49, r_ch/3.06e16, rII/3.06e16, r_stall/3.06e16


# -------------------------To solve the equation:------------------------
# -------------------    P^2r^4 -2aPr^2 -br + a^2 = 0    ----------------
def Flog(x, a, b, c):
    # return (c**2)*(10**(4*x)) - 2*a*c*(10**(2*x)) - b*10**x + a**2
    return a / (c * 10 ** (2 * x)) + np.sqrt(b) / (c * 10 ** (x * 1.5)) - 1


# -----------------------------------------------------------------
if __name__ == '__main__':
    k_rho = 1.
    ng = 1500
    res = 0.02  # kpc
    size = 1300  # kpc
    mu_H = 1.33
    m_H = 1.67e-27  # kg
    cII = 9.74e3  # m/s
    phi = 0.73
    Y = 0.23
    X = 0.75
    psi = 3.2
    eps = 2.176e-18  # Joules or 13.6 eV
    c = 3e8  # m/s
    f_trap = 2  # Verdolini et. al.
    alpha_B = 3.46e-19  # m^3/s OR 3.46e-13 cc/s, Krumholz Matzner (2009)
    k_B = 1.38e-23  # m^2kg/s^2/K
    TII = 7000.  # K
    G = 6.67e-11  # Nm^2/kg^2
    # -----------------------------------------------------------------
    start = 0  # 22504
    fn = sys.argv[1]
    loga = []
    logl = []
    Q_H0 = []
    Pamb = []
    x = []
    y = []
    age = []
    m = []
    vz = []
    global stalled, merged
    stalled = 0
    merged = 0
    # ----------------------Reading starburst-------------------------------------------
    lum = open('/Users/acharyya/SB99-v8-02/output/starburst03/starburst03.quanta',
               'r')  ##previously done with different SB output
    tail = lum.readlines()[7:]
    for lines in tail:
        loga.append(math.log10(float(lines.split()[0]) / 10 ** 6))  # in Myr
        logl.append(float(lines.split()[1]))
    f = interp1d(loga, logl, kind='cubic')
    lum.close()
    # ----------------------Reading star particles-------------------------------------------
    list = open('/Users/acharyya/models/paramlist/param_list_' + fn, 'r')
    for line in list:
        if float(line.split()[5]) >= 0.1:
            x.append(int(np.floor((float(line.split()[1]) - 0.5) * size / res)) + ng / 2)
            y.append(int(np.floor((float(line.split()[2]) - 0.5) * size / res)) + ng / 2)
            Pamb.append(float(line.split()[8]) / 10)  # N/m^2
            m.append(float(line.split()[4]))  # Msun
            age.append(float(line.split()[5]))  # in Myr
            Q_H0.append(float(line.split()[4]) * 10 ** (
                    f(math.log10(float(line.split()[5]))) - 6))  # '6' bcz starburst was run for mass of 1M Msun
            vz.append(float(line.split()[6]))
    list.close()
    # ------------------solving--------------------------------------------------------------
    f = open('/Users/acharyya/models/rad_list/rad_list_newton_' + fn, 'w')
    for i in range(start, len(x)):
        func(i)
    f.close()
    # --------------------------------------------------------------------------------
    print 'Radii list of ' + fn + ' saved.'
    print str(stalled) + ' HII regions have stalled expansion which is ' + str(
        stalled * 100. / len(x)) + ' % of the total.'
    print str(merged) + ' HII regions grown beyond own cell which is ' + str(
        merged * 100. / len(x)) + ' % of the total.'
    print(fn + ' in %s minutes' % ((time.time() - start_time2) / 60))
'''
#-------------------------Junkyard----------------------------------------
from sympy import *
a = np.zeros(np.shape(x)[0])
b = np.zeros(np.shape(x)[0])
#bl=np.zeros(np.shape(m)[0])

    #plt.scatter(m,a/(m**2)+np.sqrt(b)/(m**1.5),s = 2, edgecolor=None)
    plt.scatter(m,F(m),s = 2, edgecolor=None)
    plt.scatter(m,bl,s=2, color='forestgreen',edgecolor=None)
    #plt.ylim(0,1e-11)
    plt.show()
    #print i, P[i], Q_H0[i], r/3.086e16
    #bl = np.linspace(P[i],P[i],np.shape(m)[0])
        if(np.shape(r)[0] == 4):
    q = (b/(P[i]**2))**(1/3.0)
    r = np.array(newton_krylov(F, [q,q,q,q]))
    #r = np.array(solve(F(z), z))[0:2]    
    for j in range(len(r)):
        if not iscomplex(r[j]):
            print i, r[j]/3e16 ###
#-----------------------------------------------------------------
def Writer(q):
    with open('/Users/acharyya/models/rad_list/rad_list_'+fn, 'a') as f:
        while True: 
            s = q.get()
            if s == 'stop':
                return
            f.write(s)
            f.flush()
    #---------------------Parallelisation crap--------------------------------------------
    m = mp.Manager()
    q = m.Queue()
    p = mp.Pool(mp.cpu_count()+1)
    wp = p.apply_async(Writer, (q,))
    for i in range(start,len(x)):
        p.apply_async(func, (i,q))
    #if (fl ==1): q.put('stop')
    p.close()
    p.join()
    ##-----------------------------------------------------------------
'''
