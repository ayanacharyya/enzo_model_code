# -----python code to compute radii of HII regions around young star particles (given as input)--------
# ----------by Ayan, last modified Feb 2019--------------------
# -----  example usage:
# ------ ipython> run get_radii.py --file DD0600_lgf --merge_HII 0.04

import time
import math
import numpy as np
from scipy.interpolate import interp1d
import sys
from scipy.optimize import newton, fmin_powell
import subprocess
import argparse as ap
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
import os
HOME = os.getenv('HOME')
import lookup as lu

# ---------------function for merging HII regions within args.mergeHII kpc distance -------------------------------------------------
def merge_HIIregions(df, args):
    # Columns of table are 'x', 'y', 'z', 'vz', 'age', 'mass', 'gas_P', Q_H0''
    print 'Merging HII regions within '+ str(args.mergeHII*1e3)+ ' pc. May take 10-20 seconds...'
    groupbycol = 'cell_index'
    weightcol = 'Q_H0'
    initial_nh2r = len(df)
    g = int(np.ceil(args.galsize / args.mergeHII))
    xind = ((df['x'] - args.center + args.galsize/2.) / args.mergeHII).astype(np.int) # (df['x(kpc)'][j] - args.center) used to range from (-galsize/2, galsize/2) kpc, which is changed here to (0, galsize) kpc
    yind = ((df['y'] - args.center + args.galsize/2.) / args.mergeHII).astype(np.int)
    df[groupbycol] = xind + yind * g

    keepcols = ['x(kpc)', 'y(kpc)', ]

    if 'Sl.' in df.columns: df.drop(['Sl.'], axis=1, inplace=True)
    weighted_mean = lambda x: np.average(x, weights=df.loc[x.index, weightcol]) # function to weight by mass
    cols_to_sum = [weightcol] + ['mass']
    cols_to_wtmean = df.columns[~df.columns.isin([groupbycol] + cols_to_sum)]
    df = df.groupby([groupbycol], as_index=False).agg(dict({weightcol: {'': sum, 'count': 'count'}}.items() + {'mass': {'': sum}}.items() \
        + {item:{'':weighted_mean} for item in cols_to_wtmean}.items())).drop([groupbycol], axis=1).reset_index(drop=True)
    df.columns = [''.join(x) for x in df.columns.ravel()]
    df.rename(columns={weightcol + 'count':'count'}, inplace=True)
    print 'Merged', initial_nh2r, 'HII regions into', len(df), 'HII regions\n'
    return df

# -----------------function to compute final radius of HII regions--------------------------------------
def compute_radii(paramlist):
    # --------calculating characteristic radius-----------#
    r_ch = (alpha_B * eps ** 2 * f_trap ** 2 * psi ** 2 * paramlist['Q_H0']) / (12 * np.pi * phi * k_B ** 2 * TII ** 2 * c ** 2)
    # --------calculating characteristic time-----------#
    m_SI = paramlist['mass'] * 1.989e30  # converting Msun to kg
    age_SI = paramlist['age'] * 3.1536e13  # converting Myr to sec
    r_0 = (m_SI ** 2 * (3 - k_rho) ** 2 * G / (paramlist['gas_P'] * 8 * np.pi)) ** (1 / 4.)
    rho_0 = (2 / np.pi) ** (1 / 4.) * (paramlist['gas_P'] / G) ** (3 / 4.) / np.sqrt(m_SI * (3 - k_rho))
    t_ch = np.sqrt(4 * np.pi * rho_0 * r_0 ** k_rho * c * r_ch ** (4 - k_rho) / ((3 - k_rho) * f_trap * psi * eps * paramlist['Q_H0']))
    tau = age_SI / t_ch
    # --------calculating radiation pressure radius-----------#
    xII_rad = ((tau ** 2) * (4 - k_rho) / 2) ** (1 / (4 - k_rho))
    # --------calculating gas pressure radius-----------#
    xII_gas = ((7 - 2 * k_rho) ** 2 * tau ** 2 / (4 * (9 - 2 * k_rho))) ** (2 / (7 - 2 * k_rho))
    # --------calculating approximate instantaneous radius-----------#
    xII_apr = (xII_rad ** ((7 - k_rho) / 2) + xII_gas ** ((7 - k_rho) / 2)) ** (2 / (7 - k_rho))
    paramlist['r_inst'] = xII_apr * r_ch
    # --------calculating stall radius-----------#
    Prad = psi * eps * f_trap * paramlist['Q_H0'] / (4 * np.pi * c)
    Pgas = (3 * phi * paramlist['Q_H0'] / (4 * np.pi * alpha_B * (1 + Y / (4 * X)))) * (mu_H * m_H * cII ** 2) ** 2
    r0 = (Pgas / (paramlist['gas_P'] ** 2)) ** (1 / 3.0)
    for index in range(len(paramlist)):
        paramlist.loc[index, 'r_stall'] = 10 ** (newton(Flog, math.log10(r0[index]), args=(Prad[index], Pgas[index], paramlist.loc[index, 'gas_P']), maxiter=100))
    # --------determining minimum of the two-----------#
    paramlist['r'] = paramlist[['r_inst', 'r_stall']].min(axis=1)
    paramlist['nII'] = np.sqrt(3 * phi * paramlist['Q_H0'] / (4 * (paramlist['r'] ** 3) * np.pi * alpha_B * (1 + Y / (4 * X))))
    paramlist['<U>'] = paramlist['Q_H0'] / (4 * np.pi * paramlist['r'] ** 2 * c * paramlist['nII'])
    paramlist['log(P/k)'] = np.log10(paramlist['nII'] * TII)
    paramlist['r'] /= 3.06e16 # to convert distance to pc units
    paramlist['r_inst'] /= 3.06e16 # to convert distance to pc units
    paramlist['r_stall'] /= 3.06e16 # to convert distance to pc units
    return paramlist

# -------------------------To solve the equation:------------------------
# -------------------    P^2r^4 -2aPr^2 -br + a^2 = 0    ----------------
def Flog(x, a, b, c):
    # return (c**2)*(10**(4*x)) - 2*a*c*(10**(2*x)) - b*10**x + a**2
    return a / (c * 10 ** (2 * x)) + np.sqrt(b) / (c * 10 ** (x * 1.5)) - 1

# -----------------function to handle input/outut dataframe of list of parameters------------------------
def get_radii_for_df(paramlist, args):
    start_time = time.time()
    # -----------------------assigning default values to args-----------------------
    if args.file is not None: fn = args.file
    else: fn = 'DD0600_lgf'

    if args.mergeHII is not None: args.mergeHII = float(args.mergeHII) # kpc, within which if two HII regions are they'll be treated as merged
    args.galsize = 30. # kpc
    args.center = 0.5*1310.72022072 # kpc units, from Goldbaum simulations in cell units
    mergeHII_text = '_mergeHII=' + str(args.mergeHII) + 'kpc' if args.mergeHII is not None else ''
    # -----------------------------------------------------------------------------------
    outfilename = HOME+'/models/rad_list/rad_list_newton_' + fn + mergeHII_text

    # ----------------------Reading starburst-------------------------------------------
    if not os.path.exists(outfilename) or args.clobber:
        SB_data = pd.read_table(HOME+'/SB99-v8-02/output/starburst03/starburst03.quanta', delim_whitespace=True,comment='#', skiprows=6, \
                                header=None, names=('age', 'HI/sec', 'HI%ofL', 'HeI/sec', 'HeI%ofL', 'HeII/sec', 'HeII%ofL', 'logQ'))
        SB_data.loc[0, 'age'] = 1e-6 # Myr # force first i.e. minimum age to be 1 yr instead of 0 yr to avoid math error
        interp_func = interp1d(np.log10(SB_data['age']/1e6), SB_data['HI/sec'], kind='cubic')

        nh2r_initial = len(paramlist)
        paramlist = paramlist[['x', 'y', 'z', 'vel_z', 'age', 'mass', 'gas_P']] # only need these columns henceforth
        paramlist['Q_H0'] = paramlist['mass'] * 10 ** (interp_func(np.log10(paramlist['age'])) - 6) # '6' bcz starburst was run for mass of 1M Msun
        paramlist['gas_P'] /= 10 # to convert to N/m^2
        if args.mergeHII is not None: paramlist = merge_HIIregions(paramlist, args)

        # ------------------solving--------------------------------------------------------------
        paramlist = compute_radii(paramlist)
        print 'Using', len(paramlist), 'HII regions of', nh2r_initial

        # ------------------writing dataframe to file--------------------------------------------------------------
        header = 'Units for the following columns: \n\
        x, y, z: kpc \n\
        vel_z: km/s \n\
        age: Myr \n\
        mass: Msun \n\
        gas_P in a cell: N/m^2 \n\
        Q_H0: photons/s \n\
        r_stall: pc \n\
        r_inst: pc \n\
        r: pc \n\
        <U>: volumne averaged, dimensionless \n\
        nII: HII region number density per m^3\n\
        log(P/k): SI units\n'

        np.savetxt(outfilename, [], header=header, comments='#')
        paramlist.to_csv(outfilename, sep='\t', mode='a', index=None)
        print 'Radii list saved at', outfilename
    else:
        print 'Reading from existing file', outfilename
        paramlist = pd.read_table(outfilename, delim_whitespace=True, comment='#')

    print(fn + ' in %s minutes' % ((time.time() - start_time) / 60))
    paramlist = lu.lookup_full_df(paramlist, args)
    return paramlist


# -------------------defining constants-------------------------------------------------------
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

# -------------------------------------------------------------------------------
if __name__ == '__main__':
    # -------------------arguments parsed: MOST args are not used in here, they are for subsequent function calls-----------------------------
    # -------------------arguments parsed-------------------------------------------------------
    parser = ap.ArgumentParser(description="dummy")
    parser.add_argument('--write_file', dest='write_file', action='store_true')
    parser.set_defaults(write_file=False)
    parser.add_argument('--plot_metgrad', dest='plot_metgrad', action='store_true')
    parser.set_defaults(plot_metgrad=False)
    parser.add_argument('--plot_Zinout', dest='plot_Zinout', action='store_true')
    parser.set_defaults(plot_Zinout=False)
    parser.add_argument('--plot_hist', dest='plot_hist', action='store_true')
    parser.set_defaults(plot_hist=False)
    parser.add_argument('--plot_lummap', dest='plot_lummap', action='store_true')
    parser.set_defaults(plot_lummap=False)
    parser.add_argument('--allstars', dest='allstars', action='store_true')
    parser.set_defaults(allstars=False)
    parser.add_argument('--nooutliers', dest='nooutliers', action='store_true')
    parser.set_defaults(nooutliers=False)
    parser.add_argument('--keep', dest='keep', action='store_true')
    parser.set_defaults(keep=False)
    parser.add_argument('--saveplot', dest='saveplot', action='store_true')
    parser.set_defaults(saveplot=False)
    parser.add_argument('--clobber', dest='clobber', action='store_true')
    parser.set_defaults(clobber=False)
    parser.add_argument("--file")
    parser.add_argument("--diag")
    parser.add_argument("--Om")
    parser.add_argument("--logOHgrad")
    parser.add_argument("--fontsize")
    parser.add_argument("--outpath")
    parser.add_argument("--mergeHII")
    args, leftovers = parser.parse_known_args()

    if args.file is not None: fn = args.file
    else: fn = 'DD0600_lgf'
    # ----------------------Reading star particles-------------------------------------------
    paramlist = pd.read_table(HOME+'/models/paramlist/param_list_' + fn, delim_whitespace=True, comment='#')
    paramlist = get_radii_for_df(paramlist, args)

    # --------------------------------------------------------------------------------
    nstalled = sum(paramlist['r'] == paramlist['r_stall'])
    print str(nstalled) + ' HII regions have stalled expansion which is ' + str(nstalled * 100. / len(paramlist)) + ' % of the total.'
