# -----python code to compute fluxes of HII regions for given metallicity gradient--------
# ----------by Ayan, last modified Feb 2019--------------------
# -----  example usage:
# ------ ipython> run lookup.py --file DD0600_lgf --Om 0.5 --diag D16 --logOHgrad -0.1 --plot_metgrad --write_file --saveplot --merge_HII 0.04
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
import scipy.optimize as op
import warnings
warnings.filterwarnings("ignore")
import argparse as ap
import os
HOME = os.getenv('HOME')
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# -----------Function to check if float------------------------------
def isfloat(str):
    try:
        float(str)
    except ValueError:
        return False
    return True

# -----------------------------------------------
def num(s):
    if s[-1].isdigit():
        return str(format(float(s), '0.2e'))
    else:
        return str(format(float(s[:-1]), '0.2e'))

# --------------------------------------------------------
def calc_n2(Om, r, Q_H0):
    return (np.sqrt(3 * Q_H0 * (1 + Om) / (4 * np.pi * alpha_B * r ** 3)))

# -----------------------------------------------
def calc_U(Om, nII, Q_H0):
    return r.func(nII, np.log10(Q_H0)) * ((1 + Om) ** (4 / 3.) - (4. / 3. + Om) * (Om ** (1 / 3.)))

# ---------function to paint metallicity gradient-----------------------------------------------
def calc_Z(r, logOHcen, logOHgrad, logOHsun):
    Z = 10 ** (logOHcen - logOHsun + logOHgrad * r)  # assuming central met logOHcen and gradient logOHgrad dex/kpc, logOHsun = 8.77 (Dopita 16)
    return Z

# -------------function to delete a certain H2R from all lists--------------------------
def remove_star(indices, list_of_var):
    new_list_of_var = []
    for list in list_of_var:
        list = np.delete(list, indices, 0)
        new_list_of_var.append(list)
    return new_list_of_var

# -------------function to use KD02 R23 diagnostic for the upper Z branch--------------------------
def poly(x, R, k):
    return np.abs(np.poly1d(k)(x) - np.log10(R))

# --------------function to deal with input/output dataframe to do the whole computation-----------
def lookup_full_df(paramlist, args):
    start_time = time.time()
    # -----------------------assigning default values to args-----------------------
    if args.file is not None: fn = args.file
    else: fn = 'DD0600_lgf'

    if args.diag is not None: diag_arr = [ar for ar in args.diag.split(',')]
    else: diag_arr = ['D16'] #['KD02', 'D16']

    if args.Om is not None:
        if type(args.Om) is str: Om_arr = [float(ar) for ar in args.Om.split(',')]
        elif type(args.Om) is float: Om_arr = [args.Om]
        else: Om_arr = args.Om # already in a list
    else: Om_arr = [0.05, 0.5, 5.0]

    if args.logOHgrad is not None:
        if type(args.logOHgrad) is str:
            logOHgrad_arr = [float(ar) for ar in args.logOHgrad.split(',')]
        elif type(args.logOHgrad) is float:
            logOHgrad_arr = [args.logOHgrad]
        else:
            logOHgrad_arr = args.logOHgrad  # already in a list
    else: logOHgrad_arr = [-0.1,-0.05,-0.025,-0.01]

    if args.fontsize is not None: fs = int(args.fontsize)
    else: fs = 20 # ticklabel fontsize

    if args.outpath is not None: outpath = args.outpath
    else: outpath = HOME+'/Dropbox/papers/enzo_paper/Figs/'

    if args.mergeHII is not None: args.mergeHII = float(args.mergeHII) # kpc, within which if two HII regions are they'll be treated as merged

    args.center = 0.5*1310.72022072 # kpc
    args.galsize = 30. # kpc

    if not args.keep: plt.close('all')
    allstars_text = '_allstars' if args.allstars else ''

    # --------------------------------calculating two new columns-----------------------
    paramlist['distance'] = np.sqrt((paramlist['x'] - args.center) ** 2 + (paramlist['y'] - args.center) ** 2)  # kpc
    paramlist['logQ'] = np.log10(paramlist['Q_H0'])
    print'min, max h2r distance in kpc', np.min(paramlist['distance']), np.max(paramlist['distance']), '\n'  #
    mergeHII_text = '_mergeHII='+str(args.mergeHII)+'kpc' if args.mergeHII is not None else ''

    # ---------to plot the original simulation luminosity map------------------
    if args.plot_lummap:
        # ---figure for age map-----
        fig = plt.figure()
        plt.scatter(x, y, c=paramlist['Q_H0'], s=5, lw=0)
        cb = plt.colorbar()
        cb.set_label('Q_H0', fontsize=fs)
        plt.ylabel('y (kpc)', fontsize=fs)
        plt.xlabel('x (kpc)', fontsize=fs)
        plt.xlim(args.center - args.galsize/2, args.center + args.galsize/2)
        plt.ylim(args.center - args.galsize/2, args.center + args.galsize/2)
        ax = plt.gca()
        ax.set_xticklabels(['%.2F' % (i - args.center) for i in list(ax.get_xticks())], fontsize=fs)
        ax.set_yticklabels(['%.2F' % (i - args.center) for i in list(ax.get_yticks())], fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs)
        xdiff = np.diff(ax.get_xlim())[0]
        ydiff = np.diff(ax.get_ylim())[0]
        ax.text(ax.get_xlim()[-1] - 0.1 * xdiff, ax.get_ylim()[-1] - 0.1 * ydiff, fn, color='k', ha='right', va='center',fontsize=fs)
        if args.saveplot:
            outplotname = outpath + fn + allstars_text+'_QH0_map.eps'
            fig.savefig(outplotname)
            print 'Saved figure at', outplotname

        if args.allstars:
            # -------figure for age CDF------
            factor = 1e52 # just a factor for cleaner plot
            age_turn = 5 # Myr
            Q_H0 = np.array([x for _, x in sorted(zip(paramlist['age'], paramlist['Q_H0']), key=lambda pair: pair[0])]) / factor
            age = np.sort(paramlist['age'])
            Q_sum = np.cumsum(Q_H0)
            Q_tot = Q_sum[-1]
            Q_turn = Q_sum[np.where(age >= age_turn)[0][0]]
            fig = plt.figure()
            plt.plot(age, Q_sum, c='k')
            plt.axhline(Q_tot, c='k', ls='dotted', label='Total ionising luminosity')
            plt.axhline(Q_turn, c='k', ls='dashed', label='%.1F%% of the total at %d Myr'%(Q_turn*100./Q_tot, age_turn))
            plt.axvline(age_turn, c='k', ls='dashed')
            plt.xlabel(r'Age $<$ t Myr', fontsize=fs)
            plt.ylabel(r'Cumulative $Q_{H0}$ (10$^{\mathrm{%d}}$ ergs/s)'%np.log10(factor), fontsize=fs)
            plt.xlim(0,10)
            plt.ylim(0,7)
            plt.legend(loc='lower right', fontsize=fs)
            ax = plt.gca()
            ax.set_xticklabels(ax.get_xticks(), fontsize=fs)
            ax.set_yticklabels(ax.get_yticks(), fontsize=fs)
            if args.saveplot:
                outplotname = outpath + fn + allstars_text+'_QH0_cumulative.eps'
                fig.savefig(outplotname)
                print 'Saved plot at', outplotname

        plt.show(block=False)
    if args.allstars: sys.exit()

    # ------------Reading pre-defined line list-----------------------------
    linelist = pd.read_table(HOME+'/Mappings/lab/targetlines.txt', comment='#', delim_whitespace=True, skiprows=3, names=('wave', 'label', 'wave_vacuum'))
    linelist = linelist.sort_values(by=('wave'))

    # -----------------reading grid files onto RGI-------------------------------------------------------------------
    s = pd.read_table(HOME+'/Mappings/lab/totalspec' + r.outtag + '.txt', comment='#', delim_whitespace=True)
    ifunc = []
    for label in linelist['label']:
        try:
            l = np.reshape(np.log10(s[label]), (len(r.Z_arr), len(r.age_arr), len(r.lognII_arr), len(r.logU_arr)))
            iff = RGI((r.Z_arr, r.age_arr, r.lognII_arr, r.logU_arr), l)
            ifunc.append(iff)
        except KeyError:
            linelist = linelist[~(linelist['label'] == label)].reset_index(drop=True) # discarding label from linelist if it is not present in s[fluxes]
            pass

    # ----------looping over diag_arr, logOHgrad_arr and Om_ar to lookup fluxes-------------------------------------
    for diag in diag_arr:
        if diag == 'R23':
            logOHsun = 8.93  # KD02
            logOHcen = 9.5  # to use upper branch of R23 diag of KD02
        else:
            logOHsun = 8.77  # Dopita 16
            logOHcen = logOHsun  # 9.2 #(Ho 15)

        # ----------looping over logOHgrad_arr to lookup fluxes-------------------------------------
        for logOHgrad in logOHgrad_arr:
            # --------------create output directories---------------------------------
            path = HOME+'/models/emissionlist' + '_Zgrad' + str(logOHcen) + ',' + str(logOHgrad) + r.outtag
            subprocess.call(['mkdir -p ' + outpath], shell=True)

            # -------to plot histogram of derived Z------------------------------
            if args.plot_hist:
                print 'outtag =', r.outtag
                if diag == 'D16':
                    log_ratio = np.log10(np.divide(s['NII6584'],(s['SII6730']+s['SII6717']))) + 0.264*np.log10(np.divide(s['NII6584'],s['H6562']))
                    logOHobj_map = log_ratio + 0.45*(log_ratio + 0.3)**5 # + 8.77
                elif diag == 'KD02':
                    log_ratio = np.log10(np.divide(s['NII6584'], (s['OII3727'] + s['OII3729'])))
                    logOHobj_map = 1.54020 + 1.26602 *log_ratio  + 0.167977 *log_ratio ** 2
                print 'log_ratio med, min', np.median(log_ratio), np.min(log_ratio)  #
                print 'logOHobj_map before conversion med, min', np.median(logOHobj_map), np.min(logOHobj_map) #

                Z_list = 10**(logOHobj_map) #converting to Z (in units of Z_sol) from log(O/H) + 12
                print 'Z_list after conversion med, mean, min',np.median(Z_list), np.mean(Z_list), np.min(Z_list) #

                plt.figure()
                plt.hist(Z_list, 100, range =(-1,6)) #
                plt.title('Z_map for grids') #

                plt.xlabel('Z/Z_sol') #
                if args.saveplot:
                    outplotname = outpath + 'Z_map for grids.eps'
                    fig.savefig(outplotname)
                    print 'Saved plot at', outplotname
                plt.show(block=False) #

            # ----------looping over Om_arr to lookup fluxes-------------------------------------
            for Om in Om_arr:
                fout = path + '/emissionlist_' + fn + '_Om' + str(Om) + mergeHII_text + '.txt'
                if not os.path.exists(fout) or args.clobber:
                    paramlist['r'] *= 3.06e16 # convert to m from pc
                    paramlist['Z'] = calc_Z(paramlist['distance'], logOHcen, logOHgrad, logOHsun)
                    paramlist['nII'] = calc_n2(Om, paramlist['r'], paramlist['Q_H0'])
                    paramlist['<U>'] = calc_U(Om, paramlist['nII'], paramlist['Q_H0'])
                    paramlist['r_Strom'] = paramlist['r'] / ((1 + Om) ** (1 / 3.))
                    paramlist['r_i'] = paramlist['r_Strom']* (Om ** (1 / 3.))
                    paramlist['log(P/k)'] = np.log10(paramlist['nII']) + ltemp - 6.
                    paramlist['r'] /= 3.06e16 # convert to pc from m
                    paramlist['r_Strom'] /= 3.06e16 # convert to pc from m
                    coord = np.vstack([paramlist['Z'], paramlist['age'], np.log10(paramlist['nII']), np.log10(paramlist['<U>'])]).transpose()

                    # ---------compute fluxes by interpolating--------------------------------------
                    for ind in range(len(linelist)): paramlist[linelist.loc[ind, 'label']] = np.power(10, ifunc[ind](coord))

                    # ---to discard outliers that are beyond the parameters used to calibrate the diagnostic---
                    if args.nooutliers and diag == 'D16':
                        paramlist = paramlist[paramlist['nII'].between(10**(5.2-4+6), 10**(6.7-4+6))] #D16 models have 5.2 < lpok < 6.7
                        print 'Discarding outliers: For logOHgrad=', logOHgrad, ',Om=', Om, ':', len(x), 'out of', len(n), 'H2Rs meet D16 criteria'
                        wo = '_no_outlier'
                    else:
                        wo = ''
                    # ------------------writing dataframe to file--------------------------------------------------------------
                    if args.write_file:
                        header = 'Units for the following columns: \n\
                        x, y, z: kpc \n\
                        vel_z: km/s \n\
                        age: Myr \n\
                        mass: Msun \n\
                        gas_P in a cell: N/m^2 \n\
                        Q_H0: photons/s \n\
                        logQ: log(photons/s) \n\
                        r_stall: pc \n\
                        r_inst: pc \n\
                        r: pc \n\
                        <U>: volumne averaged, dimensionless \n\
                        nII: HII region number density per m^3\n\
                        log(P/k): SI units \n\
                        Z: metallicity in Zsun units \n\
                        fluxes: ergs/s/cm^2'

                        np.savetxt(fout, [], header=header, comments='#')
                        paramlist.to_csv(fout, sep='\t', mode='a', index=None)
                        print 'Parameter list saved at', fout
                else:
                    print 'Reading existing file from', fout
                    paramlist = pd.read_table(fout, delim_whitespace=True, comment='#')

                # --------to calculate Zout based on different diagnostics, for plots---------------------------------------
                if diag == 'D16':
                    Zout = 10 ** (np.log10(np.divide(paramlist['NII6584'], (paramlist['SII6717'] + paramlist['SII6730']))) + 0.264 * np.log10(np.divide(paramlist['NII6584'], paramlist['H6562'])))  # D16 method
                elif diag == 'KD02':
                    Zout = 1.54020 + 1.26602 * np.log10(np.divide(paramlist['NII6584'], (paramlist['OII3727'] + paramlist['OII3729']))) + 0.167977 * np.log10(np.divide(paramlist['NII6584'], (paramlist['OII3727'] + paramlist['OII3729']))) ** 2  # KD02 method
                elif diag == 'R23':
                    Zout, ratio_list = [], []
                    for k in range(len(paramlist)):
                        ratio = (paramlist.loc[k, 'OII3727'] + paramlist.loc[k, 'OII3729'] + paramlist.loc[k, 'OIII4363'])/paramlist.loc[k, 'Hbeta']
                        this_Zout = 10**(op.fminbound(poly, 7., 9., args=(ratio, [-0.996645, 32.6686, -401.868, 2199.09, -4516.46])) - logOHsun)  # k parameters from Table 3 of KD02 for q=8e7
                        Zout.append(this_Zout)
                        ratio_list.append(ratio)
                    Zout = np.array(Zout)
                    ratio_list = np.array(ratio_list)

                # ---to check final distribution of Z after looking up the grid---
                if args.plot_metgrad:
                    fig = plt.figure()
                    plt.plot(np.arange(args.galsize/2), np.poly1d((logOHgrad, logOHcen))(np.arange(args.galsize/2)) - logOHsun, c='brown',
                             ls='dotted', label='Target gradient=' + str('%.3F' % logOHgrad))  #

                    plt.scatter(paramlist['distance'], np.log10(paramlist['Z']), c='k', lw=0, alpha=0.5)
                    plt.scatter(paramlist['distance'], np.log10(Zout), c=np.log10(paramlist['nII']) - 6, lw=0, alpha=0.2)
                    linefit = np.polyfit(paramlist['distance'], np.log10(Zout), 1, cov=False)
                    x_arr = np.arange(args.galsize/2)
                    plt.plot(x_arr, np.poly1d(linefit)(x_arr), c='b', label='Fitted gradient=' + str('%.4F' % linefit[0])+', offset='+\
                                                                            str('%.1F'%((linefit[0]-logOHgrad)*100/logOHgrad))+'%')

                    plt.xlabel('Galactocentric distance (kpc)', fontsize=fs)
                    plt.ylabel(r'$\log{(Z/Z_{\bigodot})}$', fontsize=fs)
                    plt.legend(loc='lower left', fontsize=fs)
                    ax = plt.gca()
                    ax.tick_params(axis='both', labelsize=fs)
                    cb = plt.colorbar()
                    cb.set_label('Density log(cm^-3)', fontsize=fs)  # 'H6562 luminosity log(ergs/s)')#
                    if not diag == 'R23': plt.ylim(ylim_dict[logOHgrad],0.2)
                    plt.xlim(-1,12) # kpc
                    if args.saveplot:
                        outplotname = outpath+fn+'_Zgrad_'+diag+'_col_density_logOHgrad='+str(logOHgrad)+',Om='+str(Om)+wo+'.eps'
                        fig.savefig(outplotname)
                        print 'Saved plot at', outplotname
                    plt.show(block=False)

                # -----to plot Zin vs Zout--------
                if args.plot_Zinout:
                    fig = plt.figure()
                    plt.scatter(np.log10(paramlist['Z']), np.log10(Zout), c=paramlist['H6562'].values, lw=0, alpha=0.5)
                    plt.plot(np.log10(paramlist['Z']), np.log10(paramlist['Z']), c='k') #1:1 line
                    plt.xlabel('Zin', fontsize=fs)
                    plt.ylabel('Zout', fontsize=fs)
                    ax = plt.gca()
                    ax.tick_params(axis='both', labelsize=fs)
                    cb = plt.colorbar()
                    cb.set_label('H6562 luminosity log(ergs/s)', fontsize=fs)#'dist(kpc)')#'density log(cm^-3)')#
                    xdiff = np.diff(ax.get_xlim())[0]
                    ydiff = np.diff(ax.get_ylim())[0]
                    ax.text(ax.get_xlim()[-1]-0.1*xdiff, ax.get_ylim()[-1]-0.1*ydiff, 'Input gradient = %.3F'%logOHgrad, color='k', ha='right', va='center', fontsize=fs)
                    #plt.title('logOHgrad='+str(logOHgrad)+', Om='+str(Om))
                    if args.saveplot:
                        outplotname = outpath+fn+'_Zout_'+diag+'_vs_Zin_col_Ha_logOHgrad='+str(logOHgrad)+',Om='+str(Om)+wo+'.eps'
                        fig.savefig(outplotname)
                        print 'Saved plot at', outplotname
                    plt.show(block=False)

    if not args.write_file: print 'Text files not saved.'
    print('Done in %s minutes' % ((time.time() - start_time) / 60))

    return paramlist

# ---------------defining constants-----------------------------------------
ylim_dict = {-0.1:-1.2, -0.05:-0.9, -0.025:-0.5, -0.01:-0.02}
alpha_B = 3.46e-19  # m^3/s OR 3.46e-13 cc/s, Krumholz Matzner (2009)
c = 3e8  # m/s
ltemp = 4.  # assumed 1e4 K temp

# -----------------------------------------------------------------------------
if __name__ == '__main__':
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

    allstars_text = '_allstars' if args.allstars else ''
    mergeHII_text = '_mergeHII='+str(args.mergeHII)+'kpc' if args.mergeHII is not None else ''
    if args.file is not None: fn = args.file
    else: fn = 'DD0600_lgf'

    # --------------reading in the parameter list-----------------------------------------------------------
    paramlist = pd.read_table(HOME+'/models/rad_list/rad_list_newton' + allstars_text + '_' + fn + mergeHII_text, delim_whitespace=True, comment='#')
    '''---------------------------------------------
    -------------------------TESTING ZONE STARTS----------------------------
    --------------------------------------------------------'''
    '''------------------------------------------------------
    -----------------------------------------TESTING ZONE ENDS-----------------------------
    -----------------------------------------------------------'''
    paramlist = lookup_full_df(paramlist, args)
