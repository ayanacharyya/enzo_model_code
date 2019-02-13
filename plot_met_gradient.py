# python utility routine to plot trends in apparent metallicity gradient and scatter with varying properties telescope
# by Ayan

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
import sys
import subprocess
import re
import os
import scipy.special as sp
from scipy.optimize import curve_fit
from matplotlib import ticker
import collections

HOME = os.getenv('HOME') + '/'
sys.path.append(HOME + 'Work/astro/ayan_codes/mageproject/')
import ayan.splot_util as u
# -----------------------------------------------------------------
def fitfunc(xarr, slope, knee, offset):
    return float(slope) * (sp.erf(xarr * np.sqrt(np.pi)/ (float(knee)*2.)) - 1) + float(offset)
    #return float(slope) * (sp.erf(xarr / float(knee)) - 1) - float(offset)
# -----------------------------------------------------------------
logOHsun = 8.77
for_paper = 0 # 1 or 0
for_ppt = 1 # 1 or 0
inverse_xaxis = 1 #
fit_every = 0 #
fit_together = 1 #
do_one_plot = 1 #
savetab = 0 #
plot_intercept = 0
all_in_one_plot = 1
fs = 17 if for_paper or for_ppt else 8 # plot label fontsize
inpath = HOME + 'Desktop/bpt_contsub_contu_rms/'
diag_arr = ['D16', 'KD02']
Om_arr = [0.5, 5.0, 0.05]
simulation_choice_arr = ['DD0600','DD0600_lgf']
gf_dict={'DD0600':0.2, 'DD0600_lgf':0.1, 'DD0300_hgf':0.4}
ifu = collections.namedtuple('ifu', 'res snr name color')
surveys = [ifu(25, 5, 'TYPHOON', 'gold'), ifu(4, 5, 'SAMI', 'saddlebrown'), ifu(5, 5, 'CALIFA', 'lightgreen'), ifu(3, 5, 'MaNGA', 'skyblue'), \
           ifu(12, 5, 'VENGA', 'gray')]
hide = 0

# exptime_choice = None #choose from [600, 1200] #[150000,300000, 1500000, 3000000]
if do_one_plot:
    diag_arr = diag_arr[:1]
    Om_arr = Om_arr[:1]
    simulation_choice_arr = simulation_choice_arr[:1]
ycol1, ylab1 = 'slope', r'$\nabla_r \log{Z}$' #'metallicity gradient'  #
ycol2, ylab2 = 'intercept', r'log($Z_0$/$Z_\odot$)'
res_arcsec_choice = None  # choose from [0.1,0.5,0.8,1.0,1.5,2.0,2.5,3.0]*5 or None
res_phys_choice = None  # choose from [] or None
vres_choice = 30  # choose from [10,20,30,50,70,90] or None
# fixed_SNR_choice = 5 #choose from [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10] or None
size_choice = 5  # or None
power_choice = 4.7  # or None
logOHcen_choice = logOHsun  # or None
scale_length = 4.  # kpc
binto = 2 # previously 4
z = 0.04 # previously 0.013
ylim2 = (-2, 0.1)
base_cmap = plt.cm.prism  # choose any color map
a = 0.3  # alpha value for plotting errorbars
if inverse_xaxis: xmax, xmin = None, 1.5
else: xmax, xmin = 0.6, -0.02
snr_mode = 'fixed_noise'  # 'fixed_SNR' #choose between fixed_SNR and snr_cut
outtab = pd.DataFrame(columns=['Diagnostic', r'$\Omega$', 'f$_g$', 'Input gradient', 'a', 'b', 'c'])

for diag in diag_arr:
    for Om in Om_arr:
        for simulation_choice in simulation_choice_arr:
            filename_root = 'met_grad_Om='+str(Om)+'_'+diag+'_exp_scaled'
            merge_text = '_branches_merged' #if diag == 'KD02' else ''

            noconvolution_filename = filename_root + '_'+snr_mode + merge_text + '.txt'
            nonoise_noconvolution_filename = noconvolution_filename.replace('_'+snr_mode, '')
            filename = filename_root + '_'+snr_mode + '_binto'+str(binto) + merge_text + '.txt'  # noise level fixed while generating PPV cube
            nonoise_filename = filename.replace('_'+snr_mode, '')

            if os.path.exists(inpath + nonoise_filename):
                nonoise_exists = True
                print 'Foundno noise file ' + nonoise_filename + '. Over plotting no noise case.'
            else:
                nonoise_exists = False
                print 'Nonoise file ' + nonoise_filename + ' does not exist. So not plotting no noise case.'

            outpath = inpath
            # -----------------------------------------------------------------
            if res_phys_choice is None:
                xcol = 'res_phys'
                xlab = 'Number of beams per scale radius' if inverse_xaxis else r'$\Delta r$ physical/scale length'
            elif res_arcsec_choice is None:
                xcol = 'res_arcsec'
                xlab = r'$\Delta r$ (")'
            elif vres_choice is None:
                xcol = 'vres'
                xlab = r'$\Delta v$ (km/s)'
            elif size_choice is None:
                xcol = 'size'
                xlab = 'PSF size(pixels)'
            elif power_choice is None:
                xcol = 'power'
                xlab = 'PSF exponent'
            elif logOHcen_choice is None:
                xcol = 'logOHcen'
                xlab = r'True log($Z_0$/$Z_\odot$)'
            elif logOHgrad_choice is None:
                xcol = 'logOHgrad'
                xlab = r'True $\nabla_r$logZ (dex/kpc)'
            elif snr_cut_choice is None:
                xcol = 'snr_cut'
                xlab = r'SNR cut'
            '''
            elif fixed_SNR_choice is None:
                xcol ='fixed_SNR'
                xlab = 'fixed SNR'
            elif exptime_choice is None:
                xcol = 'exptime'
                xlab = 'exposure time(s)'
            '''

            # --------------selection criteria to make plots-------------------------------------------
            full_table = pd.read_table(inpath + filename, delim_whitespace=True, comment="#")
            if os.path.exists(inpath + noconvolution_filename):
                print 'Found noconvolution file ' + noconvolution_filename + '. Including no convolution case.'
                noconvolution_fulltable = pd.read_table(inpath + noconvolution_filename, delim_whitespace=True, comment="#")
                full_table = pd.concat([noconvolution_fulltable, full_table])
            else:
                print 'Noconvolution file ' + noconvolution_filename + ' does not exist. So not plotting no convolution case.'

            full_table = full_table[full_table['res_arcsec'] > 0.1]  # to NOT plot no convolution case
            fixed_noise_arr = [pd.unique(full_table[snr_mode])[i] for i in [1,2,4,5,8]] # choose a few SNR cases to plot
            # numbers here are array indices based on [1.,3.,5.,8.,10.,30.,300.,1e6,1e300]
            master_table = full_table[full_table.simulation == simulation_choice]
            master_table = master_table.drop('simulation', axis=1)
            master_table['snr_measured'] = master_table['snr_measured'].astype(np.float64)

            #---to reduce info in the plots, for use in paper------
            if for_paper:
                master_table = master_table[master_table[snr_mode].isin(fixed_noise_arr)]  ##to forcibly exclude/include specific "fixed_noise" cases
                master_table = master_table[master_table['snr_cut'].isin([0, 5])]  ##to forcibly exclude/include specific "snr" cases
                master_table = master_table[ master_table['logOHgrad'].isin([-0.1, -0.05])]  ##to forcibly exclude/include specific "Zgrad" cases
            elif for_ppt:
                master_table = master_table[master_table[snr_mode].isin(fixed_noise_arr)]  ##to forcibly exclude/include specific "fixed_noise" cases
                master_table = master_table[master_table['snr_cut'].isin([0, 5])]  ##to forcibly exclude/include specific "snr" cases
                master_table = master_table[ master_table['logOHgrad'].isin([-0.1])]  ##to forcibly exclude/include specific "Zgrad" cases

            if nonoise_exists:
                nonoise_full_table = pd.read_table(inpath + nonoise_filename, delim_whitespace=True, comment="#")
                if os.path.exists(inpath + nonoise_noconvolution_filename):
                    print 'Found nonoise_noconvolution file ' + nonoise_noconvolution_filename + '. Including no noise, no convolution case.'
                    nonoise_noconvolution_fulltable = pd.read_table(inpath + nonoise_noconvolution_filename, delim_whitespace=True, comment="#")
                    nonoise_full_table = pd.concat([nonoise_noconvolution_fulltable, nonoise_full_table])
                else:
                    print 'Nonoise_noconvolution file ' + nonoise_noconvolution_filename + ' does not exist. So not plotting no noise, no convolution case.'

                nonoise_master_table = nonoise_full_table[nonoise_full_table.simulation == simulation_choice]
                nonoise_master_table = nonoise_master_table.drop('simulation', axis=1)

            snr_cut_choice_arr = pd.unique(master_table['snr_cut'])  # [0,3,5,8] # or None
            logOHgrad_choice_arr = pd.unique(master_table['logOHgrad'])  # [-0.1,-0.05,-0.025,-0.01]
            if xcol == 'res_phys':
                inv = -1. if inverse_xaxis else 1.
                master_table[xcol] = (master_table[xcol]/scale_length)**inv  # inv=-1 to get in units of beams per scale radius
                if nonoise_exists:  nonoise_master_table[xcol] = (nonoise_master_table[xcol]/scale_length)**inv
            if xmin is None: xmin = np.min(master_table[xcol].values) * 0.5
            if xmax is None: xmax = np.max(master_table[xcol].values) * 1.1
            # -----------------------------------------------------------------
            if not all_in_one_plot:
                outpath += filename[:-4] + '/'
                subprocess.call(['mkdir -p ' + outpath], shell=True)
            t = simulation_choice + '_Om=' + str(Om) + '_' + diag + '_' + ycol1
            if not all_in_one_plot and plot_intercept: t += ',' + ycol2
            t += '_vs_' + xcol

            if res_arcsec_choice is not None:
                master_table = master_table[master_table.res_arcsec == res_arcsec_choice]
                if nonoise_exists: nonoise_master_table = nonoise_master_table[nonoise_master_table.res_arcsec == res_arcsec_choice]
                t += '_arc=' + str(res_arcsec_choice)
            if res_phys_choice is not None and res_phys_choice > 0:
                master_table = master_table[master_table.res_phys == res_phys_choice]
                if nonoise_exists: nonoise_master_table = nonoise_master_table[nonoise_master_table.res_phys == res_phys_choice]
                t += '_res=' + str(res_phys_choice)
            if vres_choice is not None:
                master_table = master_table[master_table.vres == vres_choice]
                if nonoise_exists: nonoise_master_table = nonoise_master_table[nonoise_master_table.vres == vres_choice]
                t += '_vres=' + str(vres_choice)
            if size_choice is not None:
                master_table = master_table[master_table['size'] == size_choice]
                if nonoise_exists: nonoise_master_table = nonoise_master_table[nonoise_master_table['size'] == size_choice]
            if power_choice is not None:
                master_table = master_table[master_table.power == power_choice]
                if nonoise_exists: nonoise_master_table = nonoise_master_table[nonoise_master_table.power == power_choice]
            if logOHcen_choice is not None:
                master_table = master_table[master_table.logOHcen == logOHcen_choice]
                if nonoise_exists: nonoise_master_table = nonoise_master_table[nonoise_master_table.logOHcen == logOHcen_choice]
                t += '_lOHcen=' + str(logOHcen_choice)
            if '_fixed_SNR' in filename: t += '_fixed_SNR'
            if '_fixed_noise' in filename: t += '_fixed_noise'
            if 'binto' in filename: t += '_binto' + re.findall('binto(\d+)', filename)[0]
            if 'fitupto' in filename: t += '_fitupto' + re.findall('fitupto(\d+)', filename)[0]

            nrow = len(logOHgrad_choice_arr)
            ncol = len(snr_cut_choice_arr)
            if all_in_one_plot:
                fig = plt.figure(figsize=(16, 6)) if for_ppt else plt.figure(figsize=(16, 10))
                fig.subplots_adjust(hspace=0.02, wspace=0.02, top=0.9, bottom=0.1, left=0.1, right=0.98)
                #fig.text(0.5, 0.95, t, ha='center')

            for (row, logOHgrad_choice) in enumerate(logOHgrad_choice_arr):
                master_table2 = master_table[master_table.logOHgrad == logOHgrad_choice]
                if nonoise_exists: nonoise_master_table2 = nonoise_master_table[nonoise_master_table.logOHgrad == logOHgrad_choice]
                if all_in_one_plot:

                    table1_dummy = master_table2[master_table2['snr_cut']==0].groupby(['res_arcsec','res_phys','vres','power','size','logOHcen','logOHgrad',snr_mode]).mean().reset_index().drop('realisation', axis=1)
                    y_dummy = (table1_dummy[ycol1].values - logOHgrad_choice)*100./logOHgrad_choice
                    ylim1_l, ylim1_u = np.min(y_dummy) - 20, np.max(y_dummy) + 20
                    if ylim1_l < -500: ylim1_l = -500
                    elif ylim1_l > -20: ylim1_l = -ylim1_u*0.2
                    if ylim1_u > 500: ylim1_u = 500
                    elif ylim1_u < 20: ylim1_u = -ylim1_l*0.2
                    ylim1 = (ylim1_l, ylim1_u)
                    '''
                    if snr_mode == 'fixed_noise':
                        if logOHgrad_choice == -0.1:
                            ylim1 = (-100, 10)
                        elif logOHgrad_choice == -0.05:
                            ylim1 = (-100, 10)
                        elif logOHgrad_choice == -0.025:
                            ylim1 = (-100, 100)
                        elif logOHgrad_choice == -0.01:
                            ylim1 = (-200, 200)
                        else:
                            ylim1 = None
                    else:
                        ylim1 = (-50, 30)
                    '''
                if not all_in_one_plot: title2 = t + '_lOHgrad=' + str(logOHgrad_choice)

                if nonoise_exists:
                    nonoise_y_arr = (nonoise_master_table2[ycol1].values - nonoise_master_table2['logOHgrad'].values) * 100. / \
                                    nonoise_master_table2['logOHgrad'].values
                    nonoise_x_arr = nonoise_master_table2[xcol].values
                    nonoise_y_arr = np.array([x for (y, x) in sorted(zip(nonoise_x_arr, nonoise_y_arr), key=lambda pair: pair[0])])
                    nonoise_x_arr = np.sort(nonoise_x_arr)

                for (col, snr_cut_choice) in enumerate(snr_cut_choice_arr):
                    master_table3 = master_table2[master_table2.snr_cut == snr_cut_choice]
                    if for_paper or for_ppt:
                        col_ar = ['blue', 'darkgreen', 'red', 'black', 'orange', 'pink', 'cyan']
                    else:
                        col_ar = base_cmap(np.linspace(0.1, 1, len(np.unique(master_table3[snr_mode]))))
                    if not all_in_one_plot: title = title2 + '_snrcut=' + str(snr_cut_choice)
                    # --------------framing plot title and initialising 2 panel plot-----------------------------

                    if all_in_one_plot:
                        ax1 = fig.add_subplot(nrow, ncol, row * ncol + col + 1)
                        ax2 = None  # not plotting intercept yet, for all-in-one plot
                    else:
                        if plot_intercept:
                            fig = plt.figure(figsize=(8, 6))
                            ax1 = fig.add_subplot(211)
                            ax2 = fig.add_subplot(212, sharex=ax1)
                            ax2.axhline(0., c='k', linestyle='--')
                        else:
                            fig = plt.figure(figsize=(10, 4))
                            ax1 = fig.add_subplot(111)

                            table1_dummy = master_table3.groupby(
                                ['res_arcsec', 'res_phys', 'vres', 'power', 'size', 'logOHcen', 'logOHgrad',
                                 snr_mode]).mean().reset_index().drop('realisation', axis=1)
                            y_dummy = (table1_dummy[ycol1].values - logOHgrad_choice) * 100. / logOHgrad_choice
                            ylim1_l, ylim1_u = np.min(y_dummy) - 20, np.max(y_dummy) + 20
                            ylim1 = (ylim1_l, ylim1_u)

                        fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.15, left=0.1, right=0.98)
                        if not for_paper: fig.text(0.5, 0.92, title, ha='center', fontsize=fs)
                    ax1.axhline(0., c='k', linestyle='--')
                    if nonoise_exists: plt.plot(nonoise_x_arr, nonoise_y_arr, linestyle='dotted', c='black', label='No noise')
                    # -------plotting multiple fixed_SNRs together-----------------------------

                    for (ind, fsnr) in enumerate(np.unique(master_table3[snr_mode])):
                        table = master_table3[master_table3[snr_mode] == fsnr]
                        table = table.drop(snr_mode, axis=1)
                        # -------grouping multiple realisations----------------------------------------------------------
                        table = table.drop_duplicates(subset=['res_arcsec', 'res_phys', 'vres', 'power', 'size', 'logOHcen', 'logOHgrad', \
                                                      'realisation'], keep='last')
                        table2 = table.groupby(
                            ['res_arcsec', 'res_phys', 'vres', 'power', 'size', 'logOHcen', 'logOHgrad']).mean().reset_index().drop(
                            'realisation', axis=1)
                        table3 = table.groupby(
                            ['res_arcsec', 'res_phys', 'vres', 'power', 'size', 'logOHcen', 'logOHgrad']).std().reset_index().drop(
                            'realisation', axis=1)
                        # -----------gradient plot: individual realisations------------------------------------------------------

                        if xcol == 'logOHcen': xcol = xcol - logOHsun
                        '''
                        y_arr = (table[ycol1].values - table['logOHgrad'].values)*100./table['logOHgrad'].values
                        yerr = (table[ycol1+'_u'].values)*100./table['logOHgrad'].values
                        ax1.scatter(table[xcol].values,y_arr, linewidth=0, c=col_ar[ind])
                        ax1.errorbar(table[xcol].values,y_arr,yerr=yerr, linestyle='None', c=col_ar[ind], alpha=a)    
                        '''
                        # ----------gradient plot: median of all realisations-------------------------------------------------------
                        y_arr = (table2[ycol1].values - table2['logOHgrad'].values) * 100. / table2['logOHgrad'].values
                        yerr = (table3[ycol1].values) * 100. / table3['logOHgrad'].values

                        #if not fsnr >1e100:
                        xoffset = 2.5e-3*(xmax-xmin)*(len(np.unique(master_table3[snr_mode]))-1-ind) # to artifically offset point slightly on x axis, to make errorbars clearer
                        #label_text = 'Output SNR = ' + str('%.0F' %(np.mean(table2['snr_measured'].values))) if (for_ppt or for_paper) else 'SNR inp=' + str(fsnr)
                        label_text = 'Input SNR = %.0F'%(fsnr*1.4) if fsnr < 1e10 else r'Input SNR = $\infty$'
                        ax1.scatter(table2[xcol].values + xoffset, y_arr, c=col_ar[ind], linewidth=0, s=50, label=label_text)
                        ax1.errorbar(table2[xcol].values + xoffset, y_arr, yerr=yerr, linestyle='None', c=col_ar[ind], alpha=a,lw=3)
                        '''
                        else:
                            label_text = 'Output SNR = ' + str('%.1E' % int(np.mean(table2['snr_measured'].values))) if (for_ppt or for_paper) else 'SNR inp=' + str(fsnr)
                            plt.plot(table2[xcol].values, y_arr, linestyle='dashdot', c='brown', label=label_text)
                        '''
                        # ----for fitting the points----------
                        if fit_every:
                            popt, pcov = curve_fit(fitfunc, table2[xcol].values, y_arr, p0=[80, 4, -10], sigma=yerr + 1e-5)
                            print 'fsnr=', fsnr, 'fit parm: offset, knee, slope=', popt
                            xarr = np.arange(0.1, 30, 0.01)
                            ax1.plot(xarr, fitfunc(xarr, *popt), c=col_ar[ind], alpha=0.2)
                        elif fit_together and col == ncol-1:
                            xarr_accum = np.hstack([xarr_accum, table2[xcol].values]) if ind else table2[xcol].values
                            y_arr_accum = np.hstack([y_arr_accum, y_arr]) if ind else y_arr
                            yerr_accum = np.hstack([yerr_accum, yerr]) if ind else yerr
                            if ind == len(np.unique(master_table3[snr_mode])) - 1:
                                popt, pcov = curve_fit(fitfunc, np.array(xarr_accum), np.array(y_arr_accum), p0=[80, 4, -10])#, sigma=np.array(yerr_accum)+1e-5)
                                print 'for all points, fit parm: offset, knee, slope=', popt
                                xarr = np.arange(0.1,30,0.01)
                                ax1.plot(xarr, fitfunc(xarr, *popt), c='k', alpha=0.5, linestyle='dotted')
                                outtab.loc[len(outtab)] = [diag, Om, gf_dict[simulation_choice], logOHgrad_choice] + [r'%.1F' % (item) for item in popt]
                                #ax1.text(ax1.get_xlim()[-1] * 0.8, -popt[0] - 15, 'Fit: offset=-%.2F, knee=%.2F'%(popt[0], popt[1]), color='k',fontsize=fs,ha='right')
                        # ----to plot contemporary IFU surveys---------
                        if col == ncol-1:
                            for thissurvey in surveys:
                                ax1.axvline(thissurvey.res, ls='dashed', color=thissurvey.color, alpha=0.3)
                                ax1.text(thissurvey.res+0.1, -60, thissurvey.name, color=thissurvey.color, fontsize=14, alpha=0.3, rotation=90, ha='left', va='bottom')
                                if thissurvey.name == 'TYPHOON': ax1.arrow(thissurvey.res, -20, 2, 0.1, color=thissurvey.color, \
                         head_width=2, head_length=0.5, length_includes_head=False)

                        # ------------------------------------
                        if not all_in_one_plot and plot_intercept:
                            # -----------intercept plot: individual realisations-----------------------------------
                            '''
                            y_arr = (table[ycol2].values  + logOHsun - table['logOHcen'].values)*100./(table['logOHcen'].values)
                            yerr = (table[ycol2+'_u'].values)*100./(table['logOHcen'].values)
                            ax2.scatter(table[xcol].values,y_arr, linewidth=0, c=col_ar[ind])
                            ax2.errorbar(table[xcol].values,y_arr,yerr=yerr, linestyle='None',c=col_ar[ind], alpha=a)
                            '''
                            # ----------intercept plot: median of all realisations-------------------------------------
                            y_arr = (table2[ycol2].values + logOHsun - table2['logOHcen'].values) * 100. / (
                                table2['logOHcen'].values)
                            yerr = (table3[ycol2].values) * 100. / (table3['logOHcen'].values)
                            if not fsnr > 1e100:
                                ax2.scatter(table2[xcol].values, y_arr, c=col_ar[ind], linewidth=0)
                                ax2.errorbar(table2[xcol].values, y_arr, yerr=yerr, linestyle='None', c=col_ar[ind], alpha=a)
                            else:
                                plt.plot(table2[xcol].values, y_arr, linestyle='dashdot', c='brown', label='SNR = ' + str(fsnr))

                    if (for_paper and col == ncol-1 and row == nrow-1) or (for_ppt and col == 0 and row == 0) or not all_in_one_plot:  # only for the very first subplot
                        ax1.legend(loc="lower right" if all_in_one_plot else "best", fontsize=fs if all_in_one_plot else '8')
                    if inverse_xaxis: ax1.set_xscale('log')

                    ax1.set_ylim(ylim1)
                    if inverse_xaxis:
                        ax1.set_xticks([1, 4, 6, 8, 9], minor=True)
                        ax1.set_xticks([2, 3, 5, 7, 10, 20])
                        ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                    else:
                        ax1.set_xticklabels(ax1.get_xticks(), fontsize=fs)
                    plt.xlim(xmin, xmax)
                    if row == 0:
                        '''
                        ax4 = ax1.twiny()
                        if inverse_xaxis: ax4.set_xscale('log')
                        ax4.set_xlim(ax1.get_xlim())
                        ax4.set_xticks(ax1.get_xticks())
                        ax4.set_xticklabels(['%.1F'%(4./item/(np.pi / (3600 * 180) * ((z/0.013) * 55.714285714285715 * 1e3))) for item in ax4.get_xticks()])
                        if not all_in_one_plot: ax4.set_xlabel('PSF size (in arcseconds)', fontsize=fs)
                        '''
                        if col == 0:
                            ax1.text(ax1.get_xlim()[0] * 1.02, np.diff(ax1.get_ylim()) * 0.08, 'Steeper than true gradient', color='k', fontsize=fs,
                             ha='left')
                            ax1.text(ax1.get_xlim()[0] * 1.02, -np.diff(ax1.get_ylim()) * 0.08, 'Shallower than true gradient', color='k', fontsize=fs,
                             ha='left')

                    if all_in_one_plot:
                        if row == 0: fig.text((0.5+col)*0.9/ncol + 0.08, 0.94, 'SNR cut=' + str(snr_cut_choice),  ha='center', va='center', fontsize=fs)
                        if not row == nrow - 1: ax1.get_xaxis().set_ticklabels([])
                        if col == 0:
                            ax1.set_ylabel(r'('+ylab1+r'$)_{\rm true}$'+' = ' + str(logOHgrad_choice), fontsize=fs)
                            #ax1.text(0.06, 5., r'('+ylab1+r'$)_{\rm true}$'+' = ' + str(logOHgrad_choice)+' dex/kpc', va='top', ha='left', fontsize=fs)
                            '''
                            ax1.set_yticks(np.linspace(-40,10,6))
                            ax1.set_yticklabels(ax1.get_yticks(), fontsize=fs)
                            '''
                        else:
                            ax1.get_yaxis().set_ticklabels([])
                        print 'Done logOHgrad_choice=', logOHgrad_choice, 'snr_cut_choice=', snr_cut_choice
                        if col == ncol-1: print '\n'
                    else:
                        ax1.set_xlabel(xlab, fontsize=fs)
                        ax1.set_ylabel(r'% Offset in ' + ylab1, fontsize=fs)
                        if plot_intercept:
                            ax2.set_ylabel(r'% Offset in ' + ylab2, fontsize=fs)
                            try:
                                ax2.set_ylim(ylim2)
                            except:
                                pass
                        fig.savefig(outpath + title + '.eps')
                        print 'Saved file', outpath + title + '.eps'

            if all_in_one_plot:
                fig.text(0.5, 0.02, xlab, ha='center', fontsize=fs)
                fig.text(0.5, 0.97, 'PSF size (in arcseconds)',  ha='center', va='center', fontsize=fs)
                fig.text(0.03, 0.5, r'% Offset in ' + ylab1, va='center', rotation='vertical', fontsize=fs)
                fig.savefig(outpath + t + '.eps')
                print 'Saved file', outpath + t + '.eps'
if fit_together and savetab:
    outtab.rename(columns={'Input gradient': r'('+ylab1+r'$)_{\rm true}$'}, inplace=True)
    for col in outtab.columns:
        outtab[col] = outtab[col].astype(np.str)
        outtab[col] = outtab[col].str.replace('-', r'$-$')
    outtab = outtab.replace('NAN', '-')
    outtabfile = HOME+'Dropbox/papers/enzo_paper/Tables/fitted_parameters.tex'
    outtab.to_latex(outtabfile, index=None, escape=False)
    '''
    # --adding additional material to table--
    u.insert_line_in_file(r'\multicolumn{7}{c}{SNR cut-off = '+str(snr_cut_choice)+' }' + '\\\ \n', 2, outtabfile)
    u.insert_line_in_file('\hline \n', 3, outtabfile)
    '''
    print 'Saved file', outtabfile
if hide:
    plt.close('all')
else:
    plt.show(block=False)
print 'Finished!'
