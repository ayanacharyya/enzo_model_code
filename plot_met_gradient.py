#python utility routine to plot trends in apparent metallicity gradient and scatter with varying properties telescope
#by Ayan

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
HOME = os.getenv('HOME')+'/'
#-----------------------------------------------------------------
logOHsun = 8.77
inpath = HOME+'Desktop/bpt_contsub_contu_rms/'
#filename = 'met_grad_log_paint_fixed_SNR.txt' #SNR fixed while generating PPV cube
#filename = 'met_grad_log_paint_exp_scaled_fixed_noise.txt' #noise level fixed while generating PPV cube
#filename = 'met_grad_log_paint_exp_scaled_fixed_noise_fitupto2.txt' #noise level fixed while generating PPV cube
filename = 'met_grad_log_paint_exp_scaled_fixed_noise_binto4.txt' #noise level fixed while generating PPV cube
#filename = 'met_grad_log_paint_exp_scaled_fixed_noise_binto4_fitupto2.txt' #noise level fixed while generating PPV cube
nonoise_filename = filename.replace('_fixed_noise', '')
highsnr_filename = filename.replace('_fixed_noise', '_fixed_noise_300')
if os.path.exists(inpath+nonoise_filename):
    nonoise_exists = True
    print 'Found no noise file '+nonoise_filename+'. Over plotting no noise case.'
else:
    nonoise_exists = False
    print 'No noise file '+nonoise_filename+' does not exist. So not plotting no noise case.'

if os.path.exists(inpath+highsnr_filename):
    highsnr_exists = True
    print 'Found high SNR file '+highsnr_filename+'. Over plotting high SNR case.'
else:
    highsnr_exists = False
    print 'High SNR file '+highsnr_filename+' does not exist. So not plotting high SNR case.'

outpath = inpath
#exptime_choice = None #choose from [600, 1200] #[150000,300000, 1500000, 3000000]
ycol1, ylab1 = 'slope', r'$\nabla_r$logZ'
ycol2, ylab2 = 'intercept', r'log($Z_0$/$Z_\odot$)'
simulation_choice = 'DD0600_lgf' #choose from ['DD0600_lgf','DD0600']
res_arcsec_choice = None #choose from [0.1,0.5,0.8,1.0,1.5,2.0,2.5,3.0]*5 or None
res_phys_choice = None #choose from [] or None
vres_choice = 30 #choose from [10,20,30,50,70,90] or None
#fixed_SNR_choice = 5 #choose from [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10] or None
size_choice = 5 #or None
power_choice = 4.7 #or None
logOHcen_choice = logOHsun #or None
scale_length = 4. #kpc
ylim2 = (-2,0.1)
base_cmap = plt.cm.rainbow #choose any color map
snr_mode = 'fixed_noise' #'fixed_SNR' #choose between fixed_SNR and snr_cut
a = 0.3 #alpha value for plotting errorbars
xmin, xmax = 0.05,0.37
plot_intercept = False
all_in_one_plot = True
hide = False
#-----------------------------------------------------------------
if res_phys_choice is None:
    xcol = 'res_phys'
    xlab = r'$\Delta r$ physical/scale length'
elif res_arcsec_choice is None:
    xcol = 'res_arcsec'
    xlab = r'$\Delta r$ (")'
elif vres_choice is None:
    xcol = 'vres'
    xlab = r'$\Delta v$ (km/s)'
elif size_choice is None:
    xcol ='size'
    xlab = 'PSF size(pixels)'
elif power_choice is None:
    xcol ='power'
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

#--------------selection criteria to make plots-------------------------------------------
full_table = pd.read_table(inpath+filename, delim_whitespace=True, comment="#")
master_table = full_table[full_table.simulation == simulation_choice]
master_table = master_table.drop('simulation', axis=1)
master_table['snr_measured'] = master_table['snr_measured'].astype(np.float64)
#master_table = master_table[master_table['fixed_noise'].isin([1,3,8])] ##to forcibly exclude/include specific "fixed_noise" cases
if nonoise_exists:
    nonoise_full_table = pd.read_table(inpath+nonoise_filename, delim_whitespace=True, comment="#")
    nonoise_master_table = nonoise_full_table[nonoise_full_table.simulation == simulation_choice]
    nonoise_master_table = nonoise_master_table.drop('simulation', axis=1)

if highsnr_exists:
    highsnr_full_table = pd.read_table(inpath+highsnr_filename, delim_whitespace=True, comment="#")
    highsnr_master_table = highsnr_full_table[highsnr_full_table.simulation == simulation_choice]
    highsnr_master_table = highsnr_master_table.drop('simulation', axis=1)

snr_cut_choice_arr = pd.unique(master_table['snr_cut'])#[0,3,5,8] # or None
logOHgrad_choice_arr = pd.unique(master_table['logOHgrad']) #[-0.1,-0.05,-0.025,-0.01]
if xcol == 'res_phys':
    master_table[xcol] /= scale_length #dividing by scale length in kpc
    if nonoise_exists:  nonoise_master_table[xcol] /= scale_length
    if highsnr_exists:  highsnr_master_table[xcol] /= scale_length
if xmin is None: xmin = np.min(master_table[xcol].values)*0.5
if xmax is None: xmax = np.max(master_table[xcol].values)*1.1
#-----------------------------------------------------------------
if not all_in_one_plot:
    outpath += filename[:-4]+'/'
    subprocess.call(['mkdir -p '+outpath],shell=True)
t = simulation_choice+'-Z-'+ycol1
if not all_in_one_plot and plot_intercept: t += ','+ycol2
t += '_vs_'+xcol

if res_arcsec_choice is not None:
    master_table = master_table[master_table.res_arcsec == res_arcsec_choice]
    if nonoise_exists: nonoise_master_table = nonoise_master_table[nonoise_master_table.res_arcsec == res_arcsec_choice]
    if highsnr_exists: highsnr_master_table = highsnr_master_table[highsnr_master_table.res_arcsec == res_arcsec_choice]
    t += '_arc='+str(res_arcsec_choice)
if res_phys_choice is not None and res_phys_choice > 0:
    master_table = master_table[master_table.res_phys == res_phys_choice]
    if nonoise_exists: nonoise_master_table = nonoise_master_table[nonoise_master_table.res_phys == res_phys_choice]
    if highsnr_exists: highsnr_master_table = highsnr_master_table[highsnr_master_table.res_phys == res_phys_choice]
    t += '_res='+str(res_phys_choice)
if vres_choice is not None:
    master_table = master_table[master_table.vres == vres_choice]
    if nonoise_exists: nonoise_master_table = nonoise_master_table[nonoise_master_table.vres == vres_choice]
    if highsnr_exists: highsnr_master_table = highsnr_master_table[highsnr_master_table.vres == vres_choice]
    t += '_vres='+str(vres_choice)
if size_choice is not None:
    master_table = master_table[master_table['size'] == size_choice]
    if nonoise_exists: nonoise_master_table = nonoise_master_table[nonoise_master_table['size'] == size_choice]
    if highsnr_exists: highsnr_master_table = highsnr_master_table[highsnr_master_table['size'] == size_choice]
if power_choice is not None:
    master_table = master_table[master_table.power == power_choice]
    if nonoise_exists: nonoise_master_table = nonoise_master_table[nonoise_master_table.power == power_choice]
    if highsnr_exists: highsnr_master_table = highsnr_master_table[highsnr_master_table.power == power_choice]
if logOHcen_choice is not None:
    master_table = master_table[master_table.logOHcen == logOHcen_choice]
    if nonoise_exists: nonoise_master_table = nonoise_master_table[nonoise_master_table.logOHcen == logOHcen_choice]
    if highsnr_exists: highsnr_master_table = highsnr_master_table[highsnr_master_table.logOHcen == logOHcen_choice]
    t += '_lOHcen='+str(logOHcen_choice)

if '_fixed_SNR' in filename: t += '_fixed_SNR'
if '_fixed_noise' in filename: t += '_fixed_noise'
if 'binto' in filename: t += '_binto'+re.findall('binto(\d+)',filename)[0]
if 'fitupto' in filename: t += '_fitupto'+re.findall('fitupto(\d+)',filename)[0]

nrow = len(logOHgrad_choice_arr)
ncol = len(snr_cut_choice_arr)
if all_in_one_plot:
    fig = plt.figure(figsize=(16,10))
    fig.subplots_adjust(hspace=0.02, wspace=0.02, top=0.9, bottom=0.06, left=0.1, right=0.98)
    fig.text(0.5, 0.95, t, ha='center')

for (row, logOHgrad_choice) in enumerate(logOHgrad_choice_arr):
    master_table2 = master_table[master_table.logOHgrad == logOHgrad_choice]
    if nonoise_exists: nonoise_master_table2 = nonoise_master_table[nonoise_master_table.logOHgrad == logOHgrad_choice]
    if highsnr_exists: highsnr_master_table2 = highsnr_master_table[highsnr_master_table.logOHgrad == logOHgrad_choice]
    if all_in_one_plot:
        '''
        table1_dummy = master_table2.groupby(['res_arcsec','res_phys','vres','power','size','logOHcen','logOHgrad','snr_cut',snr_mode]).mean().reset_index().drop('realisation', axis=1)
        y_dummy = (table1_dummy[ycol1].values - logOHgrad_choice)*100./logOHgrad_choice
        ylim1_l, ylim1_u = np.min(y_dummy) - 20, np.max(y_dummy) + 20
        if ylim1_l < -500: ylim1_l = -500
        elif ylim1_l > -20: ylim1_l = -ylim1_u*0.2
        if ylim1_u > 500: ylim1_u = 500
        elif ylim1_u < 20: ylim1_u = -ylim1_l*0.2
        ylim1 = (ylim1_l, ylim1_u)
        '''
        if snr_mode == 'fixed_noise':
            if logOHgrad_choice == -0.1: ylim1 = (-50,10)     
            elif logOHgrad_choice == -0.05: ylim1 = (-50,50)     
            elif logOHgrad_choice == -0.025: ylim1 = (-100,100)     
            elif logOHgrad_choice == -0.01: ylim1 = (-200,200)
            else: ylim = None
        else:
            ylim = (-50,30)
    
    if not all_in_one_plot: title2 = t + '_lOHgrad='+str(logOHgrad_choice) 
    
    if nonoise_exists:
        nonoise_y_arr = (nonoise_master_table2[ycol1].values - nonoise_master_table2['logOHgrad'].values)*100./nonoise_master_table2['logOHgrad'].values        
        nonoise_x_arr = nonoise_master_table2[xcol].values
        nonoise_y_arr = np.array([x for (y,x) in sorted(zip(nonoise_x_arr,nonoise_y_arr), key=lambda pair: pair[0])])
        nonoise_x_arr = np.sort(nonoise_x_arr)
    
    if highsnr_exists:
        highsnr_y_arr = (highsnr_master_table2[ycol1].values - highsnr_master_table2['logOHgrad'].values)*100./highsnr_master_table2['logOHgrad'].values        
        highsnr_x_arr = highsnr_master_table2[xcol].values
        highsnr_y_arr = np.array([x for (y,x) in sorted(zip(highsnr_x_arr,highsnr_y_arr), key=lambda pair: pair[0])])
        highsnr_x_arr = np.sort(highsnr_x_arr)

    for (col, snr_cut_choice) in enumerate(snr_cut_choice_arr):
        master_table3 = master_table2[master_table2.snr_cut == snr_cut_choice]
        if not all_in_one_plot: title = title2 + '_snrcut='+str(snr_cut_choice)
        #--------------framing plot title and initialising 2 panel plot---------------------------------------------------
        
        if all_in_one_plot:
            ax1 = fig.add_subplot(nrow, ncol, row*ncol+col+1)
            ax2= None #not plotting intercept yet, for all-in-one plot
        else:
            if plot_intercept:
                fig = plt.figure(figsize=(8,6))
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212, sharex=ax1)
                ax2.axhline(0., c='k', linestyle='--')
            else:
                fig = plt.figure(figsize=(10,3))
                ax1 = fig.add_subplot(111)
                
                table1_dummy = master_table3.groupby(['res_arcsec','res_phys','vres','power','size','logOHcen','logOHgrad',snr_mode]).mean().reset_index().drop('realisation', axis=1)
                y_dummy = (table1_dummy[ycol1].values - logOHgrad_choice)*100./logOHgrad_choice
                ylim1_l, ylim1_u = np.min(y_dummy) - 20, np.max(y_dummy) + 20
                ylim1 = (ylim1_l, ylim1_u)
                
            fig.subplots_adjust(hspace=0.7, top=0.8, bottom=0.15, left=0.08, right=0.98)
            fig.text(0.5, 0.92, title, ha='center')
        ax1.axhline(0., c='k', linestyle='--')
        if nonoise_exists: plt.plot(nonoise_x_arr, nonoise_y_arr, linestyle='dotted', c='black', label='No noise')
        if highsnr_exists: plt.plot(highsnr_x_arr, highsnr_y_arr, linestyle='dashdot', c='brown', label='Very high SNR')
        #-------plotting multiple fixed_SNRs together-----------------------------
        col_ar = base_cmap(np.linspace(0.1, 1, len(np.unique(master_table3[snr_mode]))))
        for (ind,fsnr) in enumerate(np.unique(master_table3[snr_mode])):
            table = master_table3[master_table3[snr_mode]==fsnr]
            table = table.drop(snr_mode, axis=1)
            #-------grouping multiple realisations----------------------------------------------------------
            table2 = table.groupby(['res_arcsec','res_phys','vres','power','size','logOHcen','logOHgrad']).mean().reset_index().drop('realisation', axis=1)
            table3 = table.groupby(['res_arcsec','res_phys','vres','power','size','logOHcen','logOHgrad']).std().reset_index().drop('realisation', axis=1)
            #-----------gradient plot: individual realisations------------------------------------------------------
            
            if xcol == 'logOHcen': xcol = xcol - logOHsun
            '''
            y_arr = (table[ycol1].values - table['logOHgrad'].values)*100./table['logOHgrad'].values
            yerr = (table[ycol1+'_u'].values)*100./table['logOHgrad'].values
            ax1.scatter(table[xcol].values,y_arr, linewidth=0, c=col_ar[ind])
            ax1.errorbar(table[xcol].values,y_arr,yerr=yerr, linestyle='None', c=col_ar[ind], alpha=a)    
            '''
            #----------gradient plot: median of all realisations-------------------------------------------------------
            y_arr = (table2[ycol1].values - table2['logOHgrad'].values)*100./table2['logOHgrad'].values
            yerr = (table3[ycol1].values)*100./table3['logOHgrad'].values
            ax1.scatter(table2[xcol].values,y_arr, c=col_ar[ind], linewidth=0, label='SNR inp='+str(fsnr)+', out='+str('%.2F +/- %.2F'%(np.mean(table2['snr_measured'].values), np.mean(table3['snr_measured'].values))))
            ax1.errorbar(table2[xcol].values,y_arr,yerr=yerr, linestyle='None',c=col_ar[ind], alpha=a)
        
            if not all_in_one_plot and plot_intercept:
                #-----------intercept plot: individual realisations------------------------------------------------------
                '''
                y_arr = (table[ycol2].values  + logOHsun - table['logOHcen'].values)*100./(table['logOHcen'].values)
                yerr = (table[ycol2+'_u'].values)*100./(table['logOHcen'].values)
                ax2.scatter(table[xcol].values,y_arr, linewidth=0, c=col_ar[ind])
                ax2.errorbar(table[xcol].values,y_arr,yerr=yerr, linestyle='None',c=col_ar[ind], alpha=a)
                '''
                #----------intercept plot: median of all realisations-------------------------------------------------------
                y_arr = (table2[ycol2].values  + logOHsun - table2['logOHcen'].values)*100./(table2['logOHcen'].values)
                yerr = (table3[ycol2].values)*100./(table3['logOHcen'].values)
                ax2.scatter(table2[xcol].values,y_arr, c=col_ar[ind], linewidth=0)
                ax2.errorbar(table2[xcol].values,y_arr,yerr=yerr, linestyle='None',c=col_ar[ind], alpha=a)
    
        if col == 0 or not all_in_one_plot: #only for the very first subplot
            ax1.legend(loc="lower left" if all_in_one_plot else "best", fontsize='4' if all_in_one_plot else '8')
        
        try: plt.xlim(xmin,xmax)
        except: pass
        ax1.set_ylim(ylim1)
        ax1.text(ax1.get_xlim()[-1]*0.99, np.diff(ax1.get_ylim())*0.08, 'steeper',color='k',fontsize=10,ha='right')
        ax1.text(ax1.get_xlim()[-1]*0.99, -np.diff(ax1.get_ylim())*0.08, 'shallower',color='k',fontsize=10,ha='right')
        
        if all_in_one_plot:
            if row == 0:
                ax3 = ax1.twiny()
                ax3.set_xlabel('SNR cut='+str(snr_cut_choice))
                ax3.set_xticks([])
            if not row == nrow-1: ax1.get_xaxis().set_ticklabels([])
            if col == 0: ax1.set_ylabel(r'Inp $\nabla_r$logZ='+str(logOHgrad_choice))
            else: ax1.get_yaxis().set_ticklabels([])
            print 'Done logOHgrad_choice=', logOHgrad_choice, 'snr_cut_choice=', snr_cut_choice
        else:
            plt.xlabel(xlab)
            ax1.set_ylabel('% offset in '+ylab1)
            if plot_intercept:
                ax2.set_ylabel('% offset in '+ylab2)
                try: ax2.set_ylim(ylim2)
                except: pass
            fig.savefig(outpath+title+'.eps')
            print 'Saved file', outpath+title+'.eps'

if all_in_one_plot:
    fig.text(0.5, 0.02, xlab, ha='center')
    fig.text(0.02, 0.5, '% offset in '+ylab1, va='center', rotation='vertical')
    fig.savefig(outpath+t+'.eps')
    print 'Saved file', outpath+t+'.eps'

if hide: plt.close('all')
else: plt.show(block=False)
print 'Finished!'