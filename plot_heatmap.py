# python utility routine to plot heat map of reliability of apparent metallicity gradient with varying spatial res and SNR
# by Ayan, Feb 2018

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
import sys
import subprocess
import re
import os
HOME = os.getenv('HOME') + '/'
sys.path.append(HOME + 'Work/astro/ayan_codes/mageproject/')
import ayan.splot_util as u
# -----------------------------------------------------------------
do_one_plot = 0 #
inverse_xaxis = 1 #
savefig = 1 #
savetab = 1 #
hide = 1 #
inpath = HOME + 'Desktop/bpt_contsub_contu_rms/'
diag_arr = ['KD02', 'D16']
Om_arr = [0.5, 0.05, 5.0]
simulation_choice_arr = ['DD0600', 'DD0600_lgf']
snr_cut_choice_arr = [5,0]
fs = 15 # fontsize

# -----------------------------------------------------------------
if do_one_plot:
    diag_arr = diag_arr[:1]
    Om_arr = Om_arr[:1]
    simulation_choice_arr = simulation_choice_arr[:1]
    snr_cut_choice_arr = snr_cut_choice_arr[:1]
logOHsun = 8.77
inv = -1. if inverse_xaxis else 1.  # inv=-1 to get in units of beams per scale radius
outpath = HOME+'Dropbox/papers/enzo_paper/' # till just outside the Figs/ or Tabs/ directories
ycol1, ylab1 = 'slope', r'$\nabla_r$logZ'
ycol2, ylab2 = 'intercept', r'log($Z_0$/$Z_\odot$)'
xcol = 'res_phys'
xlab = 'Number of beams per scale radius' if inverse_xaxis else r'Spatial resolution (in units of scale length)'
vres_choice = 30  # choose from [10,20,30,50,70,90] or None
# fixed_SNR_choice = 5 #choose from [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10] or None
size_choice = 5  # or None
power_choice = 4.7  # or None
logOHcen_choice = logOHsun  # or None
scale_length = 4.  # kpc
binto = 2 # previously 4
z = 0.04 # previously 0.013
ylim2 = (-2, 0.1)
snr_mode = 'fixed_noise'  # #choose between fixed_SNR and fixed_noise
snr_mode_inp = 'fixed_noise_inp'  #to have *good* i.e. input*1.4 values of fixed_noise on plots, table, etc.
const = 1.4  #to have *good* i.e. input*1.4 values of fixed_noise on plots, table, etc.
plt.close('all')

# -----------------------------------------------------------------
for diag in diag_arr:
    for Om in Om_arr:
        filename_root = 'met_grad_Om=' + str(Om) + '_' + diag + '_exp_scaled'
        merge_text = '_branches_merged'  # if diag == 'KD02' else ''
        noconvolution_filename = filename_root + '_' + snr_mode + merge_text + '.txt'
        filename = filename_root + '_' + snr_mode + '_binto'+str(binto) + merge_text + '.txt'  # noise level fixed while generating PPV cube

        full_table = pd.read_table(inpath + filename, delim_whitespace=True, comment="#")
        if os.path.exists(inpath + noconvolution_filename):
            print 'Found noconvolution file ' + noconvolution_filename + '. Including no convolution case.'
            noconvolution_fulltable = pd.read_table(inpath + noconvolution_filename, delim_whitespace=True, comment="#")
            full_table = pd.concat([noconvolution_fulltable, full_table])
            full_table.sort_values(by=['simulation', 'res_arcsec', 'logOHgrad', snr_mode], inplace=True)
        else:
            print 'Noconvolution file ' + noconvolution_filename + ' does not exist. So not plotting no convolution case.'
        full_table['snr_measured'] = full_table['snr_measured'].astype(np.float64)
        full_table[snr_mode_inp] = (full_table[snr_mode]*const).astype(np.float64)
        full_table = full_table[full_table[snr_mode_inp] <= 300.]  # to exclude very high SNR cases

        for simulation_choice in simulation_choice_arr:
            outtab_master = pd.DataFrame(columns=[''])
            for ind,snr_cut_choice in enumerate(snr_cut_choice_arr):
                master_table = full_table[(full_table.simulation == simulation_choice) & (full_table.vres == vres_choice) & (full_table.power == power_choice) & \
                (full_table['size'] == size_choice) & (full_table.logOHcen == logOHcen_choice) & (full_table.snr_cut == snr_cut_choice)]

                logOHgrad_arr = pd.unique(master_table['logOHgrad'])
                if do_one_plot: logOHgrad_arr = logOHgrad_arr[:1]
                logOHgrad_arr = logOHgrad_arr[~(np.isnan(logOHgrad_arr))]
                logOHgrad_text = ['']*len(logOHgrad_arr)

                fixed_noise_arr = pd.unique(master_table[snr_mode_inp])
                fixed_noise_arr = fixed_noise_arr[~(np.isnan(fixed_noise_arr))]
                res_phys_arr = pd.unique(master_table['res_phys'])
                res_phys_arr = res_phys_arr[~(np.isnan(res_phys_arr))]

                t = simulation_choice +'_Om=' + str(Om) + '_' + diag + '_heatmap_SNR_vs_res_phys_vres=' + str(vres_choice) +'_lOHcen=' + str(logOHcen_choice)
                if '_fixed_SNR' in filename: t += '_fixed_SNR'
                if '_fixed_noise' in filename: t += '_fixed_noise'
                if 'binto' in filename: t += '_binto' + re.findall('binto(\d+)', filename)[0]
                if 'fitupto' in filename: t += '_fitupto' + re.findall('fitupto(\d+)', filename)[0]

                # -----------------------------------------------------------------
                #outtab_master[''] = np.tile(fixed_noise_arr, len(logOHgrad_arr)) # to display input SNR in table
                outtab_master[''] = ['%.1F'%m[1][snr_mode].mean() for m in master_table.groupby(['logOHgrad',snr_mode_inp])] # to display mean output SNR in table
                outtab = pd.DataFrame(columns=[''] + ['%.2F' % ((item/scale_length)**inv) for item in res_phys_arr])
                for ind2,logOHgrad in enumerate(logOHgrad_arr):
                    table1 = master_table[master_table['logOHgrad'] == logOHgrad]

                    fig = plt.figure(figsize=(8, 6))
                    fig.subplots_adjust(hspace=0.02, wspace=0.02, top=0.9, bottom=0.1, left=0.1, right=0.85)
                    heat_map = []
                    for fixed_noise in fixed_noise_arr:
                        table2 = table1[table1[snr_mode_inp] == fixed_noise]
                        heat_row = []
                        for res_phys in res_phys_arr:
                            table3 = table2[table2['res_phys'] == res_phys]
                            if len(table3) > 0:
                                table4 = table3.groupby(['simulation', 'res_arcsec', 'res_phys', 'vres', 'power', 'size', 'logOHcen', \
                                'logOHgrad']).mean().reset_index().drop('realisation', axis=1)
                                offset = (table4[ycol1].values - table4['logOHgrad'].values) * 100. / table4['logOHgrad'].values
                            else: offset = [np.nan] # if data doesn't exist for that combination of SNR and res
                            heat_row.append(offset[0])
                        heat_map.append(heat_row)
                        outtab.loc[len(outtab)] = ['%.0F'%fixed_noise] + [r'%.1F' % (item) for item in heat_row]
                    if (np.array(heat_map) > 0).any(): cmin, cmax, csteps, cmap = -100, 100, 7, 'RdBu'
                    else: cmin, cmax, csteps, cmap = -100, 0, 5, 'Reds_r'
                    ax = plt.gca()
                    p = ax.pcolor(heat_map, cmap=cmap, vmin=cmin, vmax=cmax)
                    p.cmap.set_under('gray')

                    ax.set_xticks(ax.get_xticks()+0.5)
                    ax.set_xlim(0,np.shape(heat_map)[1])
                    ax.set_xticklabels(['%.1f'%((res_phys_arr[int(i-0.5)]/scale_length)**inv) for i in list(ax.get_xticks()[:-1])], fontsize=fs)
                    ax.set_xlabel(xlab, fontsize=fs)

                    ax.set_yticks(ax.get_yticks()+0.5)
                    ax.set_ylim(0,np.shape(heat_map)[0])
                    ax.set_yticklabels(['%.1f'%(fixed_noise_arr[int(i-0.5)]) for i in list(ax.get_yticks()[:-1])], fontsize=fs)
                    ax.set_ylabel('SNR', fontsize=fs)

                    ax2 = ax.twiny()
                    ax2.set_xlim(ax.get_xlim())
                    ax2.set_xticks(ax.get_xticks()[:-1])
                    ax2.set_xticklabels(['%.1F' % (res_phys_arr[int(i-0.5)] / (np.pi / (3600 * 180) * ((z/0.013) * 55.714285714285715 * 1e3))) for i in list(ax2.get_xticks())], fontsize=fs)
                    ax2.set_xlabel('PSF size (in arcseconds)', fontsize=fs)

                    #divider = make_axes_locatable(ax)
                    #cax = divider.append_axes("right", size="5%", pad=0.1)
                    cax = fig.add_axes([0.87, 0.1, 0.03, 0.8])
                    cb = plt.colorbar(p, cax=cax, ticks=np.linspace(cmin+10, cmax-10, csteps).astype(int), ax=[ax, ax2])
                    cb.set_label('% Offset from true gradient', fontsize=fs)
                    cb.ax.tick_params(labelsize=fs)

                    ax.text(ax.get_xlim()[-1]-0.2, ax.get_ylim()[-1]-0.5, 'Input gradient = '+str(logOHgrad), color='k', ha='right', va='center', fontsize=fs)
                    ax.text(ax.get_xlim()[-1]-0.2, ax.get_ylim()[-1]-1, 'SNR cut-off = '+str(snr_cut_choice), color='k', ha='right', va='center', fontsize=fs)
                    ax.set_aspect('auto')
                    ax2.set_aspect('auto')
                    '''
                    # add a "background patch" for NAN values in the plot
                    xmin, xmax = ax.get_xlim()
                    ymin, ymax = ax.get_ylim()
                    xy = (xmin, ymin)
                    width = xmax - xmin
                    height = ymax - ymin
                    p = patches.Rectangle(xy, width, height, hatch='///', fc='gray', zorder=-10) # create the patch and place it in the back of countourf (zorder!)
                    ax.add_artist(p)
                    '''
                    if savefig:
                        outfig = outpath+'Figs/'+t+'_lOHgrad='+str(logOHgrad)+'_snr_cut='+str(snr_cut_choice)+'.eps'
                        fig.savefig(outfig)
                        print 'Saved plot', outfig
                    else:
                        print 'Not saving plot.'
                    if hide: plt.close()
                    else: plt.show(block=False)
                    logOHgrad_text[ind2] = r'&\multicolumn{' + str(len(res_phys_arr)) + '}{c}{Input gradient = '+str(logOHgrad)+' dex/kpc}'
                snr_cut_off_text = r'&\multicolumn{'+str(len(res_phys_arr))+'}{c}{SNR cut-off = '+str(snr_cut_choice)+' }'
                res_text = 'SNR' + r'&\multicolumn{' + str(len(res_phys_arr)) + '}{c}{'+xlab+'}'
                #outtab_master = pd.concat([outtab_master, outtab[outtab.columns[1:]]], axis=1)
                outtab = pd.concat([outtab_master, outtab[outtab.columns[1:]]], axis=1)

                for col in outtab.columns:
                    outtab[col] = outtab[col].astype(np.str)
                    outtab[col] = outtab[col].str.replace('-', r'$-$')
                outtab = outtab.replace('NAN', '-')
                if savetab:
                    outtabfile = 'Tables/'+t+'_snr_cut='+str(snr_cut_choice)+'.tex'
                    outtab.to_latex(outpath + outtabfile, index=None, escape=False)
                    # --adding additional material to table--
                    u.insert_line_in_file(snr_cut_off_text + '\\\ \n', 2, outpath + outtabfile)
                    u.insert_line_in_file('\hline \n', 3, outpath + outtabfile)
                    u.insert_line_in_file(res_text + '\\\ \n', 4, outpath + outtabfile)  # use -1 instead of 0 to append caption to end of table

                    u.insert_line_in_file('\hline \n', 5, outpath + outtabfile)
                    for i,logOHgrad in enumerate(logOHgrad_arr):
                        index = 8 + len(fixed_noise_arr)*i + 3*i
                        u.insert_line_in_file('\hline \n', index, outpath + outtabfile)
                        u.insert_line_in_file(logOHgrad_text[i] + '\\\ \n', index+1, outpath + outtabfile)
                        u.insert_line_in_file('\hline \n', index+2, outpath + outtabfile)
                    print 'Saved file', outpath + outtabfile
                else:
                    print 'Not saving table.'

