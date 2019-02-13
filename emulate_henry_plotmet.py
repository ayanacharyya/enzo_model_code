# python utility routine to plot trends in apparent metallicity gradient following Henry's method and scatter with varying properties telescope
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

HOME = os.getenv('HOME') + '/'
# --------------------------------------------------
inpath = HOME+'Desktop/bpt_contsub_contu_rms/'
fixed_noise_arr = np.array([30, 300])/1.4
snr_cut_arr = [0, 5]
col_dict = {'onlyHII': 'r', 'onlyDIG': 'b', 'HII+DIG': 'k'}
fs = 15
plt.close('all')
# --------------------------------------------------

for fixed_noise in fixed_noise_arr:
    infile = 'emulate_henry_met_grad_DD0600Om=0.5_arc0.2_vres30.0kmps_fixed_noise'+str(fixed_noise)+'_logOHgrad-0.1.txt'
    full_table = pd.read_table(inpath + infile, delim_whitespace=True, comment="#")
    for snr_cut in snr_cut_arr:
        table_snr = full_table[full_table['snr_cut'] == snr_cut]
        diag_arr = pd.unique(table_snr['diagnostic'])
        fig, axes = plt.subplots(len(diag_arr), 1, sharex=True, figsize=(8, 6))
        fig.subplots_adjust(hspace=0.1, wspace=0.25, top=0.9, bottom=0.15, left=0.14, right=0.98)
        for (ii,diag) in enumerate(diag_arr):
            ax = axes[ii]
            table_diag = table_snr[table_snr['diagnostic'] == diag]
            comp_arr = pd.unique(table_diag['component'])
            for comp in comp_arr:
                table_comp = table_diag[table_diag['component'] == comp]
                if len(table_comp) > 0:
                    ax.plot(table_comp['res_bin'], table_comp['slope'], c=col_dict[comp], label=comp)
                    ax.errorbar(table_comp['res_bin'].values, table_comp['slope'].values, yerr=table_comp['slope_u'].values, c=col_dict[comp])
                plt.legend(fontsize=fs)
            ax.set_xlim(1,1200)
            ax.set_ylim(-0.1, 0.05)
            ax.text(ax.get_xlim()[0]+np.diff(ax.get_xlim())[0]*0.05, ax.get_ylim()[-1]-np.diff(ax.get_ylim())[0]*0.1, diag, color='k', ha='left', va='center', fontsize=fs)
            ax.set_ylabel('Z_grad (dex/kpc)', fontsize=fs)
        ax.set_xlabel('Resolution (pc/spaxel)', fontsize=fs)
        fig.text(0.5, 0.95, 'SNR_data = %.0F, SNR_cutoff = %.0F'%(fixed_noise*1.4, snr_cut), va='center', ha='center', fontsize=fs)
        plt.show(block=False)
        fig.savefig(inpath + 'slope_vs_res'+os.path.splitext(infile)[0]+'_snr'+str(snr_cut)+'.eps')
        print 'Done fixed_noise=', fixed_noise, 'snr_cut=', snr_cut