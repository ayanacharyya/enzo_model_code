#python utility routine to plot trends in apparent metallicity gradient and scatter with varying properties telescope
#by Ayan

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
import sys
HOME ='/Users/acharyya/'
#-----------------------------------------------------------------
logOHsun = 8.77
outpath = HOME+'Desktop/bpt/'
inpath = HOME+'models/enzo_model_code/'
#filename = 'met_grad_log_paint_fixedSNR5.0.txt' #SNR fixed while generating PPVcube
filename = 'met_grad_log_paint_exp.txt' #exposure time scaled with spatial res
simulation_choice = 'DD0600'#_lgf' #choose from ['DD0600_lgf','DD0600']
scale_exptime_choice = 600. #choose from [600, 1200]
ycol1, ylab1 = 'slope', r'$\nabla_r$logZ'
ycol2, ylab2 = 'intercept', r'log($Z_0$/$Z_\odot$)'
res_arcsec_choice = None #choose from [0.1,0.5,0.8,1.0,1.5,2.0,2.5,3.0] or None
vres_choice = 30 #choose from [10,20,30,50,70,90] or None
SNR_choice = 5 #choose from [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10] or None
size_choice = 10 #or None
power_choice = 4.7 #or None
logOHcen_choice = logOHsun #or None
logOHgrad_choice = -0.1 #choose from [-0.1,-0.05,-0.025,-0.01] or None
ylim1l, ylim1u = -0.05,0.05 # 4
ylim2l, ylim2u = -0.1,0.1
xmin, xmax = None, None
#-----------------------------------------------------------------
if res_arcsec_choice is None: xcol = 'res_arcsec'
elif vres_choice is None: xcol = 'vres'
elif SNR_choice is None: xcol ='SNR_thresh'
elif size_choice is None: xcol ='size'
elif power_choice is None: xcol ='power'
elif logOHcen_choice is None: xcol = 'logOHcen'
elif logOHgrad_choice is None: xcol = 'logOHgrad'

if xcol == 'res_arcsec': xlab = r'$\Delta r$ (")'
elif xcol == 'vres': xlab = r'$\Delta v$ (km/s)'
elif xcol == 'SNR_thresh': xlab = 'SNR_cut-off'
elif xcol == 'size': xlab = 'PSF size(pixels)'
elif xcol == 'power': xlab = 'PSF exponent'
elif xcol == 'logOHcen': xlab = r'True log($Z_0$/$Z_\odot$)'
elif xcol =='logOHgrad': xlab = r'True $\nabla_r$logZ (dex/kpc)'
#--------------selection criteria to make plots-------------------------------------------
full_table = pd.read_table(inpath+filename, delim_whitespace=True, comment="#")
table = full_table[full_table.simulation == simulation_choice]
table = table.drop('simulation', axis=1)
if xmin is None: xmin = np.min(table[xcol].values)*0.5
if xmax is None: xmax = np.max(table[xcol].values)*1.1
try:
    table = table[table.scale_exptime == scale_exptime_choice]
    table.drop('scale_exptime', axis=1, inplace=True)
except: pass
try:
    table = table.groupby(['res_arcsec','vres','power','size','logOHcen','logOHgrad','SNR_thresh']).median().reset_index().drop('realisation', axis=1)
    print 'Re-grouped table by taking median over different realisations.'
except: pass
#-----------------------------------------------------------------
if res_arcsec_choice is not None: table = table[table.res_arcsec == res_arcsec_choice]
if vres_choice is not None: table = table[table.vres == vres_choice]
if SNR_choice is not None: table = table[table.SNR_thresh == SNR_choice]
if size_choice is not None: table = table[table['size'] == size_choice]
if power_choice is not None: table = table[table.power == power_choice]
try:
    if logOHcen_choice is not None: table = table[table.logOHcen == logOHcen_choice]
    if logOHgrad_choice is not None: table = table[table.logOHgrad == logOHgrad_choice]
except:
    print 'This file does not have logOHcen and/or logOHgrad entries.'
    pass
#-----------------------------------------------------------------
fig = plt.figure(figsize=(10,6))
t = simulation_choice+'-Z-'+ycol1+','+ycol2+'_vs_'+xcol
if res_arcsec_choice is not None: t += '_arc='+str(res_arcsec_choice)
if vres_choice is not None: t += '_vres='+str(vres_choice)
if SNR_choice is not None: t += '_SNRcut='+str(SNR_choice)
if logOHcen_choice is not None: t += '_lOHcen='+str(logOHcen_choice)
if logOHgrad_choice is not None: t += '_lOHgrad='+str(logOHgrad_choice)
if '_exp' in filename: t += '_exp_scaled_'+str(int(scale_exptime_choice))
if '_fixedSNR' in filename: t += '_fixedSNR'+str(filename[-7:-4])

fig.text(0.5, 0.92, t, ha='center')

if xcol == 'logOHcen': xcol = xcol - logOHsun
ax1 = fig.add_subplot(2,1,1)
if logOHgrad_choice is None: true_ycol1 = table['logOHgrad'].values
else: true_ycol1 = logOHgrad_choice
plt.axhline(0., c='k', linestyle='--', label='True '+ylab1)
ax1.scatter(table[xcol].values,table[ycol1].values-true_ycol1, label='Inferred - True '+ylab1)
ax1.errorbar(table[xcol].values,table[ycol1].values-true_ycol1,yerr=table[ycol1+'_u'].values, linestyle='None')
#plt.xlabel(xcol)
plt.ylabel('Inferred - True '+ylab1)
plt.xlim(xmin,xmax)
plt.ylim(ylim1l,ylim1u)
#ax1.legend()#bbox_to_anchor=(0.35, 0.3), bbox_transform=plt.gcf().transFigure)

ax2 = fig.add_subplot(2,1,2)
if logOHcen_choice is None: true_ycol2 = table.logOHcen.values - logOHsun
else: true_ycol2 = logOHcen_choice - logOHsun
plt.axhline(0., c='k', linestyle='--', label='True '+ylab2)
ax2.scatter(table[xcol].values,table[ycol2].values-true_ycol2, label='Inferred - True '+ylab2)
ax2.errorbar(table[xcol].values,table[ycol2].values-true_ycol2,yerr=table[ycol2+'_u'].values, linestyle='None')
plt.xlabel(xlab)
plt.ylabel('Inferred - True '+ylab2)
plt.xlim(xmin,xmax)
plt.ylim(ylim2l,ylim2u)
#ax2.legend()

fig.savefig(outpath+t+'.png')
print 'Saved file', outpath+t+'.png'
plt.show(block=False)
print 'Finished!'