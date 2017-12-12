# utility routine to check HII odel grid: metallicity dependencies on line ratios
# by Ayan, Sep 2017
import time

start_time = time.time()
import numpy as np
import sys

sys.path.append('/Users/acharyya/Work/astro/ayan_codes/HIIgrid/')
import rungrid as r
import subprocess
import pandas as pd

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
from matplotlib import pyplot as plt

gridname = '/Users/acharyya/Mappings/lab/totalspec' + r.outtag + '.txt'
grid = pd.read_table(gridname, comment='#', delim_whitespace=True)

short = grid[['Z', 'age', 'nII', '<U>', 'NII6584', 'H6562', 'SII6717', 'SII6730']]
# short = short[short['nII'].between(10**(5.2-4+6),10**(6.7-4+6))] #D16 models have 5.2 < lpok < 6.7 #DD600_lgf with Zgrad8.77,-0.1 has 4.3 < lpok < 8
# short = short[short['Z'].between(0.1, 3.)] #D16 models have 0.1 < Z < 3. DD600_lgf with Zgrad8.77,-0.1 has 0.07 < Z < 0.95
# short = short[short['<U>'].between(10**-3.5,10**-2.)] #d16 models have -3.5 < logU < -2. DD600_lgf with Zgrad8.77,-0.1 has -3.5 < <U> < -2.
# color_col, islogcol = 'SII6730', 1
color_col, islogcol = 'nII', 1

short['N2Ha'] = np.log10(short['NII6584'] / short['H6562'])
short['N2short'] = np.log10(short['NII6584'] / (short['SII6717'] + short['SII6730']))
short['log_ratio'] = short['N2short'] + 0.264 * short['N2Ha']
short['logZ_D16'] = short['log_ratio'] + 0.45 * (short['log_ratio'] + 0.3) ** 5
short['Z_D16'] = 10 ** (short['logZ_D16'])

Z = short[np.isfinite(short['Z_D16'])]['Z']
Z_D16 = short[np.isfinite(short['Z_D16'])]['Z_D16']
col = short[np.isfinite(short['Z_D16'])][color_col]
if islogcol: col = np.log10(col) - 6.

fig = plt.figure()
plt.scatter(Z, Z_D16, s=20, lw=0, c=col)
cb = plt.colorbar()
cb.set_label(color_col + ' log(cm^-3)')
plt.plot(Z, Z, c='r')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
# plt.yscale('log')
plt.xlabel('input Z')
plt.ylabel('output Z')
# fig.savefig('/Users/acharyya/Desktop/bpt/4Dgrid_ZoutD16_vs_Zin.eps')
plt.show(block=False)
