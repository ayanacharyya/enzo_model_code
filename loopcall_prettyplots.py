# python convenience routine to call stuff from plotobservables.py/getppv.py in a loop, to automate things
# by Ayan
import subprocess
import sys
import numpy as np
import os

HOME = os.getenv('HOME') + '/'
import time

start_time = time.time()

multi = 1
function = HOME + 'Work/astro/ayan_codes/enzo_model_code/plotobservables.py'
snr_arr = [0]
logOHcen = 8.77
logOHgrad_arr = [-0.1]  # arg = logOHgrad
scale_exptime_arr = [1200000]
file_arr = ['DD0600_lgf']
vres_arr = [30]  # vres
arc_arr = [-99, 2.0, 2.0, 4.0]  # arg = res_arcsec
outpath = '/avatar/acharyya/enzo_models/ppvcubes_fixednoise_new_contsub_contu_wrong/'
fixed_noise_arr = [300., 300., 5., 5.]
for file in file_arr:
    for scale_exptime in scale_exptime_arr:
        for logOHgrad in logOHgrad_arr:
            for snr in snr_arr:
                for vres in vres_arr:
                    for (jj, fixed_noise) in enumerate(fixed_noise_arr):
                        subprocess.call(['python ' + function + ' --addnoise --fixed_noise '+str(fixed_noise) + ' --smooth --spec_smear ' + str(vres) + ' --wmin 6400 --Zgrad ' + str(logOHcen) + ',' + str(logOHgrad) + ' --met --snr ' + str(snr) + ' --calc --hide \
--arc ' + str(arc_arr[jj]) + ' --scale_exptime ' + str(scale_exptime) + ' --file ' + file + ' --multi_realisation ' + str(
                        multi) + ' --nocreate --res 0.04 --binto 4 --contsub --met --calc --nowrite --showbin --getmap --cmax 0 --cmin '+str(logOHgrad*12)+' --saveplot --path ' + outpath], shell=True)
                        '''
                        subprocess.call(['python ' + function + ' --addnoise --fixed_noise '+str(fixed_noise)+' --smooth --spec_smear \
    --vres ' + str(vres) + ' --wmin 6400 --Zgrad ' + str(logOHcen) + ',' + str(logOHgrad) + ' --map --line H6562 --snr ' + str(snr) + ' --hide \
    --arc ' + str(arc_arr[jj]) + ' --scale_exptime ' + str(scale_exptime) + ' --file ' + file + ' --multi_realisation ' + str(
                            multi) + ' --nocreate --res 0.04 --binto 4 --contsub --saveplot --cmin -22 --cmax -16 --path ' + outpath], shell=True)
                        '''
                    '''
                    subprocess.call(['python ' + function + ' --addnoise --fixed_noise '+str(fixed_noise_arr[-1])+' --smooth --spec_smear \
    --vres ' + str(vres) + ' --wmin 6400 --Zgrad ' + str(logOHcen) + ',' + str(logOHgrad) + ' --map --snrmap --line NII6584 --snr ' + str(snr) + ' --hide \
    --arc ' + str(arc_arr[-1]) + ' --scale_exptime ' + str(scale_exptime) + ' --file ' + file + ' --multi_realisation ' + str(
                            multi) + ' --nocreate --res 0.04 --binto 4 --contsub --saveplot --snrcmin 0 --snrcmax 20 --cmin -22 --cmax -16 --path ' + outpath], shell=True)
                    '''
print 'Finished looping in %s minutes!' % ((time.time() - start_time) / 60)
