#python convenience routine to call stuff from plotobservables.py/getppv.py in a loop, to automate things
#by Ayan
import subprocess
import sys
import numpy as np
import os
HOME = os.getenv('HOME')+'/'
import time
start_time= time.time()

multi = int(sys.argv[1]) #number of realisation
ncores = 83
function = HOME+'Work/astro/ayan_codes/enzo_model_code/plotobservables.py'
logOHcen = 8.77
logOHgrad_arr = [-0.1,-0.05,-0.025,-0.01] #arg = logOHgrad
file_arr = ['DD0600'] # 'DD0600' or 'DD0600_lgf'
arc_arr = [1.,1.5, 2.,2.5,3.,3.5,4.,5.] #arg = res_arcsec
snr_arr = [0, 3, 5, 8, 10]
outpath = '/avatar/acharyya/enzo_models/ppvcubes_fixednoise_contsub_conturms/'
fixed_noise_arr = [1.,3.,5.,8.,10.,30.]
if multi == 1:
    arc_arr += [-99] # make no convolution for only ONE realisation, since at this level of resolution all realisations would practically be identical
    fixed_noise_arr += [300.] # make SNR=300 for only ONE realisation, since at this SNR all realisations would practically be identical
for file in file_arr:
    for fixed_noise in fixed_noise_arr:
        for logOHgrad in logOHgrad_arr:
            vres = 30 #km/s
            for arc in arc_arr:
                for snr in snr_arr:
                    if file == 'DD0600_lgf': galsize = 26. # kpc
                    elif file == 'DD0600': galsize = 27.5 # kpc
                    subprocess.call(['python '+function+' --addnoise --contsub \
                    --smooth --spec_smear --res 0.04 --binto 4 --galsize '+str(galsize)+' \
                    --vres '+str(vres)+' --wmin 6400 --Zgrad '+str(logOHcen)+','+str(logOHgrad)+' --met \
                    --fixed_noise '+str(fixed_noise)+' --calc --hide --scale_exptime 1200000 --snr '+str(snr)+' \
                    --arc '+str(arc)+' --file '+file+' --ncores '+str(ncores)+' --multi_realisation '+str(multi)+' --path '+outpath],shell=True)

print 'Finished looping in %s minutes!'% ((time.time() - start_time)/60)
