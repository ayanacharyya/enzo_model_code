#python convenience routine to call stuff from plotobservables.py/getppv.py in a loop, to automate things
#by Ayan
import subprocess
import sys
import numpy as np
import os
HOME = os.getenv('HOME')+'/'
import time
start_time= time.time()

multi = int(sys.argv[1])
function = HOME+'models/enzo_model_code/plotobservables2.py'
snr_arr = np.arange(11)
logOHcen = 8.77
logOHgrad_arr = [-0.1,-0.05,-0.025,-0.01] #arg = logOHgrad
scale_exptime_arr = [240000,480000] #exp time for arc=0.5, pivoting which the exp time for other arc values would be determined
file_arr = ['DD0600_lgf','DD0600']
vres_arr = [90,70,50,30,20,10] #vres
arc_arr = [0.5,1.,2.,3.,4.,5.,7.5,10.,12.5] #arg = res_arcsec
outpath = '/avatar/acharyya/enzo_models2/ppvcubes4/'
#fixed_SNR = 5.
for file in file_arr:
    for scale_exptime in scale_exptime_arr:
        for logOHgrad in logOHgrad_arr:
            for snr in snr_arr:
                arc = 2
                for vres in vres_arr:
                    subprocess.call(['python '+function+' --addnoise --smooth --spec_smear \
--vres '+str(vres)+' --wmin 6400 --Zgrad '+str(logOHcen)+','+str(logOHgrad)+' --met --snr '+str(snr)+' --calc --hide \
--arc '+str(arc)+' --scale_exptime '+str(scale_exptime)+' --file '+file+' --multi_realisation '+str(multi)+' --path '+outpath],shell=True)
                    
                vres = 30
                for arc in arc_arr:
                    subprocess.call(['python '+function+' --addnoise --smooth --spec_smear \
--vres '+str(vres)+' --wmin 6400 --Zgrad '+str(logOHcen)+','+str(logOHgrad)+' --met --snr '+str(snr)+' --calc --hide \
--arc '+str(arc)+' --scale_exptime '+str(scale_exptime)+' --file '+file+' --multi_realisation '+str(multi)+' --path '+outpath],shell=True)

print 'Finished looping in %s minutes!'% ((time.time() - start_time)/60)
