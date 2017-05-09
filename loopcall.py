#python convenience routine to call stuff from plotobservables.py/getppv.py in a loop, to automate things
#by Ayan
import subprocess
import sys
import numpy as np
import os
HOME = os.getenv('HOME')+'/'
import time
start_time= time.time()

function = 'plotobservables.py'
snr_arr = np.arange(11)
logOHcen = 8.77
logOHgrad_arr = [-0.05]#[-0.1,-0.05,-0.025,-0.01] #arg = logOHgrad
scale_exptime_arr = [600]#,1200] #exp time for arc=0.5, pivoting which the exp time for other arc values would be determined
file_arr = ['DD0600_lgf']#,'DD0600']
#multi_arr = np.arange(10)+1 #different realisations
vres_arr = [10,20,30,50,70,90] #vres
arc_arr = [0.1,0.5,0.8,1.0,1.5,2.0,2.5,3.0] #arg = res_arcsec
outpath = HOME+'Desktop/bpt/'
#fixed_SNR = 5.
for file in file_arr:
    for scale_exptime in scale_exptime_arr:
        for logOHgrad in logOHgrad_arr:
            for snr in snr_arr:
                arc = 0.5
                for vres in vres_arr:
                    subprocess.call(['python '+function+' --addnoise --smooth --spec_smear \
--vres '+str(vres)+' --wmin 6400 --Zgrad '+str(logOHcen)+','+str(logOHgrad)+' --met --snr '+str(snr)+' --calc --hide --keep \
--arc '+str(arc)+' --scale_exptime '+str(scale_exptime)+' --file '+file+' --path '+outpath],shell=True)
                
                vres = 30
                for arc in arc_arr:
                    subprocess.call(['python '+function+' --addnoise --smooth --spec_smear \
--vres '+str(vres)+' --wmin 6400 --Zgrad '+str(logOHcen)+','+str(logOHgrad)+' --met --snr '+str(snr)+' --calc --hide --keep \
--arc '+str(arc)+' --scale_exptime '+str(scale_exptime)+' --file '+file+' --path '+outpath],shell=True)

print 'Finished looping in %s minutes!'% ((time.time() - start_time)/60)