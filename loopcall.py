#python convenience routine to call stuff from plotobservables.py/getppv.py in a loop, to automate things
#by Ayan
import subprocess
import sys

function = 'plotobservables.py'
arg_arr = [10,20,30,50,70,100,120,150] #[150,200,300,400,500,600,700,800] #arg = vres
snr = 5.
for arg in arg_arr:
    filename = 'PPV_DD0600_lgfOm=0.5,res=0.4kpc_6400.0-6782.674A_specsmeared_'+str(arg)+'kmps_smeared_moff_parm5,4.7,51_noisy_obs.fits'
    subprocess.call(['python '+function+' --fitsname '+filename+' --addnoise --smooth \
    --spec_smear --vres '+str(arg)+' --wmin 6400 --met --snr '+str(snr)+' --saveplot']\
    ,shell=True)
    
print 'Finished looping!'