#python convenience routine to call loopcall.py  in a loop, to automate things
#by Ayan
import subprocess
import os
HOME = os.getenv('HOME')+'/'

node = 9
ncores = 16
function = HOME+'models/enzo_model_code/loopcall.py'

for realisation in range(1,ncores+1):
    fout = open('output_node'+str(node)+'_core'+str(realisation),'w')
    subprocess.call(['python '+function+' '+str(realisation)],stderr=subprocess.STDOUT,stdout=fout,shell=True)
