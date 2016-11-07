#!/usr/bin/env python

import time
start_time1 = time.time()
import subprocess
import os.path
list=[]
gf = '_lgf' # _lgf OR _hgf or <blank>
index = 600 #snapshot age in Myr
number = 1 # number of snapshots desired
gap = 10 # gaps (in Myr) desired between each snapshot
#--------------------------------------------------------------
for i in range(index, index + number*gap, gap):
    print 'Start '+str(i)
    fn1 = 'DD0'+str(i)
    fn2 = fn1 + gf
    list.append(fn2)
    #out=subprocess.call(["ls", "-lh"])
    if not os.path.exists('/Users/acharyya/models/simulation_output/'+fn2):
        fntar = fn1 + '.tar.gz'
        out=subprocess.call(['curl', '-O', 'https://hub.yt/data/goldbaum2015b/feedback_20pc'+gf+'/simulation_outputs/'+fntar])
        print 'Downloaded'
        out=subprocess.call(['tar', 'xzf', fntar])
        print 'Unpacked'
        out=subprocess.call(['rm', fntar])
        print 'Deleted'
        out=subprocess.call(['mv', fn1, '/Users/acharyya/models/simulation_output/'+fn2])
        print 'Shifted'
    print 'End ' +str(i)
#print list
for l in list:
    subprocess.call(['python', 'filterstars.py', l])
    #out=subprocess.call(['rm', '-r', '/Users/acharyya/models/simulation_output/'+l+'/*'])
    subprocess.call(['python', 'vreshalpha.py', l])
    subprocess.call(['python', 'plot_vreshalpha.py', l])
print(str(int((time.time() - start_time1)/60))+' minutes '+str((time.time() - start_time1)%60)+' seconds')
subprocess.call('echo Done dhinchak in '+str((time.time() - start_time)/60)+' minutes | mail -s "hi fi alert" \
acharyyaayan@gmail.com',shell=True)