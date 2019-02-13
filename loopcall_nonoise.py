#python convenience routine to call stuff from plotobservables.py/getppv.py in a loop for no noise case, to automate things
#by Ayan
import subprocess
import sys
import os
HOME = os.getenv('HOME')+'/'
sys.path.append(HOME+'Work/astro/ayan_codes/enzo_model_code/')
import loopcall_fixednoise as lf
import time
start_time= time.time()
import argparse as ap
parser = ap.ArgumentParser(description="calling plotobservables for full parameter space")

parser.add_argument('--machine')
parser.add_argument('--multi')
parser.add_argument('--gf')
parser.add_argument('--runbig', dest='runbig', action='store_true')
parser.set_defaults(runbig=False)
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
args, leftovers = parser.parse_known_args()
if args.machine is None: args.machine = 'raijin'
if args.gf is None: gf = 'LG' # gas fraction; LG = low gf, FG = fiducial gf
else: gf = args.gf
if args.multi is None: multi = 1
else: multi = int(args.multi)

if args.test:
    print '---------this is for test purpose only------------------------------'
    print 'real= NO-NOISE'
    print 'machine=', args.machine
    print 'outpath=', lf.outpath
    print 'starting res=', lf.res
    print 'ncores=', lf.ncores
    print 'whichmpi=', lf.which_mpirun
    print 'function=', lf.function
    print 'Om_arr=', lf.Om_arr
    print 'logOHgrad_arr=', lf.logOHgrad_arr
    print 'file_arr=', lf.file_arr
    print 'is runbig?', args.runbig
    print 'arc_arr=', lf.arc_arr
    print 'binto=', lf.binto
    print 'redshift=', lf.z
    print '\n'
else:
    for Om in lf.Om_arr:
        for file in lf.file_arr:
            for logOHgrad in lf.logOHgrad_arr:
                for arc in lf.arc_arr:
                    # ----------to resume from where the job last stopped---------
                    queue = 'large' if args.runbig else 'small'
                    track_job_file = '/'.join(lf.outpath.split('/')[:-2]) + '/trackjob_gf' + gf + '_' + queue + '_real0.txt'
                    line = str(Om) + '\t\t' + file + '\t\t' + str(logOHgrad) + '\t\t' + str(arc) + '\n'
                    if os.path.exists(track_job_file) and line in open(track_job_file, 'r').readlines():
                        continue  # skip the set of parameters if it has already been done before
                    # --------------to run the job-------------------------
                    outfile = lf.outpath + 'output_PPV_' + file + 'Om=' + str(Om) + '_arc=' + str(arc) + '_specsmeared_' + \
                              str(lf.vres) + 'kmps_smeared_moff_parm6.0,7.5,4.7,37_binto'+str(lf.binto)+'_obs_real' + \
                              str(multi) + '_Zgrad' + str(lf.logOHcen) + ',' + str(logOHgrad) + lf.mergespec_text.replace(' ', '') + '.txt'

                    command = 'python '+lf.function+' --contsub --smooth --spec_smear --res '+str(lf.res)+' --Om ' + str(Om) + ' --binto '+\
                              str(lf.binto)+' --vres '+str(lf.vres)+' --Zgrad '+str(lf.logOHcen)+','+str(logOHgrad) + ' --met \
                            --calc --hide --scale_exptime 1200000 --arc '+str(arc)+' --file '+file+' --z '+str(lf.z)+' --ncores '\
                              +str(lf.ncores)+' --multi_realisation 1 --path '+lf.outpath+' --which mpirun ' + lf.which_mpirun +\
                        lf.mergespec_text + ' --toscreen 1>'+outfile
                    subprocess.call([command], shell=True)
                    # ----------for tracking how far the job has progressed---------
                    if not os.path.exists(track_job_file):
                        head = '#File to store information of how far a job has progressed, so as to restart from there\n\
                         #by Ayan\n\
                        Om      file        logOHgrad       arc     \n'
                        open(track_job_file, 'w').write(head)
                    open(track_job_file, 'w').write(line)

print 'Finished looping in %s minutes!'% ((time.time() - start_time)/60)
