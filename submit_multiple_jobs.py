#--python script to submit multiple jobs for different noise realisations---#
#--by Ayan Jan, 2018-----#
import datetime
import subprocess
import os
HOME = os.getenv('HOME')

jobscript_path = HOME+'/Work/astro/ayan_codes/enzo_model_code/'
workdir = '/avatar/acharyya/enzo_models'
jobscript_template = 'jobscript_template.txt'
name_root = '_fxdnoise_real'
callfile = jobscript_path + 'loopcall_fixednoise.py'
nnodes = '3'
ncores = '28'
nhours = '10'
start, stop = 1, 10 # start and stop realisation indices

for real in range(start,stop+1):
    jobname = name_root[1:]+str(real)
    out_jobscript = jobscript_path + 'jobscript'+name_root+str(real)+'.sh'
    outfile = 'output_loopcall'+name_root+str(real)+'.out'
    replacements = {'run_name': jobname, 'nnodes': nnodes, 'nhours': nhours, 'callfile': callfile, 'workdir': workdir,\
                    'real': str(real), 'outfile': outfile, 'ncores': ncores}  # keywords to be replaced in template jobscript

    with open(jobscript_path + jobscript_template) as infile, open(out_jobscript, 'w') as outfile:
        for line in infile:
            for src, target in replacements.iteritems():
                line = line.replace(src, target)
            outfile.write(line) # replacing and creating new jobscript file

    print 'Going to submit job '+jobname+' at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    subprocess.call(['qsub '+out_jobscript], shell=True)
    subprocess.call(['rm '+out_jobscript], shell=True) # removing newly created jobscript file to reduce clutter

print 'Submitted all '+str(stop - start + 1)+' jobs from real '+str(start)+' to '+str(stop)+'\n'