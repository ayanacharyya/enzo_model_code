###### Select resources #####
#PBS -N Ayan_enzo_fixednoise_real1
#PBS -l select=3:ncpus=28:mpiprocs=28
#### Request exclusive use of nodes #####
#PBS -l place=scatter:excl
#PBS -l walltime=07:00:00
##### Queue #####
#PBS -q largemem
##### Mail Options #####
#PBS -m abe
#PBS -M u5877042@anu.edu.au
##### Change to current working directory #####
cd /avatar/acharyya/enzo_models
##### Execute Program #####
python ./loopcall_fixednoise.py 1 1>output_loopcall_fixednoise_real1.out 2>&1