#!/usr/bin/env python
# Christoph Federrath 2016

from mpi4py import MPI
import numpy as np
import datetime

def print_mpi(string):
    comm = MPI.COMM_WORLD
    print "["+str(comm.rank)+"] "+string

def print_master(string):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print "["+str(comm.rank)+"] "+string

# subroutine that return time since 1970 in seconds to high precision
def TimestampSec():
    return (datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()


#### MAIN ####

start_time = TimestampSec()

# get MPI ranks
comm = MPI.COMM_WORLD
nPE = comm.size
myPE = comm.rank
print_master("Total number of MPI ranks = "+str(nPE))
comm.Barrier()

# define n and local arrays
n = long(1e7)
sum_local = np.array(0.0)
sum_global = np.array(0.0)

# domain decomposition
my_start = myPE * (n / nPE);
my_end   = (myPE+1) * (n / nPE) - 1;
# last PE gets the rest
if (myPE == nPE-1): my_end = n;
print_mpi("my_start = "+str(my_start)+", my_end = "+str(my_end))

# loop over local chunk of loop
for i in range(my_start, my_end+1):
    sum_local += i

# MPI collective communication (all reduce)
comm.Allreduce(sum_local, sum_global, op=MPI.SUM)
print_master("sum_global = "+str(sum_global))

end_time = TimestampSec()
print_master("Runtime: "+str(end_time-start_time)+"s")