#test python code to check distribution across cores in MPI
#Ayan, March 2018

import numpy as np
import sys
import argparse as ap
parser = ap.ArgumentParser(description="test")
parser.add_argument('--mode')
parser.add_argument('--size')
parser.add_argument('--ncpus')
args, leftovers = parser.parse_known_args()
if args.mode is None: args.mode = 'old'
if args.size is None: args.size = 581
if args.ncpus is None: args.ncpus = 4

nslice = int(args.size)
ncores = int(args.ncpus)
count = 0
for rank in range(ncores):
    if args.mode == 'new':
    # domain decomposition
        split_at_cpu = nslice - ncores*(nslice/ncores)
        nper_cpu1 = (nslice / ncores)
        nper_cpu2 = nper_cpu1 + 1
        if rank < split_at_cpu:
            core_start = rank * nper_cpu2
            core_end = (rank+1) * nper_cpu2 - 1
        else:
            core_start = split_at_cpu * nper_cpu2 + (rank - split_at_cpu) * nper_cpu1
            core_end = split_at_cpu * nper_cpu2 + (rank - split_at_cpu + 1) * nper_cpu1 - 1
    elif args.mode == 'old':
        # domain decomposition
        core_start = rank * (nslice / ncores);
        core_end   = (rank+1) * (nslice / ncores) - 1;
        # last PE gets the rest
        if (rank == ncores-1): core_end = nslice-1;
    nshare = core_end - core_start + 1
    count += nshare
    print "["+str(rank)+"]core_start = "+str(core_start)+", core_end = "+str(core_end)+" : total = "+str(nshare)
print '\ntotal jobs given =', nslice, 'total jobs dispatched =', count