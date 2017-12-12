#python routine to simply change the units of PPV cubes in a folder, by a factor
#-Ayan, May 2017

import os
HOME = os.getenv('HOME')+'/'
from astropy.io import fits
import sys
sys.path.append(HOME+'models/enzo_model_code/')
import numpy as np
import plotobservables as p

galsize = 26. #kpc
input_dir = '/avatar/acharyya/enzo_models/ppvcubes3/'
output_dir = '/avatar/acharyya/enzo_models/ppvcubes3b/'
list = os.listdir(input_dir)

for file in list:
    if file[-5:] != '.fits': continue
    data = fits.open(input_dir+file)[0].data
    g = np.shape(data)[0]
    data /= (galsize*1000./g)**2 #convert from ergs/s to ergs/s/pc^2
    p.write_fits(output_dir+file, data, fill_val=np.nan)

print 'Finished!'