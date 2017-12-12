#!/usr/bin/env python

import numpy as np
import yt
import math
import matplotlib.pyplot as plt
import subprocess
import sys

sys.argv = sys.argv[1:]
# fn = sys.argv[1]
for fn in sys.argv:
    # print fn
    ds = yt.load('/Users/acharyya/models/simulation_output/' + fn + '/' + fn.split()[0][0:6])
    dd = ds.all_data()


    def young_stars(pfilter, dd):
        # filter = np.logical_and(np.logical_and(dd[pfilter.filtered_type, 'particle_type'] == 2, dd[pfilter.filtered_type, \
        # 'creation_time'] > 0), dd[pfilter.filtered_type, 'age'].in_units('Myr') <= 5)
        filter = np.logical_and(dd[pfilter.filtered_type, 'creation_time'] > 0,
                                dd[pfilter.filtered_type, 'age'].in_units \
                                    ('Myr') <= 5)
        return filter


    yt.add_particle_filter('young_stars', function=young_stars, filtered_type='all', requires=['age', \
                                                                                               'creation_time'])
    ds.add_particle_filter('young_stars')
    print 'filtered, now extracting parameters...'  ###
    xg = dd['young_stars', 'particle_position_x']

    # x = xg.in_units('kpc')
    yg = dd['young_stars', 'particle_position_y']
    # y = yg.in_units('kpc')
    zg = dd['young_stars', 'particle_position_z']
    # z = zg.in_units('kpc')
    '''
    vz = dd['young_stars','particle_velocity_z']
    vz = vz.in_units('km/s')
    a = dd['young_stars', 'age']
    a = a.in_units('Myr')
    m = dd['young_stars','particle_mass']
    m = m.in_units('Msun')
    coord = np.zeros((len(xg),3))
    print 'extracting more parameters...' ###
    for i, v in enumerate(xg):
        coord[i,:]=(xg[i],yg[i],zg[i])
    print 'extracting even more parameters...' ###
    p = ds.find_field_values_at_points([('gas','pressure')],coord)
    d = ds.find_field_values_at_points([('gas','density')],coord)
    T = ds.find_field_values_at_points([('gas','temperature')],coord)
    Z = ds.find_field_values_at_points([('gas','metallicity')],coord)
    print 'writing to file...' ###
    fout=open('/Users/acharyya/models/paramlist/param_list_'+fn, 'w')
    for i, v in enumerate(xg):
        s=str(i)+' '+str(xg[i].value)+' '+str(yg[i].value)+' '+str(zg[i].value)+' '+str(m[i].value)+' '+str(a[i].value)+\
        ' '+str(vz[i].value)+' '+str(d[i].value)+' '+str(p[i].value)+' '+str(T[i].value)+' '+str(Z[i].value)+'\n'
        #print str(d[i])+' '+str(p[i])+' '+str(T[i])+' '+str(Z[i])
        fout.write(s)
    fout.close()
    '''
    print 'Sorted formed young stars in ' + fn + ' ...\n'
