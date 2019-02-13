# ---python code to filter out young (<5 Myr) star particles and their properties from Enzo simulation-----
# -----by Ayan, last modified Feb 2019-------
# -----  example usage:
# ------ ipython> run filterstars.py --file DD0600_lgf

import numpy as np
import yt
import time
start_time = time.time()
import argparse as ap
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
import os
HOME = os.getenv('HOME')
import get_radii as gr

# -------------filter function--------------
def young_stars(pfilter, dd):
    # filter = np.logical_and(np.logical_and(dd[pfilter.filtered_type, 'particle_type'] == 2, dd[pfilter.filtered_type, \
    # 'creation_time'] > 0), dd[pfilter.filtered_type, 'age'].in_units('Myr') <= 5)
    #filter = dd[pfilter.filtered_type, 'creation_time'] > 0 # for ALL stars
    filter = np.logical_and(dd[pfilter.filtered_type, 'creation_time'] > 0,
                            dd[pfilter.filtered_type, 'age'].in_units('Myr') <= 5)
    return filter

if __name__ == '__main__':
    # -------------------arguments parsed: MOST args are not used in here, they are for subsequent function calls-----------------------------
    parser = ap.ArgumentParser(description="dummy")
    parser.add_argument('--write_file', dest='write_file', action='store_true')
    parser.set_defaults(write_file=False)
    parser.add_argument('--plot_metgrad', dest='plot_metgrad', action='store_true')
    parser.set_defaults(plot_metgrad=False)
    parser.add_argument('--plot_Zinout', dest='plot_Zinout', action='store_true')
    parser.set_defaults(plot_Zinout=False)
    parser.add_argument('--plot_hist', dest='plot_hist', action='store_true')
    parser.set_defaults(plot_hist=False)
    parser.add_argument('--plot_lummap', dest='plot_lummap', action='store_true')
    parser.set_defaults(plot_lummap=False)
    parser.add_argument('--allstars', dest='allstars', action='store_true')
    parser.set_defaults(allstars=False)
    parser.add_argument('--nooutliers', dest='nooutliers', action='store_true')
    parser.set_defaults(nooutliers=False)
    parser.add_argument('--keep', dest='keep', action='store_true')
    parser.set_defaults(keep=False)
    parser.add_argument('--saveplot', dest='saveplot', action='store_true')
    parser.set_defaults(saveplot=False)
    parser.add_argument('--plotmap', dest='plotmap', action='store_true')
    parser.set_defaults(plotmap=False)
    parser.add_argument('--clobber', dest='clobber', action='store_true')
    parser.set_defaults(clobber=False)
    parser.add_argument("--file")
    parser.add_argument("--diag")
    parser.add_argument("--Om")
    parser.add_argument("--logOHgrad")
    parser.add_argument("--fontsize")
    parser.add_argument("--outpath")
    parser.add_argument("--mergeHII")
    args, leftovers = parser.parse_known_args()

    if args.file is not None: fn = args.file
    else: fn = 'DD0600_lgf'

    # ----------------------------------------------------------------------------------------
    outfilename = HOME+'/models/paramlist/param_list_'+fn
    if not os.path.exists(outfilename) or args.clobber:
        ds = yt.load(HOME+'/models/simulation_output/' + fn + '/' + fn[:6])
        dd = ds.all_data()

        if args.plotmap:
            # ---------to plot the original simulation projected density map------------------
            p = yt.visualization.plot_window.ProjectionPlot(ds, 'z', ('gas','density'), width=(20, 'kpc'), fontsize=40)
            p.set_unit('density', 'Msun/pc**2')
            p.set_zlim('density', 10**0., 3*10**2)
            p.save(HOME+'/Dropbox/papers/enzo_paper/Figs/' + fn + '_gas_density_map.eps')

        yt.add_particle_filter('young_stars', function=young_stars, filtered_type='all', requires=['age', \
                                                                                                   'creation_time'])
        ds.add_particle_filter('young_stars')

        print 'Filtered stars, now extracting parameters...'  ###
        xg = dd['young_stars', 'particle_position_x']
        x = xg.in_units('kpc')
        yg = dd['young_stars', 'particle_position_y']
        y = yg.in_units('kpc')
        zg = dd['young_stars', 'particle_position_z']
        z = zg.in_units('kpc')

        vz = dd['young_stars','particle_velocity_z']
        vz = vz.in_units('km/s')
        a = dd['young_stars', 'age']
        a = a.in_units('Myr')
        m = dd['young_stars','particle_mass']
        m = m.in_units('Msun')

        print 'Extracting more parameters...' ###
        coord = np.vstack([xg, yg, zg]).transpose()
        p = ds.find_field_values_at_points([('gas','pressure')],coord)
        d = ds.find_field_values_at_points([('gas','density')],coord) #ambient gas density only at point where young stars are located
        T = ds.find_field_values_at_points([('gas','temperature')],coord)
        Z = ds.find_field_values_at_points([('gas','metallicity')],coord)

        paramlist = pd.DataFrame({'x':x, 'y':y, 'z':z, 'vel_z':vz, 'age':a, 'mass':m, 'gas_density':d, 'gas_P':p, 'gas_T':T, 'gas_Z':Z})
        header = 'Units for the following columns: \n\
        x, y, z: kpc \n\
        vel_z: km/s \n\
        age: Myr \n\
        mass: Msun \n\
        gas_density in a cell: simulation units \n\
        gas_P in a cell: simulation units \n\
        gas_T in a cell: simulation units \n\
        gas_Z in a cell: simulation units'
        np.savetxt(outfilename, [], header=header, comments='#')
        paramlist.to_csv(outfilename, sep='\t', mode='a', index=None)
        print 'Saved file at', outfilename
    else:
        print 'Reading from existing file', outfilename
        paramlist = pd.read_table(outfilename, delim_whitespace=True, comment='#')

    print fn + ' completed in %s minutes' % ((time.time() - start_time) / 60)
    paramlist = gr.get_radii_for_df(paramlist, args)
