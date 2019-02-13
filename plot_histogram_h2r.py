#------Code to compute/plot HII region statistics of Enzo sims, including merging------#
#------by Ayan, Feb 2019-------#

import numpy as np
import subprocess
from matplotlib import pyplot as plt
import pylab
import os
HOME = os.getenv('HOME')
WD = '/Work/astro/ayan_codes/enzo_model_code/'
import sys
sys.path.append(HOME + WD)
import plotobservables as po
import argparse as ap
import matplotlib.cm as cm
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
import get_radii as gr
from copy import deepcopy

# --------Function to save figure----------
def saveplot(fig, outfile, silent=False):
    fig.savefig(outfile)
    if not silent: print 'Saved plot at', outfile
    return 0

# -------Function to plot color coded 2D map-------
def plot_2d(data, colorbycol=None, colorlabel=None, sizebycol=None, xcol='x', ycol='y', xlabel='x (pc)', ylabel='y (pc)', lim=None, figsize=(8,6), args=None, outfile='test.eps'):
    fig = plt.figure(figsize=figsize)
    if colorbycol is None: color = ['k']*len(data)
    elif args.plot_truesize:
        norm = plt.Normalize()
        color = cm.rainbow(norm(data[colorbycol]))
    else: color = data[colorbycol]

    if sizebycol is None: size = [20]*len(data)
    elif args.plot_truesize: size = data[sizebycol] / 1e3 # kpc
    else: size = data[sizebycol]**2 # since size denotes the area, hence quadratically related to r_m(pc)
    if lim is not None: lim = (-10, 10) # kpc

    ax = plt.gca()
    nbins = (lim[1] - lim[0])/args.res if args.plot_truesize else 20
    for item in np.linspace(lim[0], lim[1], nbins):
        ax.axvline(item + args.center, linestyle='dotted', c='k', lw=-0.5, alpha=0.3)
        ax.axhline(item + args.center, linestyle='dotted', c='k', lw=-0.5, alpha=0.3)

    if args.plot_truesize:
        print 'Plotting circles with true size..will take a few seconds..\n'
        for (x, y, r, c) in zip(data[xcol], data[ycol], size, color):
            circle = pylab.Circle((x,y), radius=r, alpha=0.5, color=c, linewidth=0)
            ax.add_patch(circle)
    else:
        plt.scatter(data[xcol], data[ycol], c=color, s=size, lw=0, alpha=0.5)
        if colorbycol is not None:
            cbar = plt.colorbar()
            cbar.set_label(colorlabel)

    ax.set_xticks([(item + args.center) for item in np.linspace(lim[0], lim[1], 5)])
    ax.set_yticks([(item + args.center) for item in np.linspace(lim[0], lim[1], 5)])
    ax.set_xticklabels(['%.2F' %(i - args.center) for i in list(ax.get_xticks())])
    ax.set_yticklabels(['%.2F' %(i - args.center) for i in list(ax.get_yticks())])
    plt.xlim((item + args.center for item in lim))
    plt.ylim((item + args.center for item in lim))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if args.saveplot: saveplot(fig, outfile)
    if not args.hide: plt.show(block=False)
    else: plt.close(fig)
    return fig

# -------Function to plot standardised histogram-------
def plot_histogram(data_arr, nbins=50, histtype='step', color_arr=None, color='k', label_arr=None, xlabel='Data', ylabel='Frequency', outfile='test.eps', figsize=(8,6), xlim=None, args=None):
    fig = plt.figure(figsize=figsize)
    if np.shape(data_arr)[0] > 10: # if data is for single histogram, else plotting multiple histograms
        color_arr = [color]
        data_arr = [data_arr]
        label_arr = [None]

    for index, data in enumerate(data_arr):
        data = data[np.isfinite(data)] # to get rid of NaNs and infs, if any
        histogram = plt.hist(data, histtype=histtype, bins=nbins, color=color_arr[index], label=label_arr[index])

    if xlim is not None: plt.xlim(xlim)
    else: plt.xlim(np.min(histogram[1])*0.1, np.max(histogram[1])*1.1)
    if np.shape(data_arr)[0] <= 10: plt.legend(loc='best')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if args.saveplot: saveplot(fig, outfile)
    if not args.hide: plt.show(block=False)
    else: plt.close(fig)
    return fig

if __name__ == '__main__':
    # -------------------arguments parsed-------------------------------------------------------
    parser = ap.ArgumentParser(description="histogram plotting tool")
    parser.add_argument('--nooutlier', dest='nooutlier', action='store_true')
    parser.set_defaults(nooutlier=False)
    parser.add_argument('--saveplot', dest='saveplot', action='store_true')
    parser.set_defaults(saveplot=False)
    parser.add_argument('--hide', dest='hide', action='store_true')
    parser.set_defaults(hide=False)
    parser.add_argument('--keep', dest='keep', action='store_true')
    parser.set_defaults(keep=False)
    parser.add_argument('--plot_truesize', dest='plot_truesize', action='store_true')
    parser.set_defaults(plot_truesize=False)
    parser.add_argument('--write_file', dest='write_file', action='store_true')
    parser.set_defaults(write_file=False)
    parser.add_argument('--clobber', dest='clobber', action='store_true')
    parser.set_defaults(clobber=False)
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

    parser.add_argument("--path")
    parser.add_argument("--galsize")
    parser.add_argument("--center")
    parser.add_argument("--outtag")
    parser.add_argument("--file")
    parser.add_argument("--Om")
    parser.add_argument("--res")
    parser.add_argument("--arc")
    parser.add_argument("--Zgrad")
    parser.add_argument("--mergeHII")
    parser.add_argument("--diag")
    parser.add_argument("--fontsize")
    parser.add_argument("--outpath")
    args, leftovers = parser.parse_known_args()

    # -------------------default arguments assigned-------------------------------------------------------
    if args.outtag is None: args.outtag = '_sph_logT4.0_MADtemp_Z0.05,5.0_age0.0,5.0_lnII5.0,12.0_lU-4.0,-1.0_4D'

    if args.Zgrad is not None:
        args.logOHcen, args.logOHgrad = [float(ar) for ar in args.Zgrad.split(',')]
        args.gradtext = '_Zgrad' + str(args.logOHcen) + ',' + str(args.logOHgrad)
    else:
        args.logOHcen, args.logOHgrad = po.logOHsun, 0.
        args.gradtext = ''
    args.outtag = args.gradtext + args.outtag

    if args.path is None: args.path = HOME + '/Desktop/bpt_contsub_contu_rms/'
    subprocess.call(['mkdir -p ' + args.path], shell=True)  # create output directory if it doesn't exist

    if args.file is None: args.file = 'DD0600_lgf'  # which simulation to use

    if args.Om is not None: args.Om = float(args.Om)
    else: args.Om = 0.5

    if args.res is not None: args.res = float(args.res)
    else: args.res = 0.04  # kpc: base resolution to constructPPV cube, usually (close to) the base resolution of simulation

    if args.mergeHII is not None: args.mergeHII = float(args.mergeHII) # kpc, within which if two HII regions are they'll be treated as merged

    args.galsize = 30. # kpc
    args.center = 0.5*1310.72022072 # kpc units, from Goldbaum simulations in cell units
    if not args.keep: plt.close('all')

    # --------------------Setting titles for plots------------------------------------------------------
    mergeHII_text = '_mergeHII='+str(args.mergeHII)+'kpc' if args.mergeHII is not None else ''
    title = args.file + mergeHII_text + '_Om' + str(args.Om)

    # --------------------Reading in the star cluster list------------------------------------------------------
    infile = po.getfn(args)
    if not os.path.exists(infile) and args.mergeHII is not None:
        dummy_args = deepcopy(args)
        dummy_args.mergeHII = None
        dummy_infile = po.getfn(dummy_args)
        dummylist = pd.read_table(dummy_infile, comment = '#', delim_whitespace = True)
        print 'Merged HII region parameter file did not exist. Creating now..'
        dummylist = gr.get_radii_for_df(dummylist, args)

    starlist = pd.read_table(infile, comment='#', delim_whitespace=True)
    initial_nh2r = len(starlist)
    if args.nooutlier:
        starlist = starlist[(starlist['nII'] >= 10 ** (5.2 - 4 + 6)) & (starlist['nII'] <= 10 ** (6.7 - 4 + 6))].reset_index(drop=True)  # D16 models have 5.2 < lpok < 6.7 #DD600_lgf with logOHgrad8.77,-0.1 has 4.3 < lpok < 8
        title += '_no_densityoutlier'
    print 'Using ' + str(len(starlist)) + ' out of ' + str(initial_nh2r) + ' HII regions'

    # -------------------Assigning H2R to grid at a given resolution------------------------------------------------------
    g = int(np.ceil(args.galsize / args.res))
    mass = np.zeros((g, g))
    number = np.zeros((g, g))
    for count, j in enumerate(range(len(starlist))):
        if count % (len(starlist)/6) == 0: print 'Particle ' + str(count + 1) + ' of ' + str(len(starlist))
        xind = int((starlist['x'][j] - args.center + args.galsize/2.) / args.res) # (starlist['x(kpc)'][j] - args.center) used to range from (-galsize/2, galsize/2) kpc, which is changed here to (0, galsize) kpc
        yind = int((starlist['y'][j] - args.center + args.galsize/2.) / args.res)
        mass[xind][yind] += starlist['mass'][count]
        number[xind][yind] += 1
    print '\n'

    # -------------------Plot outer radius histogram------------------------------------------------------
    rado_fig = plot_histogram(starlist['r'], xlabel='Outer radius (pc)', xlim=(0, 25), outfile=args.path + title + '_outer_radius_hist.eps', ylabel='Frequency', nbins=25, histtype='step', color='k', args=args)

    # -------------------Plot outer radius histogram------------------------------------------------------
    scatter_fig = plot_2d(starlist, colorbycol='mass', colorlabel='Mass (Msun)', sizebycol='r', xcol='x', ycol='y', xlabel='X (kpc)', ylabel='Y (kpc)', lim=(-10, 10), figsize=(8,6), args=args, outfile=args.path + title + '_2D.eps')
    '''
    # -------------------Plot Stromgen radius histogram------------------------------------------------------
    rads_fig = plot_histogram(starlist['r_Strom'], xlabel='Stromgen radius (pc)', xlim=(0, 25), outfile=args.path + title + '_stromgen_radius_hist.eps', ylabel='Frequency', nbins=25, histtype='step', color='k', args=args)
    
    # -------------------Plot density histogram------------------------------------------------------
    den_fig = plot_histogram(np.log10(starlist['nII']), xlabel='log(n_e)', xlim=(6, 10), outfile=args.path + title + '_density_hist.eps', ylabel='Frequency', nbins=10, histtype='step', color='k', args=args)

    # -------------------Plot logU histogram------------------------------------------------------
    ion_fig = plot_histogram(np.log10(starlist['<U>']), xlabel='log(<U>)', xlim=(-4, -1), outfile=args.path + title + '_logU_hist.eps', ylabel='Frequency', nbins=10, histtype='step', color='k', args=args)

    # -------------------Plot pressure histogram------------------------------------------------------
    lpok_fig = plot_histogram(starlist['log(P/k)'], xlabel='log(P/k)', xlim=(4, 8.5), outfile=args.path + title + '_lpok_hist.eps', ylabel='Frequency', nbins=25, histtype='step', color='k', args=args)

    # -------------------Plot CMF at given resolution------------------------------------------------------
    cmf_fig = plot_histogram(np.log10(mass.flatten()), xlabel='log (Cluster mass (Msun))', xlim=(2.4, 4.8), outfile=args.path + title + '_cmf_hist.eps', ylabel='Frequency', nbins=10, histtype='step', color='k', args=args)

    # -------------------Plot CMF at given resolution------------------------------------------------------
    num_fig = plot_histogram(np.log10(number.flatten()), xlabel='log (Number of HII region in each cell)', xlim=(0, 2.2), outfile=args.path + title + '_number_hist.eps', ylabel='Frequency', nbins=10, histtype='step', color='k', args=args)
    '''
    print 'Finished!'