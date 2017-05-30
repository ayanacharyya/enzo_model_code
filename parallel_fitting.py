#--python utility code which takes a PPV cube as input, convolves its each spectral slice in PARALLEL
#--and returns the convolved cube
#--by Ayam, May, 2017
import time
import datetime
import os
HOME = os.getenv('HOME')
import sys
sys.path.append(HOME+'/Work/astro/mageproject/ayan/')
sys.path.append(HOME+'/models/enzo_model_code/')
import splot_util as su
import plotobservables as po
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma as gf2s
from scipy.optimize import curve_fit
from mpi4py import MPI
from astropy.io import fits
import argparse as ap
parser = ap.ArgumentParser(description="parallel cube fitting tool")
#-------------------------------------------------------------------------------------------
def print_mpi(string, outputfile=None):
    comm = MPI.COMM_WORLD
    if outputfile is None:
        print "["+str(comm.rank)+"] "+string
    else:
        ofile = open(outputfile, 'a')
        ofile.write("["+str(comm.rank)+"] "+string+"\n")
        ofile.close()

def print_master(string, outputfile=None):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        if outputfile is None:
            print "["+str(comm.rank)+"] "+string
        else:
            ofile = open(outputfile, 'a')
            ofile.write("["+str(comm.rank)+"] "+string+"\n")
            ofile.close()

#-------------Fucntion for fitting multiple lines----------------------------
def fit_all_lines(wlist, llist, wave, flux, resoln, pix_i, pix_j, nres=5, z=0, z_err=0.0001, silent=True, showplot=False, outputfile='junk.txt') :
    ofile = open(outputfile,'a')
    flam = flux/wave #converting flux to per wavelength unit (i.e. ergs/s/pc^2/A), before sending off to fitting routine
    wave, flam = np.array(wave), np.array(flam) #converting to numpy arrays
    #---to remove negative flux values----------
    idx = np.where(flam<0)[0]
    if len(idx) > 1:
        if idx[0] == 0: idx = idx[1:]
        if idx[-1] == len(flam)-1 : idx = idx[:-1]
        flam[idx]=(flam[idx-1] + flam[idx+1])/2. # replacing negative fluxes with average of nearest neighbours
    #-----------------------------------
    kk, count, flux_array, flux_error_array = 1, 0, [], []
    ndlambda_left, ndlambda_right = [nres]*2 #how many delta-lambda wide will the window (for line fitting) be on either side of the central wavelength, default 5
    try:
        count = 1
        first, last = [wlist[0]]*2
    except IndexError:
        pass
    while kk <= len(llist):
        center1 = last
        if kk == len(llist):
            center2 = 1e10 #insanely high number, required to plot last line
        else:
            center2 = wlist[kk]
        if center2*(1. - ndlambda_left/resoln) > center1*(1. + ndlambda_right/resoln):
            leftlim = first*(1.-ndlambda_left/resoln) 
            rightlim = last*(1.+ndlambda_right/resoln)
            wave_short = wave[(leftlim < wave) & (wave < rightlim)]
            flam_short = flam[(leftlim < wave) & (wave < rightlim)]
            if not silent: 
                ofile.write('Trying to fit '+str(llist[kk-count:kk])+' line/s at once. Total '+str(count)+'\n')
            try: 
                popt, pcov = fitline(wave_short, flam_short, wlist[kk-count:kk], resoln, z=z, z_err=z_err)
                if showplot:
                    plt.axvline(leftlim, linestyle='--',c='g')
                    plt.axvline(rightlim, linestyle='--',c='g')
                ndlambda_left, ndlambda_right = [nres]*2
                if not silent: ofile.write('Done this fitting!'+'\n')
            except TypeError, er:
                if not silent: ofile.write('Trying to re-do this fit with broadened wavelength window..\n')
                ndlambda_left+=1
                ndlambda_right+=1
                continue
            except (RuntimeError, ValueError), e:
                popt = np.zeros(count*3 + 1) #if could not fit the line/s fill popt with zeros so flux_array gets zeros
                pcov = np.zeros((count*3 + 1,count*3 + 1)) #if could not fit the line/s fill popt with zeros so flux_array gets zeros
                ofile.write('Could not fit lines '+str(llist[kk-count:kk])+' for pixel '+str(pix_i)+', '+str(pix_j)+'\n')
                pass
                
            for xx in range(0,count):
                #in popt for every bunch of lines, element 0 is the continuum(a)
                #and elements (1,2,3) or (4,5,6) etc. are the height(b), mean(c) and width(d)
                #so, for each line the elements (0,1,2,3) or (0,4,5,6) etc. make the full suite of (a,b,c,d) gaussian parameters
                #so, for each line, flux f (area under gaussian) = sqrt(2pi)*(b-a)*d
                #also the full covariance matrix pcov looks like:
                #|00 01 02 03 04 05 06 .....|
                #|10 11 12 13 14 15 16 .....|
                #|20 21 22 23 24 25 26 .....|
                #|30 31 32 33 34 35 36 .....|
                #|40 41 42 43 44 45 46 .....|
                #|50 51 52 53 54 55 56 .....|
                #|60 61 62 63 64 65 66 .....|
                #|.. .. .. .. .. .. .. .....|
                #|.. .. .. .. .. .. .. .....|
                #
                #where, 00 = var_00, 01 = var_01 and so on.. (var = sigma^2)
                #let var_aa = vaa (00), var_bb = vbb(11), var_ab = vab(01) = var_ba = vba(10) and so on..
                #for a single gaussian, f = const * (b-a)*d
                #i.e. sigma_f^2 = d^2*(saa^2 + sbb^2) + (b-a)^2*sdd^2 (dropping the constant for clarity of explanation)
                #i.e. var_f = d^2*(vaa + vbb) + (b-a)^2*vdd
                #the above holds if we assume covariance matrix to be diagonal (off diagonal terms=0) but thats not the case here
                #so we also need to include off diagnoal covariance terms while propagating flux errors
                #so now, for each line, var_f = d^2*(vaa + vbb) + (b-a)^2*vdd + 2d^2*vab + 2d*(b-a)*(vbd - vad)
                #i.e. in terms of element indices,
                #var_f = 3^2(00 + 11) + (1-0)^2*33 - (2)*3^2*10 + (2)*3*(1-0)*(13-03),
                #var_f = 6^2(00 + 44) + (4-0)^2*66 - (2)*6^2*40 + (2)*6*(4-0)*(46-06),
                #var_f = 9^2(00 + 77) + (1-0)^2*99 - (2)*9^2*70 + (2)*9*(7-0)*(79-09), etc.
                #
                popt_single= np.concatenate(([popt[0]],popt[3*xx+1:3*(xx+1)+1]))               
                flux = np.sqrt(2*np.pi)*(popt_single[1] - popt_single[0])*popt_single[3] #total flux = integral of guassian fit ; resulting flux in ergs/s/pc^2 units
                flux_array.append(flux)
                flux_error = np.sqrt(2*np.pi*(popt_single[3]**2*(pcov[0][0] + pcov[3*xx+1][3*xx+1])\
                + (popt_single[1]-popt_single[0])**2*pcov[3*(xx+1)][3*(xx+1)]\
                - 2*popt_single[3]**2*pcov[3*xx+1][0]\
                + 2*(popt_single[1] - popt_single[0])*popt_single[3]*(pcov[3*xx+1][3*(xx+1)] - pcov[0][3*(xx+1)])\
                )) # var_f = 3^2(00 + 11) + (1-0)^2*33 - (2)*3^2*10 + (2)*3*(1-0)*(13-03)
                flux_error_array.append(flux_error)
                if showplot:
                    leftlim = popt_single[2]*(1.-nres/resoln) 
                    rightlim = popt_single[2]*(1.+nres/resoln)
                    wave_short_single = wave[(leftlim < wave) & (wave < rightlim)]
                    plt.plot(wave_short_single, np.log10(su.gaus(wave_short_single,1, *popt_single)),lw=1, c='r')
            if showplot:
                if count >1: plt.plot(wave_short, np.log10(su.gaus(wave_short, count, *popt)),lw=2, c='g')                   
                plt.draw()
                        
            first, last = [center2]*2
            count = 1
        else:
            last = center2
            count += 1
        kk += 1
    #-------------------------------------------------------------------------------------------
    flux_array = np.array(flux_array)
    flux_error_array = np.array(flux_error_array)
    flux_array[flux_array<1.] = 0. #filtering out obvious non-detections and setting those fluxes to 0
    ofile.close()
    return flux_array, flux_error_array
#-------------------------------------------------------------------------------------------
def fitline(wave, flam, wtofit, resoln, z=0, z_err=0.0001):
    v_maxwidth = 10*po.c/resoln #10*vres in km/s
    z_allow = 3*z_err #wavelengths are at restframe; assumed error in redshift
    p_init, lbound, ubound = [np.abs(flam[0])],[0.],[np.inf]
    for xx in range(0, len(wtofit)):
        fl = np.max(flam) #flam[(np.abs(wave - wtofit[xx])).argmin()] #flam[np.where(wave <= wtofit[xx])[0][0]]
        p_init = np.append(p_init, [fl-flam[0], wtofit[xx], wtofit[xx]*2.*gf2s/resoln])
        lbound = np.append(lbound,[0., wtofit[xx]*(1.-z_allow/(1.+z)), wtofit[xx]*1.*gf2s/resoln])
        ubound = np.append(ubound,[np.inf, wtofit[xx]*(1.+z_allow/(1.+z)), wtofit[xx]*v_maxwidth*gf2s/po.c])
    popt, pcov = curve_fit(lambda x, *p: su.gaus(x, len(wtofit), *p),wave,flam,p0= p_init, max_nfev=10000, bounds = (lbound, ubound))
    return popt, pcov
#-------------------------------------------------------------------------------------------
#def makemapcube(ppv, dispsol, wlist, llist, vres, outputfile='junk.txt', parallel=True):
if __name__ == '__main__':
    parser.add_argument('--silent', dest='silent', action='store_true')
    parser.set_defaults(silent=False)
    parser.add_argument('--spec_smear', dest='spec_smear', action='store_true')
    parser.set_defaults(spec_smear=False)
    parser.add_argument('--vdel')
    parser.add_argument('--vdisp')
    parser.add_argument('--vres')
    parser.add_argument('--nhr')
    parser.add_argument('--wmin')
    parser.add_argument('--wmax')
    parser.add_argument('--fitsname')
    parser.add_argument('--fittedcube')
    parser.add_argument('--fittederror')
    parser.add_argument('--outputfile')
    args, leftovers = parser.parse_known_args()
    vdel = float(args.vdel)
    vdisp = float(args.vdisp)
    vres = float(args.vres)
    nhr = int(args.nhr)
    if args.wmin !=  'None':
        wmin = float(args.wmin)
    else:
        wmin = None #Angstrom; starting wavelength of PPV cube

    if args.wmax !=  'None':
        wmax = float(args.wmax)
    else:
        wmax = None #Angstrom; ending wavelength of PPV cube
    outputfile = args.outputfile
    
    w, dummy2, dummy3, new_w, dummy4, wlist, llist, dummy7 = po.get_disp_array(vdel, vdisp, vres, nhr, wmin=wmin ,wmax=wmax, spec_smear=args.spec_smear)
    if args.spec_smear: dispsol = new_w[1:]
    else: dispsol = w
    ppv = fits.open(args.fitsname)[0].data
    x = np.shape(ppv)[0]
    y = np.shape(ppv)[1]
    ncells = x*y
    mapcube = np.zeros((x,y, len(wlist)))
    errorcube = np.zeros((x,y, len(wlist)))
    resoln = po.c/vres
    
    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = '+str(ncores)+'. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), outputfile=outputfile)
    mapcube_local = np.zeros(np.shape(mapcube))
    errorcube_local = np.zeros(np.shape(errorcube))
    comm.Barrier()
    t_start = MPI.Wtime() ### Start stopwatch ###

    core_start = rank * (ncells/ncores)
    core_end   = (rank+1) * (ncells/ncores)       
    if (rank == ncores-1): core_end = ncells # last PE gets the rest
    for k in range(core_start, core_end):
        i, j = k/x, k%x
        mapcube_local[i,j,:], errorcube_local[i,j,:] = fit_all_lines(wlist, llist, dispsol, ppv[i,j,:], resoln, i, j, nres=5, z=0, z_err=0.0001, silent=True, outputfile=outputfile)
        if args.silent: print_mpi('Fitted cell '+str(k)+' i.e. cell '+str(i)+','+str(j)+' of '+str(ncells)+' cells', outputfile=outputfile)
        else: print_mpi('Fitted cell '+str(k)+' i.e. cell '+str(i)+','+str(j)+' of '+str(ncells)+' cells')
    comm.Barrier()
    comm.Allreduce(mapcube_local, mapcube, op=MPI.SUM)
    comm.Allreduce(errorcube_local, errorcube, op=MPI.SUM)
    if rank ==0:
        po.write_fits(args.fittedcube, mapcube, fill_val=np.nan, outputfile=outputfile)        
        po.write_fits(args.fittederror, errorcube, fill_val=np.nan, outputfile=outputfile)    
            
    t_diff = MPI.Wtime()-t_start ### Stop stopwatch ###
    print_master('deb14: parallely: time taken for fitting of '+str(ncells)+' cells with '+str(ncores)+' cores= '+ str(t_diff/60.)+' min', outputfile=outputfile)
