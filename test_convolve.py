# python routine to test convolution and rebinning
# by Ayan, Oct 2017

import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy import convolution as con
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate, ndimage, signal
import warnings

warnings.filterwarnings("ignore")
from astropy.stats import gaussian_fwhm_to_sigma as gf2s
import sys
import time

# -----------------------------------------------------------------
res_phys, fwhm = 2., 6  # fwhm to convolve in pixels and physical resolution in kpc
fwhm_sample = 4  # final pix per beam to be sampled with (by congrid)
keep, cmax, cmin = 0, -23, -16


# -------------------------------------------------------------------------------------------
def congrid(a, newdims, method='linear', centre=False, minusone=True):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array(a.shape)
    ndims = len(a.shape)
    if len(newdims) != ndims:
        print "[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions."
        return None
    newdims = np.asarray(newdims, dtype=float)
    dimlist = []

    if method == 'neighbour':
        for i in range(ndims):
            base = np.indices(newdims)[i]
            dimlist.append((old[i] - m1) / (newdims[i] - m1) \
                           * (base + ofs) - ofs)
        cd = np.array(dimlist).round().astype(int)
        newa = a[list(cd)]

    elif method in ['nearest', 'linear']:
        # calculate new dims
        for i in range(ndims):
            base = np.arange(newdims[i])
            dimlist.append((old[i] - m1) / (newdims[i] - m1) \
                           * (base + ofs) - ofs)
        # specify old dims
        olddims = [np.arange(i, dtype=np.float) for i in list(a.shape)]

        # first interpolation - for ndims = any
        mint = interpolate.interp1d(olddims[-1], a, kind=method)
        newa = mint(dimlist[-1])

        trorder = [ndims - 1] + range(ndims - 1)
        for i in range(ndims - 2, -1, -1):
            newa = newa.transpose(trorder)

            mint = interpolate.interp1d(olddims[i], newa, kind=method)
            newa = mint(dimlist[i])

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose(trorder)

    elif method in ['spline']:
        oslices = [slice(0, j) for j in old]
        oldcoords = np.ogrid[oslices]
        nslices = [slice(0, j) for j in list(newdims)]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        # make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = ndimage.map_coordinates(a, newcoords)
    else:
        print "Congrid error: Unrecognized interpolation type.\n", \
            "Currently only \'neighbour\', \'nearest\',\'linear\',", \
            "and \'spline\' are supported."
        return None
    newa *= old[0] * old[1] / (newdims[0] * newdims[1])  # Ayan's ammendment; to keep total flux conserved.
    return newa


# -----------------------------------------------------------------
def rebin(array, dimensions=None, scale=None):
    """ Return the array ``array`` to the new ``dimensions`` conserving flux the flux in the bins
    The sum of the array will remain the same
 
    >>> ar = numpy.array([
        [0,1,2],
        [1,2,3],
        [2,3,4]
        ])
    >>> rebin(ar, (2,2))
    array([
        [1.5, 4.5]
        [4.5, 7.5]
        ])
    Raises
    ------
 
    AssertionError
        If the totals of the input and result array don't agree, raise an error because computation may have gone wrong
 
    Reference
    =========
    +-+-+-+
    |1|2|3|
    +-+-+-+
    |4|5|6|
    +-+-+-+
    |7|8|9|
    +-+-+-+
    """
    if dimensions is not None:
        if isinstance(dimensions, float):
            dimensions = [int(dimensions)] * len(array.shape)
        elif isinstance(dimensions, int):
            dimensions = [dimensions] * len(array.shape)
        elif len(dimensions) != len(array.shape):
            raise RuntimeError('')
    elif scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            dimensions = map(int, map(round, map(lambda x: x * scale, array.shape)))
        elif len(scale) != len(array.shape):
            raise RuntimeError('')
    else:
        raise RuntimeError('Incorrect parameters to rebin.\n\trebin(array, dimensions=(x,y))\n\trebin(array, scale=a')
    import itertools
    # dY, dX = map(divmod, map(float, array.shape), dimensions)
    if np.shape(array) == dimensions: return array  # no rebinning actually needed
    result = np.zeros(dimensions)
    for j, i in itertools.product(*map(xrange, array.shape)):
        (J, dj), (I, di) = divmod(j * dimensions[0], array.shape[0]), divmod(i * dimensions[1], array.shape[1])
        (J1, dj1), (I1, di1) = divmod(j + 1, array.shape[0] / float(dimensions[0])), divmod(i + 1,
                                                                                            array.shape[1] / float(
                                                                                                dimensions[1]))

        # Moving to new bin
        # Is this a discrete bin?
        dx, dy = 0, 0
        if (I1 - I == 0) | ((I1 - I == 1) & (di1 == 0)):
            dx = 1
        else:
            dx = 1 - di1
        if (J1 - J == 0) | ((J1 - J == 1) & (dj1 == 0)):
            dy = 1
        else:
            dy = 1 - dj1
        # Prevent it from allocating outide the array
        I_ = min(dimensions[1] - 1, I + 1)
        J_ = min(dimensions[0] - 1, J + 1)
        result[J, I] += array[j, i] * dx * dy
        result[J_, I] += array[j, i] * (1 - dy) * dx
        result[J, I_] += array[j, i] * dy * (1 - dx)
        result[J_, I_] += array[j, i] * (1 - dx) * (1 - dy)
    allowError = 0.1
    assert (array.sum() < result.sum() * (1 + allowError)) & (array.sum() > result.sum() * (1 - allowError))
    return result


# -----------------------------------------------------------------
def plotmap(map, cmin=None, cmax=None, title='', makelog=True):
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.1, left=0.1, right=0.9)
    ax = plt.subplot(111)

    if cmin is None:
        cmin = np.min(np.ma.compressed(np.ma.masked_where((map <= 0) | (~np.isfinite(map)), map)))
        if makelog: cmin = np.log10(cmin)
    if cmax is None:
        cmax = np.max(np.ma.compressed(np.ma.masked_where((map <= 0) | (~np.isfinite(map)), map)))
        if makelog: cmax = np.log10(cmax)
    if makelog: map = np.log10(map)

    p = ax.imshow(np.transpose(map), cmap='rainbow', vmin=cmin, vmax=cmax)
    fps = galsize / np.shape(map)[0]  # final pixel size
    ax.set_xticklabels([i * fps - 13 - 2 for i in list(ax.get_xticks())])
    ax.set_yticklabels([i * fps - 13 - 2 for i in list(ax.get_yticks())])
    plt.ylabel('y(kpc)')
    plt.xlabel('x(kpc)')
    plt.title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(p, cax=cax).set_label('log(units)' if makelog else 'units')
    plt.show(block=False)


# -----------------------------------------------------------------
def convolve(map, fwhm, pow=4.7, size=5, kernel='moff'):
    map = np.ma.filled(map, fill_value=np.nan)
    print 'convolve: map shape=', np.shape(map)
    if kernel is 'moff':
        sig = fwhm / (2 * np.sqrt(2 ** (1. / pow) - 1.))
    elif kernel is 'gauss':
        sig = gf2s * fwhm
    size = int(np.round(size * sig))
    if size % 2 == 0: size += 1
    print 'convolve: fwhm, sig, total size, phys_res=', fwhm, sig, size, fwhm * 26. / np.shape(map)[0]
    if kernel is 'moff':
        kernel = con.Moffat2DKernel(sig, pow, x_size=size, y_size=size)
    elif kernel is 'gauss':
        kernel = con.Gaussian2DKernel(sig, x_size=size, y_size=size)
    result = con.convolve_fft(map, kernel, normalize_kernel=True)
    median = np.median(np.ma.compressed(np.ma.masked_where(map <= 0, map)))
    result[np.log10(np.abs(median)) - np.log10(np.abs(
        result)) > 10.] = 0.  # discarding any resulting pixel that is more than O(10) lower than input data, to avoid round off error pixels
    return result


# -----------------------------------------------------------------
def get_dist(map, galsize=26.):
    (rows, cols) = np.shape(map)
    d_list = [np.sqrt((x * galsize / rows - galsize / 2. - 2) ** 2 + (y * galsize / cols - galsize / 2. - 2) ** 2) for x
              in xrange(rows) for y in xrange(cols)]
    return d_list


# -------------------------------------------------------------------------------------------
def mydiag(data, title=''):
    data = np.ma.masked_where(~np.isfinite(data), data)
    data = np.ma.masked_where(data <= 0, data)
    data = np.ma.compressed(data.flatten())
    print title + ': Len, mean, median, stdev, max, min = ', len(data), np.mean(data), np.median(data), np.max(
        data), np.min(data)


# -----------------------------------------------------------------
if __name__ == '__main__':
    galsize = 26.  # kpc
    MAP = fits.open('/Users/acharyya/Desktop/test_slice_253of581.fits')[0].data  # halpha
    initial = galsize / np.shape(MAP)[0]  # cell size in kpc
    MAP *= (initial * 1e3) ** 2  # to change input units ergs/s/pc^2 to ergs/s
    # MAP2 = fits.open('/Users/acharyya/Desktop/test_slice_284of581.fits')[0].data * (initial*1e3)**2 #to change input units ergs/s/pc^2 to ergs/s #n2
    if not keep: plt.close('all')

    fwhm = float(fwhm)
    fwhm_base = res_phys / initial
    MAPconv = convolve(MAP, fwhm_base)

    intermediate_pix_size = galsize / int(np.round(galsize / (res_phys / fwhm)))
    fwhm = res_phys / intermediate_pix_size
    start_time = time.time()
    # MAPbin = congrid(MAP, (int(galsize/intermediate_pix_size), int(galsize/intermediate_pix_size)))
    MAPbin = rebin(MAP, (int(galsize / intermediate_pix_size), int(galsize / intermediate_pix_size)))
    print 'Rebinning 1 slice completed in %s minutes\n' % ((time.time() - start_time) / 60)
    MAPbinconv = convolve(MAPbin, fwhm)

    print 'shapes=', np.shape(MAP), np.shape(MAPbin)  #
    print 'sums=', np.sum(MAP), np.sum(MAPconv), np.sum(MAPbin), np.sum(MAPbinconv)  #
    # sys.exit() #

    final_pix_size = galsize / int(np.round(galsize / (res_phys / fwhm_sample)))
    MAPbinconv_unbin = congrid(MAPbinconv, (int(galsize / final_pix_size), int(galsize / final_pix_size)))
    print 'congrid: input/output size ratio, final pix per beam=', np.shape(MAPbinconv)[0] / float(
        np.shape(MAPbinconv_unbin)[0]), \
        res_phys / (galsize / np.shape(MAPbinconv_unbin)[0])

    plotmap(MAP / ((galsize / np.shape(MAP)[0]) * 1e3) ** 2, cmin=cmin, cmax=cmax, title='MAP')
    plotmap(MAPconv / ((galsize / np.shape(MAPconv)[0]) * 1e3) ** 2, cmin=cmin, cmax=cmax,
            title='MAP convolved ppb=' + str(fwhm_base) + ' pix_size=' + str(initial))
    plotmap(MAPbin / ((galsize / np.shape(MAPbin)[0]) * 1e3) ** 2, cmin=cmin, cmax=cmax,
            title='MAP binned x' + str('%.1F' % (intermediate_pix_size / initial)))
    plotmap(MAPbinconv / ((galsize / np.shape(MAPbinconv)[0]) * 1e3) ** 2, cmin=cmin, cmax=cmax,
            title='MAP binned x' + str('%.1F' % (intermediate_pix_size / initial)) + ' convolved ppb=' + str(
                fwhm) + ' pix_size=' + str(galsize / np.shape(MAPbinconv)[0]))
    plotmap(MAPbinconv_unbin / ((galsize / np.shape(MAPbinconv_unbin)[0]) * 1e3) ** 2, cmin=cmin, cmax=cmax,
            title='MAP binned x' + str('%.1F' % (intermediate_pix_size / initial)) + ' convolved ppb=' + str(
                fwhm_sample) + ' pix_size=' + str(galsize / np.shape(MAPbinconv_unbin)[0]) + ' rebinned x' + str(
                '%.1F' % (final_pix_size / intermediate_pix_size)))

    plt.figure()
    plt.scatter(get_dist(MAP), np.log10(MAP / ((galsize / np.shape(MAP)[0]) * 1e3) ** 2).flatten(), c='k', lw=0, s=3,
                label='MAP')
    plt.scatter(get_dist(MAPconv), np.log10(MAPconv / ((galsize / np.shape(MAPconv)[0]) * 1e3) ** 2).flatten(), c='g',
                lw=0, s=1, label='convolve ppb=' + str(fwhm_base) + ' pix_size=' + str(initial))

    plt.scatter(get_dist(MAPbin), np.log10(MAPbin / ((galsize / np.shape(MAPbin)[0]) * 1e3) ** 2).flatten(), c='b',
                lw=0, s=1, label='bin')
    plt.scatter(get_dist(MAPbinconv), np.log10(MAPbinconv / ((galsize / np.shape(MAPbinconv)[0]) * 1e3) ** 2).flatten(),
                c='r', lw=0, s=1,
                label='bin-> convolve ppb=' + str(fwhm) + ' pix_size=' + str(galsize / np.shape(MAPbinconv)[0]))
    plt.scatter(get_dist(MAPbinconv_unbin),
                np.log10(MAPbinconv_unbin / ((galsize / np.shape(MAPbinconv_unbin)[0]) * 1e3) ** 2).flatten(), c='y',
                lw=0, s=1, label='bin-> convolved-> unbin')

    plt.xlabel('r(kpc)')
    plt.ylabel('log(flux)')
    plt.ylim(-40, -10)
    plt.xlim(0, 14)  # kpc
    lg = plt.legend(loc="lower left")
    for ind in range(len(lg.legendHandles)): lg.legendHandles[ind]._sizes = [30]
    # plt.gcf().savefig('/Users/acharyya/Dropbox/MarkGroupMeetings/10.20.2017/ayan/test_smooth_ppb'+str(fwhm1)+'.eps')
    plt.show(block=False)

    if 'MAP2' in locals():
        ratio = MAP2 / MAP
        MAPconv2 = convolve(MAP2, fwhm_base)
        ratioconv = MAPconv2 / MAPconv
        ratio = np.ma.masked_where((~np.isfinite(ratio)) | (ratio <= 0.), ratio)
        ratioconv = np.ma.masked_where((~np.isfinite(ratioconv)) | (ratioconv <= 0.), ratioconv)

        print ''
        mydiag(MAP, 'MAP')
        mydiag(MAP2, 'MAP2')
        mydiag(ratio, 'ratio')
        print ''
        mydiag(MAPconv, 'MAPconv')
        mydiag(MAPconv2, 'MAPconv2')
        mydiag(ratioconv, 'ratioconv')

        plotmap(MAP2 / ((galsize / np.shape(MAP2)[0]) * 1e3) ** 2, cmin=cmin, cmax=cmax, title='MAP2')
        plotmap(MAPconv2 / ((galsize / np.shape(MAPconv2)[0]) * 1e3) ** 2, cmin=cmin, cmax=cmax,
                title='MAP2 convolved ppb=' + str(fwhm_base) + ' pix_size=' + str(initial))

        plt.figure()
        plt.scatter(get_dist(MAP2), np.log10(MAP2).flatten(), c='k', lw=0, s=3, label='MAP2')
        plt.scatter(get_dist(MAPconv2), np.log10(MAPconv2).flatten(), c='g', lw=0, s=1,
                    label='convolve ppb=' + str(fwhm_base) + ' pix_size=' + str(initial))
        plt.xlabel('r(kpc)')
        plt.ylabel('log(flux)')
        plt.ylim(-40, -10)
        plt.xlim(0, 14)  # kpc
        lg = plt.legend(loc="lower left")
        for ind in range(len(lg.legendHandles)): lg.legendHandles[ind]._sizes = [30]
        plt.show(block=False)

        plotmap(ratio, title='MAP2/MAP', makelog=False)
        plotmap(ratioconv, title='MAP2/MAP convolved ppb=' + str(fwhm_base) + ' pix_size=' + str(initial),
                makelog=False)

        plt.figure()
        plt.scatter(get_dist(MAP), ratio, c='k', lw=0, s=3, label='MAP2/MAP')
        plt.scatter(get_dist(MAP), ratioconv, c='g', lw=0, s=1,
                    label='after convolve ppb=' + str(fwhm_base) + ' pix_size=' + str(initial))
        plt.xlabel('r(kpc)')
        plt.ylabel('MAP ratio')
        plt.ylim(0, 0.3)
        plt.xlim(0, 14)  # kpc
        lg = plt.legend(loc="top left")
        for ind in range(len(lg.legendHandles)): lg.legendHandles[ind]._sizes = [30]
        plt.show(block=False)
