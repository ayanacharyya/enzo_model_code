import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from astropy.io import fits
from operator import itemgetter
import plotobservables as po
import fixppvfit as fp
import warnings

warnings.filterwarnings("ignore")
import argparse as ap

parser = ap.ArgumentParser(description="observables generating tool")


# -------------------------------------------------------------------------------------------
def onclick(event):
    global anim_running, anim
    if anim_running:
        anim.event_source.stop()
        print 'Animation paused..Click to resume.'
        anim_running = False
    else:
        anim.event_source.start()
        anim_running = True
        print 'Animation running..Click to pause.'


# ------------Reading pre-defined line list-----------------------------
def readtargetlines(targetfile, wmin=None, wmax=None):
    if wmin is None: wmin = 0.
    if wmax is None: wmax = 1e6
    target = []
    finp = open('/Users/acharyya/Mappings/lab/' + targetfile, 'r')
    l = finp.readlines()[3:]
    for lin in l:
        if len(lin.split()) > 1 and lin[0] != '#':
            w = float(lin.split()[0])
            if w >= wmin and w <= wmax:
                target.append([w, lin.split()[1]])
    finp.close()
    target = sorted(target, key=itemgetter(0))
    lines = np.array(target)[:, 1]
    return lines


# -------------------------------------------------------------------------------------------
def anim_cube(cmin, cmax, galsize=26.0, pause=1, mapcube=False, wmin=None, wmax=None):
    global ppvcube, lim, fn, path, anim, anim_running, lines
    if cmin is None: cmin = np.min(np.log10(ppvcube))
    if cmax is None: cmax = np.max(np.log10(ppvcube))
    if mapcube: lines = readtargetlines('targetlines.txt', wmin=wmin, wmax=wmax)
    ppvcube = np.ma.masked_where(np.log10(ppvcube) < cmin, ppvcube)
    res = galsize / np.shape(ppvcube)[0]  # kpc/pix
    if lim is None: lim = np.shape(ppvcube)[2]
    ylab = 'Log(flux)'
    xlab = 'rest-wavelength slice'
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.1, left=0.1, right=0.9)
    ax = plt.subplot(111)
    p = ax.imshow(np.log10(ppvcube[:, :, 0]), cmap='rainbow', vmin=cmin, vmax=cmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    ax.set_xticklabels([i * res - galsize / 2 for i in list(ax.get_xticks())])
    ax.set_yticklabels([i * res - galsize / 2 for i in list(ax.get_yticks())])
    ax.set_ylabel('y(kpc)')
    ax.set_xlabel('x(kpc)')
    plt.colorbar(p, cax=cax).set_label(ylab)
    title = ax.set_title('')

    ax2 = fig.add_axes([0.54, 0.74, 0.3, 0.1])
    y = np.log10(np.sum(ppvcube, axis=(0, 1)))
    p2 = ax2.plot(np.arange(np.shape(ppvcube)[2]), y, lw=1)
    ax2.set_title('Total spectrum', fontsize=10)
    ax2.set_xlim(0, np.shape(ppvcube)[2] - 1)
    # ax2.set_ylim(y[0], y[-1])
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    line = ax2.axvline(100, c='r')

    # -------------------------------------------------------------------------------------------
    def init():
        p.set_data(np.log10(ppvcube[:, :, 0]))
        title.set_text('')
        line.set_xdata(0)
        return p, title, line
        # -------------------------------------------------------------------------------------------

    def animate(i):
        global ppvcube, lim, lines
        p.set_data(np.log10(ppvcube[:, :, i]))
        if mapcube:
            title.set_text(lines[i] + ' map of ' + str(lim) + ' maps')
        else:
            title.set_text('Slice ' + str(i + 1) + ' of ' + str(lim))
        line.set_xdata(i)
        return p, title, line

    # -------------------------------------------------------------------------------------------
    # if 'ax' in locals(): fig.delaxes(ax)
    cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
    anim = animation.FuncAnimation(fig, animate, frames=lim, interval=pause, blit=False,
                                   init_func=init)  # interval in milliseconds
    anim_running = True
    print 'Animation running..Click to pause.'
    # anim.save('anim_'+path+fn[:-5]+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()


# -------------------------------------------------------------------------------------------
def plot_ratiomap(mapcube, num, den, cmin, cmax, galsize=26.0, keep=False, arg=None):
    lines = readtargetlines('targetlines.txt')
    res = galsize / np.shape(mapcube)[0]  # kpc/pix
    lim = np.shape(mapcube)[2]

    num_map, den_map = np.zeros(np.shape(mapcube[:, :, 0])), np.zeros(np.shape(mapcube[:, :, 0]))
    for i in range(len(num)): num_map += mapcube[:, :, np.where(lines == num[i])[0][0]]
    for i in range(len(den)): den_map += mapcube[:, :, np.where(lines == den[i])[0][0]]
    ratiomap = num_map / den_map

    if not keep: plt.close('all')
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.1, left=0.1, right=0.9)
    ax = plt.subplot(111)
    p = ax.imshow(np.log10(ratiomap), cmap='rainbow', vmin=cmin, vmax=cmax)
    ax.set_xticklabels([i * res - galsize / 2 for i in list(ax.get_xticks())])
    ax.set_yticklabels([i * res - galsize / 2 for i in list(ax.get_yticks())])
    ax.set_ylabel('y(kpc)')
    ax.set_xlabel('x(kpc)')
    plt.colorbar(p).set_label('Log(flux ratio)')
    plt.title(('Ratio of', num, '/', den, 'lines', arg))
    plt.show(block=False)


# -------------------------------------------------------------------------------------------

global ppvcube, lim, fn, path, pause
parser.add_argument("--cmin")
parser.add_argument("--cmax")
parser.add_argument("--path")
parser.add_argument("--galsize")
parser.add_argument("--pause")
parser.add_argument("--lim")
parser.add_argument("--numlabels")
parser.add_argument("--denlabels")
parser.add_argument("--line")
parser.add_argument("--X")
parser.add_argument("--Y")
parser.add_argument("--wmin")
parser.add_argument("--wmax")
parser.add_argument('--noanim', dest='noanim', action='store_true')
parser.set_defaults(noanim=False)
parser.add_argument('--plotratio', dest='plotratio', action='store_true')
parser.set_defaults(plotratio=False)
parser.add_argument('--plotmap', dest='plotmap', action='store_true')
parser.set_defaults(plotmap=False)
parser.add_argument('--plotspec', dest='plotspec', action='store_true')
parser.set_defaults(plotspec=False)
parser.add_argument('--mapcube', dest='mapcube', action='store_true')
parser.set_defaults(mapcube=False)
parser.add_argument('--keep', dest='keep', action='store_true')
parser.set_defaults(keep=False)
args, leftovers = parser.parse_known_args()

if args.cmin is not None:
    cmin = float(args.cmin)
else:
    cmin = None

if args.cmax is not None:
    cmax = float(args.cmax)
else:
    cmax = None

if args.galsize is not None:
    galsize = float(args.galsize)
else:
    galsize = 26.0  # kpc

if args.pause is not None:
    pause = float(args.pause)
elif args.mapcube:
    pause = 1000  # 1sec pause if each slice is a fitted line map
else:
    pause = 1  # milli sec

if args.lim is not None:
    lim = int(args.lim)
else:
    lim = None  # slices till which to show plot

if args.path is not None:
    path = args.path
    print 'Using path=', path
else:
    path = '/Users/acharyya/Desktop/bpt/'  # which path to use
    print 'Path not specified. Using default', path, '. Use --path option to specify path.'

if args.numlabels is not None:
    numlabels = [item for item in args.numlabels.split(',')]
else:
    numlabels = ['NII6584']

if args.denlabels is not None:
    denlabels = [item for item in args.denlabels.split(',')]
else:
    denlabels = ['H6562']

if args.line is not None:
    line = args.line
else:
    line = 'H6562'

if args.wmin is not None:
    wmin = float(args.wmin)
else:
    wmin = None  # Angstrom; starting wavelength of PPV cube

if args.wmax is not None:
    wmax = float(args.wmax)
else:
    wmax = None  # Angstrom; ending wavelength of PPV cube

# -------------------------------------------------------------------------------------------
fn = sys.argv[1]
fn_prefix = 'PPV'
arg_arr = [30]  # [150,200,300,400,500,600,700,800] #arg = vres
for arg in arg_arr:
    plt.pause(3)
    if fn[-5:] != '.fits': fn += '.fits'
    ppvcube = fits.open(path + fn)[0].data
    if len(np.shape(ppvcube)) < 3:
        print 'This fits file is not a cube. Exitting.'
        sys.exit()
    ppvcube = np.ma.masked_where(ppvcube == 0, ppvcube)
    if args.plotratio:
        plot_ratiomap(ppvcube, numlabels, denlabels, cmin, cmax, galsize=galsize, keep=args.keep,
                      arg='for vres ' + str(arg))
    elif not args.noanim:
        anim_cube(cmin, cmax, galsize=galsize, pause=pause, mapcube=args.mapcube, wmin=wmin, wmax=wmax)
print 'Finished!'
