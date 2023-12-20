import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpph import GridSnapshot
from matplotlib.colors import Normalize
import cmasher as cmr
import unyt
gamma = 5.0 / 3.0 # Adiabatic index

parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="+", help="Nbody file")
parser.add_argument("--mass",action='store_true')
parser.add_argument("--temp",action='store_true')
parser.add_argument("--xHI",action='store_true')
parser.add_argument("-b",type=float,default=None,help="Box around center")
parser.add_argument("-inc",type=int,default=1,help="Increment")
parser.add_argument("-XH",type=float,default=1.0,help="Hydrogen mass fraction")
args = parser.parse_args()

# Set color normalization for temperature and HI/HII fraction
cmap_ndens = 'viridis'
cmap_T = 'afmhot'
cmap_x = 'Spectral_r'

# Hydrogen mass fraction
XH = args.XH

gs = GridSnapshot(args.file[0])
u_cgs = gs.u_cgs * unyt.erg / unyt.g
xHI = 1.0 - gs.xfrac
boxsize_Mpc = gs.boxsize
time_Myr = gs.time
N = u_cgs.shape[0]
print(np.nanmin(u_cgs[np.nonzero(u_cgs)]))

# Print info
print(f"t       = {time_Myr:.2f} Myr")
print(f"boxsize = {boxsize_Mpc*1000:.2f} kpc")

# Set region and image extent
dr_kpc = 1000 * boxsize_Mpc / N
boxsize_kpc = 1000*boxsize_Mpc
if args.b is None:
    il = 0
    ir = N
else:
    b = args.b
    il = int(N//2 - b/dr_kpc)
    ir = int(N//2 + b/dr_kpc)
extent_kpc = [il*dr_kpc - boxsize_kpc/2,
            ir*dr_kpc - boxsize_kpc/2,
            il*dr_kpc - boxsize_kpc/2,
            ir*dr_kpc - boxsize_kpc/2]

if args.mass:
    quant = np.log10(gs.dens_cgs)
    cmap = cmap_ndens
    norm = Normalize(max(quant.min(),-7),quant.max())
    title = "$\log n_\mathrm{H}\ \mathrm{[cm^{-3}]}$"
elif args.temp:
    mu_grid = 1.0 / (XH * (1+gs.xfrac+0.25*(1.0/XH-1.0)))
    temp = ((gamma-1) * mu_grid * unyt.mass_hydrogen/unyt.kb * u_cgs).to('K').value
    quant = np.log10(temp)
    cmap = cmap_T
    norm = Normalize(quant.min(),4.8)
    title = "$\log T\ \mathrm{[K]}$"
else:
    quant = np.log10(xHI)
    cmap = cmap_x
    title = "$\log x_\mathrm{HI}$"
    norm = Normalize(-5,0)

# Create interactive figure
zi = N // 2
z_current = zi
incr = int(args.inc)

def switch(event):
    global z_current
    global incr
    # "Fast scroll"
    if event.key == 'shift+up' or event.key == 'shift+down':
        aincr = 10*incr
    else:
        aincr = incr
    # Increment if up or down
    up = event.key == 'up' or event.key == 'shift+up'
    down = event.key == 'down' or event.key == 'shift+down'
    zt = z_current
    if up:
        zt += aincr
    elif down:
        zt -= aincr
    if up or down:
        if zt in range(N):
            im.set_data(quant[:,:,zt].T)
            z_current = zt
            ttext.set_text(f"$z={zt*dr_kpc - boxsize_kpc/2:.2f}$ kpc")
            fig.canvas.draw()

fig, ax = plt.subplots(figsize=(9.5,8),tight_layout=True)
im = ax.imshow(quant[:,:,zi].T,origin='lower',cmap=cmap,norm=norm,extent=extent_kpc)

fig.canvas.mpl_connect('key_press_event',switch)
plt.colorbar(im)
ax.set_title(title)
ax.set_xlabel("$x\ \mathrm{[kpc]}$")
ax.set_ylabel("$y\ \mathrm{[kpc]}$")

# Text to display the current z coordinate
ttext = ax.text(0.02, 0.98, f"$z={zi*dr_kpc - boxsize_kpc/2:.2f}$ kpc",
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes,
                color='white', fontsize=14)

plt.show()