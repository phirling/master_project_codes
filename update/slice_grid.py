import numpy as np
import matplotlib.pyplot as plt
import argparse
from my_format import GridSnapshot
from matplotlib.patches import Circle
from tqdm import tqdm
from scipy.integrate import quad
from matplotlib.colors import CenteredNorm
import cmasher as cmr
import unyt

H0 = 72.0 * unyt.km / unyt.s / unyt.Mpc
rho_c = 3*H0**2 / (8*np.pi*unyt.G)
rho_c = rho_c.to(unyt.Solar_Mass/unyt.kpc**3)
gamma = 5.0 / 3.0

parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="+", help="Nbody file")
parser.add_argument("-x", type=int,default=None)
parser.add_argument("-y", type=int,default=None)
parser.add_argument("-z", type=int,default=None)
parser.add_argument("-interp", type=str,default=None,help="Imshow interpolation to use")
parser.add_argument("-o",default=None)
parser.add_argument("--neutral",action='store_true',help="show neutral H fraction rather than ionized")
parser.add_argument("--fullH",action='store_true',help="show the full H density in the left panel (not HI/HII)")
args = parser.parse_args()

fname = args.file[0]
print(fname)

XH = 1.0 # Hydrogen mass fraction

# Load data
gs = GridSnapshot(fname)
u_cgs = gs.u_cgs * unyt.erg / unyt.g
boxsize_Mpc = gs.boxsize
print(boxsize_Mpc)
if args.neutral:
    xfrac = 1.0 - gs.xfrac
else:
    xfrac = gs.xfrac

# This is either the neutral or ionized hydrogen density
if args.fullH: dens_cgs = XH * gs.dens_cgs
else: dens_cgs = XH * xfrac * gs.dens_cgs

# Compute temperature
mu_grid = 1.0 / (XH * (1+xfrac+0.25*(1.0/XH-1.0)))
temp_cgs = ((gamma-1) * mu_grid * unyt.mass_hydrogen/unyt.kb * u_cgs).to('K').value

N = dens_cgs.shape[0]
dr_Mpc = boxsize_Mpc / N
extent_kpc = (0,1000*boxsize_Mpc,0,1000*boxsize_Mpc)
ctr = N//2-1
xsp_full = np.linspace(-(boxsize_Mpc-dr_Mpc)/2 , +(boxsize_Mpc-dr_Mpc)/2 , N)


fig, ax = plt.subplots(1,3,constrained_layout=True,figsize=(14,4),squeeze=False)
for k in range(3):
    ax[0,k].set_xlabel("$x$ [kpc]")
    ax[0,k].set_ylabel("$y$ [kpc]")
if args.fullH: ax[0,0].set_title("H (HI+HII) Density [g/cm3]")
elif args.neutral: ax[0,0].set_title("HI Density [g/cm3]")
else: ax[0,0].set_title("HII Density [g/cm3]")
#ax[0,0].set_title("Density [g/cm3]")
ax[0,1].set_title("Temperature [K]")
if args.neutral: ax[0,2].set_title("Neutral Fraction")
else: ax[0,2].set_title("Ionized Fraction")

if args.x is not None:
    dens_slice = dens_cgs[int(args.x),:,:]
    temp_slice = temp_cgs[int(args.x),:,:]
    x_HI_slice = xfrac[int(args.x),:,:]
elif args.y is not None:
    dens_slice = dens_cgs[:,int(args.y),:]
    temp_slice = temp_cgs[:,int(args.y),:]
    x_HI_slice = xfrac[:,int(args.y),:]
elif args.z is not None:
    dens_slice = dens_cgs[:,:,int(args.z)]
    temp_slice = temp_cgs[:,:,int(args.z)]
    x_HI_slice = xfrac[:,:,int(args.z)]
else:
    dens_slice = dens_cgs[:,:,ctr]
    temp_slice = temp_cgs[:,:,ctr]
    x_HI_slice = xfrac[:,:,ctr]

iitp = str(args.interp)
im0 = ax[0,0].imshow(dens_slice.T,extent=extent_kpc,origin='lower',norm='log',cmap='viridis',interpolation=iitp)
im1 = ax[0,1].imshow(temp_slice.T,extent=extent_kpc,origin='lower',norm='log',cmap='cmr.dusk',interpolation=iitp)
im2 = ax[0,2].imshow(x_HI_slice.T,extent=extent_kpc,origin='lower',norm='log',cmap='Spectral_r',vmin=2e-4,vmax=1,interpolation=iitp)

plt.colorbar(im0)
plt.colorbar(im1)
plt.colorbar(im2)

if args.o is not None:
    fn = str(args.o)
    plt.savefig(fn)
else:
    plt.show()
