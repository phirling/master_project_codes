import numpy as np
import matplotlib.pyplot as plt
import argparse
from my_format import GridSnapshot
from tqdm import tqdm
import cmasher as cmr
import unyt

H0 = 72.0 * unyt.km / unyt.s / unyt.Mpc
rho_c = 3*H0**2 / (8*np.pi*unyt.G)
rho_c = rho_c.to(unyt.Solar_Mass/unyt.kpc**3)
gamma = 5.0 / 3.0

parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="+", help="Nbody file")
parser.add_argument("-logM200",type=float,default=None,help="Virial mass of halo in solar masses")
parser.add_argument("-c",type=float,default=20,help="NFW Concentration of halo")
parser.add_argument("-shell_thickness", type=int,default=1,help="Thickness of radial shells in number of cells")
parser.add_argument("-o",default=None)
args = parser.parse_args()

fname = args.file[0]
print(fname)

XH = 1.0 # Hydrogen mass fraction

# For analytical comparison
fb = 0.15
if args.logM200 is not None:
    M200 = 10.0**args.logM200 * unyt.Solar_Mass
    plot_analytical = True
else:
    M200 = 1e7 * unyt.Solar_Mass# to avoid errors
    plot_analytical = False
c = args.c

# Derived NFW parameters
delta_c = 200/3. * c**3/( np.log(1+c) - c/(1+c) )
rho_0 = (delta_c * rho_c)
r_s = (1/c * (M200/(4/3.*np.pi*rho_c*200))**(1/3.)).to('kpc')
r200 = c * r_s
def Density(r):
    return fb*rho_0/((r/r_s)*(1+r/r_s)**2)

# Load data
gs = GridSnapshot(fname)
u_cgs = gs.u_cgs * unyt.erg / unyt.g
boxsize_Mpc = gs.boxsize
xHII = gs.xfrac
ndens_cgs = gs.dens_cgs

# Compute temperature
mu_grid = 1.0 / (XH * (1+xHII+0.25*(1.0/XH-1.0)))
temp_cgs = ((gamma-1) * mu_grid * unyt.mass_hydrogen/unyt.kb * u_cgs).to('K').value

N = ndens_cgs.shape[0]
dr_Mpc = boxsize_Mpc / N
extent_kpc = (0,1000*boxsize_Mpc,0,1000*boxsize_Mpc)
ctr = N//2-1
xsp = np.logspace(np.log10(dr_Mpc/2),np.log10(boxsize_Mpc/2)) * unyt.Mpc
xsp_full = np.linspace(-(boxsize_Mpc-dr_Mpc)/2 , +(boxsize_Mpc-dr_Mpc)/2 , N)
ndens_analytical = (XH * Density(xsp) / unyt.mass_hydrogen).to('1/cm**3')

shellsize = args.shell_thickness
rbin_edges = np.arange(0,boxsize_Mpc/2,shellsize*dr_Mpc)
rbin_centers = rbin_edges[:-1] + np.diff(rbin_edges) / 2
nbins = len(rbin_edges)
xx,yy,zz = np.meshgrid(xsp_full,xsp_full,xsp_full)
rr = np.sqrt(xx**2 + yy**2 + zz**2)
zero_arr = np.zeros_like(rr)

dens_shells = np.empty(nbins-1)
temp_shells = np.empty(nbins-1)
x_HI_shells = np.empty(nbins-1)
for i in tqdm(range(nbins-1)):
    # shell_indices = np.where(np.logical_and(rr > rbin_edges[i],rr <= rbin_edges[i+1]))
    # shell_num = len(shell_indices)
    # dens_shells[i] = np.sum(dens_cgs[shell_indices]) / shell_num
    marr = np.where(np.logical_and(rr > rbin_edges[i],rr <= rbin_edges[i+1]), ndens_cgs, zero_arr)
    dens_shells[i] = marr.sum() / np.count_nonzero(marr)

    tarr = np.where(np.logical_and(rr > rbin_edges[i],rr <= rbin_edges[i+1]), temp_cgs, zero_arr)
    temp_shells[i] = tarr.sum() / np.count_nonzero(tarr)

    xarr = np.where(np.logical_and(rr > rbin_edges[i],rr <= rbin_edges[i+1]), xHII, zero_arr)
    x_HI_shells[i] = xarr.sum() / np.count_nonzero(tarr)

#fig, ax = plt.subplots(1,3,constrained_layout=True,figsize=(12,5.5),sharex=True,height_ratios=[5,2],squeeze=False)
fig, ax = plt.subplots(1,3,constrained_layout=True,figsize=(12,4.5),sharex=True,squeeze=False)
print(ax.shape)
ax[0,0].set_ylabel(r"$n_\mathrm{H}(r)$ [$\mathrm{cm^{-3}}$]")
ax[0,1].set_ylabel(r"$T(r)$ [K]")
ax[0,2].set_ylabel(r"$x_\mathrm{HII}(r)$")
ax[0,0].loglog(rbin_centers,dens_shells,'.-')
ax[0,1].loglog(rbin_centers,temp_shells,'.-')
ax[0,2].loglog(rbin_centers,x_HI_shells,'.-')

if plot_analytical:
    ax[0,0].loglog(xsp,ndens_analytical,'--')
    #ax[0,1].loglog(xsp,temp_analytical,'--')

if args.o is not None:
    fn = str(args.o)
    plt.savefig(fn)
else:
    plt.show()