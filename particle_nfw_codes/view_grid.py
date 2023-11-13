import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle as pkl
from tqdm import tqdm
from matplotlib.colors import CenteredNorm

parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="+", help="Nbody file")
parser.add_argument("-type", type=str,default="slice")
parser.add_argument("-M200",type=float,default=1e7,help="Virial mass of halo (total mass)")
parser.add_argument("-c",type=float,default=17,help="NFW Concentration of halo")
parser.add_argument("-z", type=int,default=None)
parser.add_argument("-shell_thickness", type=int,default=2,help="Thickness of radial shells in number of cells")
args = parser.parse_args()
fname = args.file[0]

# For analytical comparison
rho_c = 1.27209e+11
fb = 0.15
M200 = args.M200
c = args.c
delta_c = 200/3. * c**3/( np.log(1+c) - c/(1+c) )
rho_0 = delta_c * rho_c
r_s = 1/c * (M200/(4/3.*np.pi*rho_c*200))**(1/3.)
r200 = c*r_s
def NFW_Density(r):
    return fb*rho_0/((r/r_s)*(1+r/r_s)**2)

# Conversion factors
Msun_to_g = 1.989e33
Mpc_to_cm = 3.085678e24
Msun_per_Mpc3_to_g_per_cm3 = Msun_to_g / (Mpc_to_cm**3)

# Load data
with open(fname,"rb") as f:
    res = pkl.load(f)

if isinstance(res,dict):
    dens_cgs = res["dens_cgs"]
    boxsize = res["boxsize"]
else:
    # For debug old files
    dens_cgs = res
    boxsize = 0.00908833410701378# 0.005874066682425128
N = dens_cgs.shape[0]
dr = boxsize / N
extent = (0,boxsize,0,boxsize)
ctr = N//2-1
xsp = np.logspace(np.log10(dr/2),np.log10(boxsize/2))
xsp_full = np.linspace(-(boxsize-dr)/2 , +(boxsize-dr)/2 , N)
dens_analytical = NFW_Density(xsp) * Msun_per_Mpc3_to_g_per_cm3

if args.z is None:
    zl = ctr
else:
    zl = int(args.z)

logdens_slice = np.log10(dens_cgs)[:,:,zl]
logdens_proj = np.log10(dens_cgs.sum(axis=2))
xslice = dens_cgs[ctr:,ctr,ctr]


if args.type == "slice":
    plt.imshow(logdens_slice,extent=extent)
    plt.colorbar()
elif args.type == "proj":
    plt.imshow(logdens_proj,extent=extent)
    plt.colorbar()
elif args.type == "xaxis":
    plt.plot(xsp,xslice,'.-')
    plt.plot(xsp,dens_analytical,'--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("x [Mpc]")
    plt.ylabel("Density [cgs]")
elif args.type == "spherical":
    #nbins = 5
    #rbin_edges = np.logspace(np.log10(dr/2),np.log10(boxsize/2),nbins)
    shellsize = args.shell_thickness
    rbin_edges = np.arange(0,boxsize/2,shellsize*dr)
    rbin_centers = rbin_edges[:-1] + np.diff(rbin_edges) / 2
    nbins = len(rbin_edges)
    xx,yy,zz = np.meshgrid(xsp_full,xsp_full,xsp_full)
    rr = np.sqrt(xx**2 + yy**2 + zz**2)
    zero_arr = np.zeros_like(rr)

    dens_shells = np.empty(nbins-1)
    for i in tqdm(range(nbins-1)):
        marr = np.where(np.logical_and(rr > rbin_edges[i],rr <= rbin_edges[i+1]), dens_cgs, zero_arr)
        dens_shells[i] = marr.sum() / np.count_nonzero(marr)
        #print(np.count_nonzero(marr))
    plt.loglog(rbin_centers,dens_shells,'.-')
    plt.loglog(xsp,dens_analytical,'--')
    #plt.loglog(rbin_centers,NFW_Density(rbin_centers) * Msun_per_Mpc3_to_g_per_cm3,'--')
elif args.type == "percell":
    xx,yy,zz = np.meshgrid(xsp_full,xsp_full,xsp_full)
    rr = np.sqrt(xx**2 + yy**2 + zz**2)
    #plt.imshow(rr[:,zl,:],norm='log',cmap='viridis_r')
    dens_analytical_grid = NFW_Density(rr) * Msun_per_Mpc3_to_g_per_cm3
    relerr = (dens_cgs - dens_analytical_grid) / dens_analytical_grid
    plt.imshow(relerr[:,:,zl],norm=CenteredNorm(),cmap='RdBu',extent=extent)
    plt.colorbar()
plt.show()