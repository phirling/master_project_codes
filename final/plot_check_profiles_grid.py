import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import mpph
gamma = 5.0/3.0

# Parse user input
parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="+", help="snapshot files to be imaged")
parser.add_argument("-logM200",type=float,default=7,help="Virial mass of halo (total mass)")
parser.add_argument("-c",type=float,default=20,help="NFW Concentration of halo")
parser.add_argument("-nbins", type=int, default=64, help="Number of radii to sample (bins)")
parser.add_argument("-xh0", type=float, default=2e-4, help="Ionized fraction of hydrogen in the IC file")
parser.add_argument("-XH", type=float, default=1.0, help="Hydrogen Mass Fraction")
parser.add_argument("-o",default=None)
parser.add_argument("--tex",action='store_true',help="Use LaTeX")

args = parser.parse_args()
fname = args.file[0]
nbins = int(args.nbins)
xHII = float(args.xh0)
if args.tex: mpl.rcParams["text.usetex"] = True

# Baryonic fraction of the NFW halo
fb = 0.15

# Mean molecular weight
mu = mu_grid = 1.0 / (args.XH * (1+xHII+0.25*(1.0/args.XH-1.0)))

# Extract profiles from grid
rbc, pndens = mpph.get_density_profile_grid(fname,nbins)
rbc, ptemp = mpph.get_temperature_profile_grid(fname,nbins)

# Model
nfw = mpph.get_NFW_props(args.logM200,args.c)
rho_0 = nfw['rho_0'].to('Msun/kpc**3').value
r_s = nfw['r_s'].to('kpc').value
r_200 = nfw['r_200'].to('kpc').value
G = mpph.G_kpc_Ms_kms

# Compute analytical profiles
npoints_an = 10*nbins
r_an = np.logspace(np.log10(rbc[0]),np.log10(rbc[-1]),npoints_an)
pndens_analytical = fb * mpph.NFW_Density(r_an,rho_0,r_s) * mpph.Msun_per_kpc3_to_cgs / mpph.mass_hydrogen_g
ppress_analytical = np.empty(npoints_an)
for k in range(npoints_an):
    ppress_analytical[k] = mpph.NFW_Pressure(r_an[k],10*r_200,rho_0,r_s,G)
ptemp_analytical = (gamma-1.0) * mu * mpph.mass_hydrogen_g/mpph.k_B_cgs * \
    mpph.NFW_internal_energy(ppress_analytical,r_an,rho_0,r_s) * 1e10 # (km/s)^2 -> (cm/s)^2

# Plot result
cgrid = "coral"
cmodel = "skyblue"
fig, ax = plt.subplots(1,2,figsize=(10,4),tight_layout=True)
ax[0].loglog(rbc,pndens,label='Grid',color=cgrid)
ax[0].loglog(r_an,pndens_analytical,label='Model',ls='--',color=cmodel)
ax[1].loglog(rbc,ptemp,label='Grid',color=cgrid)
ax[1].loglog(r_an,ptemp_analytical,label='Model',ls='--',color=cmodel)

ax[0].set_xlabel("$r$ [kpc]")
ax[1].set_xlabel("$r$ [kpc]")
ax[0].set_ylabel("$n_\mathrm{H}$ [cm$^{-3}$]")
ax[1].set_ylabel("$T$ [K]")
ax[0].legend()
ax[1].legend()
ax[0].set_title("Density Profile")
ax[1].set_title("Temperature Profile")
plt.show()