import numpy as np
from matplotlib import pyplot as plt
import argparse
from pNbody import Nbody

fontsz = 11
plt.rcParams.update({"font.size": fontsz})
figsize = (7,5)

# Parse user input
parser = argparse.ArgumentParser(
    description="Plot multiple density profiles against theoretical prediction"
)
parser.add_argument("file", nargs="+", help="snapshot files to be imaged")
#parser.add_argument("-logRmin", type=float, default=-1.15, help="Min Radius for the analytical plot")
#parser.add_argument("-logRmax", type=float, default=2.25, help="Max Radius for the analytical plot")
parser.add_argument("-M200",type=float,default=3.14799e+08,help="Virial mass of halo (total mass)")
parser.add_argument("-c",type=float,default=17.21,help="NFW Concentration of halo")
parser.add_argument("-nbins", type=int, default=64, help="Number of radii to sample (bins)")
parser.add_argument("-shift", type=float, default=0.0, help="Shift applied to particles in params.yml")
parser.add_argument("-o",default=None)

args = parser.parse_args()
fname = args.file
print(fname)
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

logRmin = np.log10(r_s / 50)
logRmax = np.log10(2*r200)

# Define radial bins & init arrays
#rbins = np.logspace(args.logRmin, args.logRmax, args.nbins)
rbins = np.logspace(logRmin, logRmax, args.nbins)

# Calculate & Plot Density profile
fig, ax = plt.subplots(figsize=figsize)
ax.set_xlabel("$r$ [Mpc]",fontsize=fontsz+4)
ax.set_ylabel(r"$\rho(r)$ [M$_{\odot}$ Mpc$^{-3}$]",fontsize=fontsz+4)
ax.loglog()
ax.set_title(f"$M_{{200}} = {M200 : .3e}$ M$_{{\odot}}$, $c={c:.2f}$")
ax.axvline(r_s,color='orangered',label=f"$r_s = {r_s:.3e}$ Mpc")
ax.axvline(r200,color='cornflowerblue',label=f"$r_{{200}} = {r200:.3e}$ Mpc")
ax.legend()

# Functions to compute density
# (1) Spherical average in radial bins
def plot_dens_spherical(fname,ax):
    nb = Nbody(fname)
    mass = nb.mass
    pos = nb.pos - nb.boxsize[0]/2
    r = np.sqrt(np.sum(pos ** 2, 1))# / 1000

    # Methods to compute density profile
    def mass_ins(R):
        return ((r < R) * mass).sum()

    mass_ins_vect = np.vectorize(mass_ins)

    def density(R):
        return np.diff(mass_ins_vect(R)) / np.diff(R) / (4.0 * np.pi * R[1:] ** 2)

    dens = density(rbins)
    rs = rbins[1:]

    # remove empty bins
    c = dens > 0
    dens = np.compress(c, dens)
    rs = np.compress(c, rs)

    ax.plot(rs, dens,color='black',lw=2)

    return rs, dens


rs, dens_numerical = plot_dens_spherical(fname,ax)
dens_analytical = NFW_Density(rs)
ax.plot(rs, dens_analytical, c="darkturquoise",ls='--' ,lw=2.3,label="Model (NFW)")

# diff = dens_numerical - dens_analytical
# q75, q25 = np.percentile(dens_numerical, [75 ,25])
# iqr = q75 - q25
# nrmsd = np.std(diff) / iqr
# print(nrmsd)

if args.o is not None:
    fn = str(args.o)
    plt.savefig(fn)
else:
    plt.show()