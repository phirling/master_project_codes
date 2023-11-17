import numpy as np
from matplotlib import pyplot as plt
import argparse
from scipy.integrate import quad
import constants as cst
from pNbody import Nbody

fontsz = 11
plt.rcParams.update({"font.size": fontsz})
figsize = (12,5)

# Parse user input
parser = argparse.ArgumentParser(
    description="Plot multiple density profiles against theoretical prediction"
)
parser.add_argument("file", nargs="+", help="snapshot files to be imaged")
#parser.add_argument("-logRmin", type=float, default=-1.15, help="Min Radius for the analytical plot")
#parser.add_argument("-logRmax", type=float, default=2.25, help="Max Radius for the analytical plot")
parser.add_argument("-M200",type=float,default=3.14799e+08,help="Virial mass of halo (total mass)")
parser.add_argument("-c",type=float,default=17.21,help="NFW Concentration of halo")
parser.add_argument("-fb",type=float,default=0.15,help="Baryonic fraction of halo (gas fraction)")
parser.add_argument("-nbins", type=int, default=64, help="Number of radii to sample (bins)")
parser.add_argument("-shift", type=float, default=0.0, help="Shift applied to particles in params.yml")
parser.add_argument("-o",default=None)

args = parser.parse_args()
fname = args.file
print(fname)

M200 = args.M200
c = args.c
fb = args.fb
delta_c = 200/3. * c**3/( np.log(1+c) - c/(1+c) )
rho_0 = delta_c * cst.rho_c
r_s = 1/c * (M200/(4/3.*np.pi*cst.rho_c*200))**(1/3.)
r200 = c*r_s

# Density
def Density(r):
    return rho_0/((r/r_s)*(1+r/r_s)**2)

# Integrand for the pressure
def integrand(r):
    return Density(r) * cst.G * 4*np.pi*rho_0*r_s**3 * ( np.log(1+r/r_s) - (r/r_s)/(1+r/r_s) ) /  r**2

# Temperature
def T(r,rmax):
    press = np.zeros(len(r))
    for i in range(len(r)):
        press[i] = quad(integrand, r[i], rmax, args=())[0]
    u = press/(cst.gamma-1)/Density(r)
    T = (cst.gamma-1)*cst.mu*cst.mh/cst.kb * u
    return T

# Define radial bins & init arrays
logRmin = np.log10(r_s / 50)
logRmax = np.log10(2*r200)

#rbins = np.logspace(args.logRmin, args.logRmax, args.nbins)
rbins = np.logspace(logRmin, logRmax, args.nbins)

# Calculate & Plot Density and Temperature profile
fig, ax = plt.subplots(1,2,constrained_layout=True,figsize=(12,5.5))
ax[0].set_xlabel("$r$ [Mpc]",fontsize=fontsz+4)
ax[1].set_xlabel("$r$ [Mpc]",fontsize=fontsz+4)
ax[0].set_ylabel(r"$\rho(r)$ [M$_{\odot}$ Mpc$^{-3}$]",fontsize=fontsz+4)
ax[1].set_ylabel(r"$T(r)$ [K]",fontsize=fontsz+4)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[0].set_title(f"$M_{{200}} = {M200 : .3e}$ M$_{{\odot}}$, $c={c:.2f}$")
for k in range(2):
    ax[k].axvline(r_s,color='orangered',label=f"$r_s = {r_s:.3e}$ Mpc")
    ax[k].axvline(r200,color='cornflowerblue',label=f"$r_{{200}} = {r200:.3e}$ Mpc")
    ax[k].legend()

# Functions to compute density
# (1) Spherical average in radial bins
def plot_dens_temp_spherical(fname,ax1,ax2):
    nb = Nbody(fname)
    mass = nb.mass
    u = nb.u_init
    pos = nb.pos - nb.boxsize[0]/2
    r = np.sqrt(np.sum(pos ** 2, 1))# / 1000

    # Methods to compute density profile
    def mass_ins(R):
        return ((r < R) * mass).sum()
    def energy_ins(R):
        return ((r < R) * u * mass).sum()
    
    mass_ins_vect = np.vectorize(mass_ins)
    energy_ins_vect = np.vectorize(energy_ins)

    def density(R):
        #return np.diff(mass_ins_vect(R)) / np.diff(R) / (4.0 * np.pi * R[1:] ** 2)
        return np.diff(mass_ins_vect(R)) / np.diff(R)

    def temperature(R):
        u_in_shell = np.diff(energy_ins_vect(R))
        m_in_shell = np.diff(mass_ins_vect(R))
        return (cst.gamma-1)*cst.mu*cst.mh/cst.kb * u_in_shell/m_in_shell
    
    
    # We use centered bins
    rs = rbins[:-1] + np.diff(rbins)/2
    dens = density(rbins) / (4.0 * np.pi * rs ** 2)
    temp = temperature(rbins)

    # # remove empty bins
    # c = dens > 0
    # dens = np.compress(c, dens)
    # rs = np.compress(c, rs)

    ax[0].plot(rs, dens,color='black',lw=2)
    ax[1].plot(rs, temp,color='black',lw=2)

    return rs, dens, temp

rs, dens_numerical,temp_numerical = plot_dens_temp_spherical(fname,ax[0],ax[1])
dens_analytical = fb * Density(rs)
temp_analytical = T(rs,10*r200)
ax[0].plot(rs, dens_analytical, c="darkturquoise",ls='--' ,lw=2.3,label="Model (NFW)")
ax[1].plot(rs, temp_analytical, c="darkturquoise",ls='--' ,lw=2.3,label="Model (NFW)")

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