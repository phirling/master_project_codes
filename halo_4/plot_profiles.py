import numpy as np
import matplotlib.pyplot as plt
import argparse
from my_format import GridSnapshot
import constants as cst
from scipy.integrate import quad
from tqdm import tqdm
import cycler
import matplotlib as mpl

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", help="Nbody file")
parser.add_argument("-M200",type=float,default=1e7,help="Virial mass of halo (total mass)")
parser.add_argument("-c",type=float,default=17,help="NFW Concentration of halo")
parser.add_argument("-shell_thickness", type=int,default=2,help="Thickness of radial shells in number of cells")
parser.add_argument("-o",default=None)
args = parser.parse_args()

# Set color cycle & corresponding cmap
ncolor = len(args.files)
color_cycle = plt.cm.GnBu_r(np.linspace(0, 1,ncolor,endpoint=False))
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color_cycle)
cmap = mpl.cm.GnBu_r

# Halo parameters
fb = 0.15
M200 = args.M200
c = args.c
delta_c = 200/3. * c**3/( np.log(1+c) - c/(1+c) )
rho_0 = delta_c * cst.rho_c
r_s = 1/c * (M200/(4/3.*np.pi*cst.rho_c*200))**(1/3.)
r200 = c*r_s

# Conversion factors
UnitDensity_in_cgs = cst.UnitMass_in_g / (cst.UnitLength_in_cm**3)

# Primodrial Hydrogen mass fraction
XH = 0.76 

# Make figure
fig, ax = plt.subplots(1,3,constrained_layout=True,figsize=(12,4.2),squeeze=False)
ax[0,0].set_xlabel("$r$ [Mpc]")
ax[0,1].set_xlabel("$r$ [Mpc]")
ax[0,2].set_xlabel("$r$ [Mpc]")

ax[0,0].set_ylabel(r"$\rho(r)$ [M$_{\odot}$ Mpc$^{-3}$]")
ax[0,1].set_ylabel(r"$T(r)$ [K]")
ax[0,2].set_ylabel(r"$x_\mathrm{HI}(r)$")

ax[0,0].set_title("Density")
ax[0,1].set_title("Temperature")
ax[0,2].set_title("Ionized Hydrogen Fraction")
fig.suptitle(fr"$M_{{200}} = {M200:.1e}$ M$_{{\odot}}$, $c={c:.2f}$")

for u in range(3):
    ax[0,u].axvline(r_s,ls=':',color='black')
    ax[0,u].axvline(r200,ls='--',color='black')

# Loop through files
times = []
for fn in tqdm(args.files):

    gs = GridSnapshot(fn)
    dens_cgs = gs.dens_cgs
    u_cgs = gs.u_cgs
    xfrac = gs.xfrac
    boxsize = gs.boxsize
    # Compute temperature assuming standard primoridal He fraction
    # Mean molecular mass in each cell: rho_gas / (nHI + nHII + nHeI + ne)
    # Assuming nHI = (1-x)nH, nHII = xnH, ne = xnH + deducing nHeI from primordial mass fraction:
    mu_grid = 1.0 / (XH * (1+xfrac+0.25*(1.0/XH-1.0)))
    temp_cgs = (cst.gamma-1) * mu_grid * cst.mh_cgs/cst.kb_cgs * u_cgs
    #temp_cgs = (cst.gamma-1)*cst.mu*cst.mh_cgs/cst.kb_cgs * u_cgs

    if gs.time in times:
        ls = ':'
        clr = 'black'
    else: 
        ls = '-'
        clr = None
    times.append(gs.time)
    N = dens_cgs.shape[0]
    dr = boxsize / N
    xsp = np.logspace(np.log10(dr/2),np.log10(boxsize/2))
    xsp_full = np.linspace(-(boxsize-dr)/2 , +(boxsize-dr)/2 , N)
    

    rbin_edges = np.arange(0,boxsize/2,args.shell_thickness*dr)
    rbin_centers = rbin_edges[:-1] + np.diff(rbin_edges) / 2
    nbins = len(rbin_edges)
    xx,yy,zz = np.meshgrid(xsp_full,xsp_full,xsp_full)
    rr = np.sqrt(xx**2 + yy**2 + zz**2)
    zero_arr = np.zeros_like(rr)

    dens_shells = np.empty(nbins-1)
    temp_shells = np.empty(nbins-1)
    x_HI_shells = np.empty(nbins-1)

    for i in range(nbins-1):

        marr = np.where(np.logical_and(rr > rbin_edges[i],rr <= rbin_edges[i+1]), dens_cgs, zero_arr)
        dens_shells[i] = marr.sum() / np.count_nonzero(marr) / UnitDensity_in_cgs

        tarr = np.where(np.logical_and(rr > rbin_edges[i],rr <= rbin_edges[i+1]), temp_cgs, zero_arr)
        temp_shells[i] = tarr.sum() / np.count_nonzero(tarr)

        xarr = np.where(np.logical_and(rr > rbin_edges[i],rr <= rbin_edges[i+1]), xfrac, zero_arr)
        x_HI_shells[i] = xarr.sum() / np.count_nonzero(tarr)


    ax[0,0].loglog(rbin_centers,dens_shells,marker='.',ls=ls,color=clr)
    ax[0,1].loglog(rbin_centers,temp_shells,marker='.',ls=ls,color=clr)
    ax[0,2].loglog(rbin_centers,x_HI_shells,marker='.',ls=ls,color=clr)

# TODO: save time of each snapshot file
if times[-1] == times[0]: vmax = times[0]+1
else: vmax = times[-1]
norm = mpl.colors.Normalize(vmin=times[0], vmax=vmax)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=ax[0,2], orientation='vertical', label='Time [Myr]')

if args.o is not None:
    fn = str(args.o)
    plt.savefig(fn,dpi=300)
else:
    plt.show()


# Reference
"""


# NFW density
def Density(r):
    return fb*rho_0/((r/r_s)*(1+r/r_s)**2)
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

dens_analytical = Density(xsp) * UnitDensity_in_cgs
temp_analytical = T(xsp,10*r200)

if plot_analytical:
    ax[0,0].loglog(xsp,dens_analytical,'--')
    ax[0,1].loglog(xsp,temp_analytical,'--')

    ratio_dens = dens_shells / (Density(rbin_centers) * UnitDensity_in_cgs)
    ratio_temp = temp_shells / T(rbin_centers,2*r200)

    ax[1,0].set_ylabel("Ratio")
    ax[1,1].set_ylabel("Ratio")
    ax[1,0].semilogx(rbin_centers,ratio_dens)
    ax[1,1].semilogx(rbin_centers,ratio_temp)
    ax[1,0].axhline(1,ls='--',color='black')
    ax[1,1].axhline(1,ls='--',color='black')
"""