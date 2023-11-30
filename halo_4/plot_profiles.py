import numpy as np
import matplotlib.pyplot as plt
import argparse
from my_format import GridSnapshot
import constants as cst
from scipy.integrate import quad
from tqdm import tqdm
import cycler
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import cmasher as cmr

MYR = 3.15576E+13
parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", help="Nbody file")
parser.add_argument("-M200",type=float,default=1e7,help="Virial mass of halo (total mass)")
parser.add_argument("-c",type=float,default=17,help="NFW Concentration of halo")
parser.add_argument("-nbins", type=int,default=None,help="Number of radial bins to use (sets the shell size)")
parser.add_argument("--log",action='store_true')
parser.add_argument("-o",default=None)
parser.add_argument("--neutral",action='store_true',help="show neutral H fraction rather than ionized")
parser.add_argument("--fullH",action='store_true',help="show the full H density in the left panel (not HI/HII)")
parser.add_argument("--marker",action='store_true',help="Put a marker of center of bin on plot")
parser.add_argument("-tmax",type=float,default=2.0,help="Final time on colorbar in Myr")
args = parser.parse_args()

# Set color map & norm
#cmap = mpl.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0, vmax=float(args.tmax)) # Myr
colors = ["firebrick", "orangered","peachpuff","skyblue" ,"steelblue"] # wheat
cmap = LinearSegmentedColormap.from_list("mycmap", colors)

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

if args.fullH: ax[0,0].set_title("HI + HII Density")
elif args.neutral: ax[0,0].set_title("HI Density")
else: ax[0,0].set_title("HII Density")
ax[0,1].set_title("Temperature")
if args.neutral: ax[0,2].set_title("Neutral Hydrogen Fraction")
else: ax[0,2].set_title("Ionized Hydrogen Fraction")
fig.suptitle(fr"$M_{{200}} = {M200:.1e}$ M$_{{\odot}}$, $c={c:.2f}$")

for u in range(3):
    ax[0,u].axvline(r_s,ls=':',color='black')
    ax[0,u].axvline(r200,ls='--',color='black')

    ax[0,u].set_yscale('log')
    if args.log:
        ax[0,u].set_xscale('log')

# Colormap to the right of plots
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=ax[0,2], orientation='vertical', label='Time [Myr]')

# Loop through files
times = []
for j,fn in enumerate(tqdm(args.files)):
    gs = GridSnapshot(fn)
    u_cgs = gs.u_cgs
    boxsize = gs.boxsize
    # xfrac is either the ionized or neutral fraction
    if args.neutral:
        xfrac = 1.0 - gs.xfrac
    else:
        xfrac = gs.xfrac
    # dens_cgs is either the full H density or the HI/HII density
    if args.fullH:
        dens_cgs = XH * gs.dens_cgs
    else:
        dens_cgs = XH * xfrac * gs.dens_cgs
    
    # Compute temperature assuming standard primoridal He fraction
    # Mean molecular mass in each cell: rho_gas / (nHI + nHII + nHeI + ne)
    # Assuming nHI = (1-x)nH, nHII = xnH, ne = xnH + deducing nHeI from primordial mass fraction:
    mu_grid = 1.0 / (XH * (1+xfrac+0.25*(1.0/XH-1.0)))
    temp_cgs = (cst.gamma-1) * mu_grid * cst.mh_cgs/cst.kb_cgs * u_cgs

    time_myr = gs.time
    clr = cmap(norm(time_myr))
    if time_myr in times: ls = '--'
    else: ls = '-'
    if args.marker: mk = '.'
    else: mk = None
    times.append(time_myr)
    N = dens_cgs.shape[0]
    dr = boxsize / N
    xsp = np.logspace(np.log10(dr/2),np.log10(boxsize/2))
    xsp_full = np.linspace(-(boxsize-dr)/2 , +(boxsize-dr)/2 , N)
    
    if args.nbins is None:
        nbins = N // 4 + 1
    else:
        nbins = int(args.nbins)
    rbin_edges = np.linspace(0,boxsize/2,nbins)
    shell_thickness = np.diff(rbin_edges)[0] / dr
    if j == 0:
        if shell_thickness < 1.0:
            print("Warning: Shell thickness is less than a grid cell")
        else:
            print(f"Using spherical shells of thickness {shell_thickness:.3f} dr")
    rbin_centers = rbin_edges[:-1] + np.diff(rbin_edges) / 2
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


    ax[0,0].plot(rbin_centers,dens_shells,marker=mk,ls=ls,color=clr)
    ax[0,1].plot(rbin_centers,temp_shells,marker=mk,ls=ls,color=clr)
    ax[0,2].plot(rbin_centers,x_HI_shells,marker=mk,ls=ls,color=clr)

print(times)

if args.o is not None:
    fn = str(args.o)
    plt.savefig(fn,dpi=300)
else:
    plt.show()