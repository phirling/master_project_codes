import numpy as np
import matplotlib.pyplot as plt
import argparse
from my_format import GridSnapshot
from matplotlib.patches import Circle
from tqdm import tqdm
import constants as cst
from scipy.integrate import quad
from matplotlib.colors import CenteredNorm

parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="+", help="Nbody file")
parser.add_argument("-type", type=str,default="slice")
parser.add_argument("-M200",type=float,default=None,help="Virial mass of halo (total mass)")
parser.add_argument("-c",type=float,default=17,help="NFW Concentration of halo")
parser.add_argument("-z", type=int,default=None)
parser.add_argument("-shell_thickness", type=int,default=2,help="Thickness of radial shells in number of cells")
parser.add_argument("-o",default=None)
args = parser.parse_args()


fname = args.file[0]
print(fname)

# For analytical comparison
fb = 0.15
if args.M200 is not None:
    M200 = args.M200
    plot_analytical = True
else:
    M200 = 1e7 # to avoid errors
    plot_analytical = False

c = args.c
delta_c = 200/3. * c**3/( np.log(1+c) - c/(1+c) )
rho_0 = delta_c * cst.rho_c
r_s = 1/c * (M200/(4/3.*np.pi*cst.rho_c*200))**(1/3.)
r200 = c*r_s

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

# Conversion factors
UnitDensity_in_cgs = cst.UnitMass_in_g / (cst.UnitLength_in_cm**3)

# Load data
gs = GridSnapshot(fname)
dens_cgs = gs.dens_cgs
u_cgs = gs.u_cgs
temp_cgs = (cst.gamma-1)*cst.mu*cst.mh_cgs/cst.kb_cgs * u_cgs
boxsize = gs.boxsize
xfrac = gs.xfrac

N = dens_cgs.shape[0]
dr = boxsize / N
extent = (0,1000*boxsize,0,1000*boxsize)
ctr = N//2-1
xsp = np.logspace(np.log10(dr/2),np.log10(boxsize/2))
xsp_full = np.linspace(-(boxsize-dr)/2 , +(boxsize-dr)/2 , N)
dens_analytical = Density(xsp) * UnitDensity_in_cgs
temp_analytical = T(xsp,2*r200)
bs = 1000*boxsize/2.0
r200_circ = Circle((bs,bs),1000*r200,fill=False,color='white',ls='--')

if args.z is None:
    zl = ctr
else:
    zl = int(args.z)

logdens_slice = np.log10(dens_cgs)[:,:,zl]
logdens_proj = np.log10(dens_cgs.sum(axis=2))


if args.type == "slice":
    fig, ax = plt.subplots(1,3,constrained_layout=True,figsize=(14,4),squeeze=False)
    for k in range(3):
        ax[0,k].set_xlabel("$x$ [kpc]")
        ax[0,k].set_ylabel("$y$ [kpc]")
    ax[0,0].set_title("Density [g/cm3]")
    ax[0,1].set_title("Temperature [K]")
    ax[0,2].set_title("Ionized Fraction")

    im0 = ax[0,0].imshow(dens_cgs[:,:,zl].T,extent=extent,origin='lower',norm='log',cmap='viridis')
    im1 = ax[0,1].imshow(temp_cgs[:,:,zl].T,extent=extent,origin='lower',norm='log',cmap='jet')
    im2 = ax[0,2].imshow(xfrac[:,:,zl].T,extent=extent,origin='lower',norm='log',cmap='Spectral_r',vmin=2e-4,vmax=1)

    plt.colorbar(im0)
    plt.colorbar(im1)
    plt.colorbar(im2)

    if plot_analytical:
        ax[0,0].add_patch(r200_circ)
    #plt.imshow(logdens_slice,extent=extent)
    #plt.colorbar(label=r"$\log_{10} \rho$ [cgs]")
    #plt.xlabel("$x$ [kpc]")
    #plt.ylabel("$y$ [kpc]")
    #plt.title("Density (Slice)")
    #plt.gca().add_patch(r200_circ)

elif args.type == "proj":
    plt.imshow(logdens_proj,extent=extent)
    plt.colorbar()
elif args.type == "xaxis":
    dens_xslice = dens_cgs[ctr:,ctr,ctr]
    temp_xslice = temp_cgs[ctr:,ctr,ctr]

    fig, ax = plt.subplots(2,2,constrained_layout=True,figsize=(12,5.5),sharex=True,height_ratios=[5,2])
    ax[1,0].set_xlabel("$r$ [Mpc]")
    ax[1,1].set_xlabel("$r$ [Mpc]")
    ax[0,0].set_ylabel(r"$\rho(r)$ [M$_{\odot}$ Mpc$^{-3}$]")
    ax[0,1].set_ylabel(r"$T(r)$ [K]")
    ax[0,0].loglog(xsp,dens_xslice,'.-')
    ax[0,1].loglog(xsp,temp_xslice,'.-')

    if plot_analytical:
        ax[0,0].loglog(xsp,dens_analytical,'--')
        ax[0,1].loglog(xsp,temp_analytical,'--')

        ratio_dens = dens_xslice / (Density(xsp) * UnitDensity_in_cgs)
        ratio_temp = temp_xslice / T(xsp,2*r200)

        ax[1,0].set_ylabel("Ratio")
        ax[1,1].set_ylabel("Ratio")
        ax[1,0].semilogx(xsp,ratio_dens)
        ax[1,1].semilogx(xsp,ratio_temp)
        ax[1,0].axhline(1,ls='--',color='black')
        ax[1,1].axhline(1,ls='--',color='black')
    #fig.subplots_adjust(hspace=0)
    
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
    temp_shells = np.empty(nbins-1)
    for i in tqdm(range(nbins-1)):
        # shell_indices = np.where(np.logical_and(rr > rbin_edges[i],rr <= rbin_edges[i+1]))
        # shell_num = len(shell_indices)
        # dens_shells[i] = np.sum(dens_cgs[shell_indices]) / shell_num
        marr = np.where(np.logical_and(rr > rbin_edges[i],rr <= rbin_edges[i+1]), dens_cgs, zero_arr)
        dens_shells[i] = marr.sum() / np.count_nonzero(marr)

        tarr = np.where(np.logical_and(rr > rbin_edges[i],rr <= rbin_edges[i+1]), temp_cgs, zero_arr)
        temp_shells[i] = tarr.sum() / np.count_nonzero(tarr)

        #print(np.count_nonzero(marr))
    #plt.loglog(1000*rbin_centers,temp_shells,'.-')
    #plt.loglog(1000*xsp,dens_analytical,'--')
    #plt.xlabel("$x$ [kpc]")
    #plt.ylabel("Density [cgs]")
    #plt.title("Spherical Density Profile (Grid)")
    fig, ax = plt.subplots(2,2,constrained_layout=True,figsize=(12,5.5),sharex=True,height_ratios=[5,2])
    ax[1,0].set_xlabel("$r$ [Mpc]")
    ax[1,1].set_xlabel("$r$ [Mpc]")
    ax[0,0].set_ylabel(r"$\rho(r)$ [M$_{\odot}$ Mpc$^{-3}$]")
    ax[0,1].set_ylabel(r"$T(r)$ [K]")
    ax[0,0].loglog(rbin_centers,dens_shells,'.-')
    ax[0,1].loglog(rbin_centers,temp_shells,'.-')

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
    fig.subplots_adjust(hspace=0)

    #plt.loglog(rbin_centers,NFW_Density(rbin_centers) * Msun_per_Mpc3_to_g_per_cm3,'--')
elif args.type == "percell":
    xx,yy,zz = np.meshgrid(xsp_full,xsp_full,xsp_full)
    rr = np.sqrt(xx**2 + yy**2 + zz**2)
    #plt.imshow(rr[:,zl,:],norm='log',cmap='viridis_r')
    dens_analytical_grid = Density(rr) * UnitDensity_in_cgs
    relerr = (dens_cgs - dens_analytical_grid) / dens_analytical_grid
    plt.imshow(relerr[:,:,zl],norm=CenteredNorm(),cmap='RdBu',extent=extent)
    plt.colorbar()

if args.o is not None:
    fn = str(args.o)
    plt.savefig(fn)
else:
    plt.show()