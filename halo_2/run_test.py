import sys
sys.path.append("../pyc2ray_pdm/")
sys.path.append("../halo_1/")
import pyc2ray as pc2r
from pyc2ray.c2ray_base import YEAR
import numpy as np
import time
import argparse
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import nfw

# ======================================================================
parser = argparse.ArgumentParser()

# Global & source parameters
parser.add_argument("-N",type=int,default=128)
parser.add_argument("-numsrc",type=int,default=1,help="Number of sources to use (isotropically)")
parser.add_argument("-Rsrc",type=float,default=0,help="Distance of the sources from the center of the box")
parser.add_argument("--debug",action='store_true',help="Debug mode uses a single source placed at left box edge")

# General script parameters
parser.add_argument("-dt",default=None,type=float,help="Timestep to use if solving chemistry. If none, only raytrace once")
parser.add_argument("--gpu",action='store_true')
parser.add_argument("--plot",action='store_true')
parser.add_argument("--plotdens",action='store_true')
parser.add_argument("-o",type=str,default=None)
args = parser.parse_args()

# ======================================================================

# r_s = 0.8 kpc 8.0e-4 # Mpc
# rho_0 = 0.0025 # 10^10 msun / kpc^3

# Convert
paramfile = "parameters.yml"
N = int(args.N)
center = N//2-1 # Center of the box (in C-indexing from 0)
boxsize = 0.05 #0.2 # Mpc (~ 2 * 8 * r_200)
dr = boxsize / N
rmin = dr * (-center)       # Leftmost cell
rmax = dr * (N-1-center)    # Rightmost cell
extent = (rmin,rmax,rmin,rmax)

# Darwin's Halo:
r_200 =  1.37714e-02 # Virial Radius [Mpc]
M_200 =  3.14799e+08 # DM Virial Mass [Msun]
c    = 17.21         # Halo concentration
eps = 0.5 * dr       # Gravitational Softening (at halo center)
fb = 0.15            # Baryonic fraction

# Create C2Ray object
sim = pc2r.C2Ray_Minihalo(paramfile, N, args.gpu)

# =======
# SOURCES
# =======
if args.debug:
    # We use a single source at fixed location (left box edge, center)
    numsrc = 1
    R_src = 0
    x_rand = np.array([1])
    y_rand = np.array([1 + center])
    z_rand = np.array([1 + center])
else:
    # We use sources distributed isotropically at a fixed radius R_src of the halo
    numsrc = int(args.numsrc)
    R_src = args.Rsrc

    # Create random generator with fixed seed and generate angles
    gen = np.random.default_rng(100)
    phi_rand = gen.uniform(0.0, 2 * np.pi, numsrc)
    theta_rand = np.arccos(gen.uniform(-1.0, 1.0, numsrc))

    # We must add 1 because source positions are given in 1-indexing (Fortran style)
    x_rand = np.rint(1 + center + R_src * np.sin(theta_rand) * np.cos(phi_rand))
    y_rand = np.rint(1 + center + R_src * np.sin(theta_rand) * np.sin(phi_rand))
    z_rand = np.rint(1 + center + R_src * np.cos(theta_rand))

srcpos = np.empty((3,numsrc))
srcpos[0,:] = x_rand
srcpos[1,:] = y_rand
srcpos[2,:] = z_rand
tot_power = 1
source_power = tot_power / numsrc
srcflux = source_power * np.ones(numsrc)

# =======
# DENSITY
# =======
M_200_gas = fb * M_200
nfw_halo = nfw.NFW(M_200_gas,c,eps)
ndens = nfw_halo.density_gridded(N,boxsize)
sim.ndens = ndens

# ===========================
# DO RAYTRACING & SAVE OUTPUT
# ===========================

# Call raytracing once to compute the Gamma field everywhere
if args.dt is None:
    sim.do_raytracing(srcflux,srcpos)
else:
    dt = args.dt * YEAR
    sim.evolve3D(dt,srcflux,srcpos)
    plt.imshow(sim.xh[:,:,center].T,norm='log',origin='lower',cmap='Spectral_r',extent=extent)
    plt.colorbar()
    plt.title("Ionized H Fraction")
    plt.xlabel("$x-x_\mathrm{{NFW}}$ [Mpc]")
    plt.ylabel("$y-y_\mathrm{{NFW}}$ [Mpc]")
    # plt.show()

# dt = 65231*YEAR #1e2*YEAR
# Or call evolve to use C2Ray to solve the chemistry
#    for i in range(5):
#        sim.evolve3D(dt,srcflux,srcpos)
#        plt.imshow(sim.xh[:,:,center].T,norm='log',origin='lower',cmap='Spectral_r')
#        plt.show()

# Save the output (ionization rate + metadata)
res = {
    'Gamma' : sim.phi_ion,
    'ndens' : sim.ndens,
    'srcpos' : srcpos,
    'N' : N,
    'R_src' : R_src,
    'numsrc' : numsrc,
}

if args.o is not None:
    outfn = str(args.o)
    print("Saving result in " + outfn + " ...")
    with open(outfn,"wb") as f:
        pkl.dump(res,f)

if args.plot:
    fig1,ax1 = plt.subplots()
    im1 = ax1.imshow(sim.phi_ion[:,:,center].T,cmap='plasma',origin='lower',norm='log',extent=extent)
    plt.colorbar(im1)
    ax1.set_title("Photoionization Rate [1/s]")
    ax1.set_xlabel("$x-x_\mathrm{{NFW}}$ [Mpc]")
    ax1.set_ylabel("$y-y_\mathrm{{NFW}}$ [Mpc]")

if args.plotdens:
    fig2,ax2 = plt.subplots()
    im2 = ax2.imshow(ndens[:,:,center].T,norm='log',origin='lower',extent=extent)
    #ax2.scatter(dr*(x_rand-1-center),dr*(y_rand-1-center),s=20,marker='*',c='orangered')
    plt.colorbar(im2)
    ax2.set_title("H Density [atom/cm$^3$]")
    ax2.set_xlabel("$x-x_\mathrm{{NFW}}$ [Mpc]")
    ax2.set_ylabel("$y-y_\mathrm{{NFW}}$ [Mpc]")
    circle_r200 = Circle((0,0),r_200,fill=False,ls='--',color='white')
    circle_src = Circle((0,0),dr*R_src,fill=False,ls='--',color='gold')
    ax2.add_patch(circle_r200)
    ax2.add_patch(circle_src)
    tf = 0.8
    ax2.text(tf*r_200,tf*r_200,"$r_{200}$",color='white')
    ax2.text(tf*dr*R_src,tf*dr*R_src,"$r_{s}$",color='gold')

plt.show()