import sys
sys.path.append("../pyc2ray_pdm/")
import pyc2ray as pc2r
import numpy as np
import time
import argparse
import pickle as pkl
import matplotlib.pyplot as plt
import nfw

# ======================================================================
parser = argparse.ArgumentParser()

# Global & source parameters
parser.add_argument("-N",type=int,default=128)
parser.add_argument("-numsrc",type=int,default=1,help="Number of sources to use (isotropically)")
parser.add_argument("-Rsrc",type=float,default=0,help="Distance of the sources from the center of the box")
parser.add_argument("--debug",action='store_true',help="Debug mode uses a single source placed at left box edge")
parser.add_argument("-dens",type=str,default='sphere',help="Type of density to use: 'const','sphere','nfw' are possible modes")

# Parameters for constant density
parser.add_argument("-cstdens",type=float,default=2e-12,help="Value of the constant density")

# Parameters for homogeneous sphere
parser.add_argument("-Rhalo",type=int,default=0,help="Radius of the sphere")

# Parameters for NFW density
parser.add_argument("-M200",type=float,default=25986.21038221,help="M200 mass of the NFW halo (in Msun)")
parser.add_argument("-c",type=float,default=5.0,help="Concentration of the NFW halo")
parser.add_argument("-eps",type=float,default=None,help="Gravitational softening of the NFW halo (in Mpc)")


# General script parameters
parser.add_argument("--gpu",action='store_true')
parser.add_argument("--plot",action='store_true')
parser.add_argument("--plotdens",action='store_true')
parser.add_argument("-o",type=str,default=None)
args = parser.parse_args()

# ======================================================================

# Convert
paramfile = "parameters.yml"
N = int(args.N)
center = N//2-1 # Center of the box (in C-indexing from 0)
R_halo = int(args.Rhalo)

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
dens_ins = 0.01
dens_out = 2e-9
xi = np.arange(0,N)
halopos = np.array([center,center,center])
X,Y,Z = np.meshgrid(xi,xi,xi)
rr = np.sqrt((X - halopos[0])**2 + (Y - halopos[1])**2 + (Z - halopos[2])**2)
if args.dens == 'sphere':
    # Density field: homogeneous sphere of fixed radius R_halo with high density
    # outside the sphere, the density is very low (optically thin IGM)
    print("Using hard homogeneous sphere density")
    ndens = np.where(rr <= R_halo,dens_ins,dens_out)
elif args.dens == 'const':
    print("Using constant density")
    ndens = args.cstdens * np.ones((N,N,N))
elif args.dens == 'nfw':
    print("Using NFW density")
    boxsize = 0.01
    dr = boxsize / N
    M200 = args.M200
    c = args.c
    if args.eps is None: eps = dr/2.
    else: eps = args.eps

    nfw_halo = nfw.NFW(M200,c,eps)
    ndens = nfw_halo.density_gridded(N,boxsize)
else:
    raise NameError("Unknown density: ",args.dens)

sim.ndens = ndens

# ===========================
# DO RAYTRACING & SAVE OUTPUT
# ===========================
# Call raytracing once to compute the Gamma field everywhere
sim.do_raytracing(srcflux,srcpos)

# Save the output (ionization rate + metadata)
res = {
    'Gamma' : sim.phi_ion,
    'ndens' : sim.ndens,
    'srcpos' : srcpos,
    'N' : N,
    'R_src' : R_src,
    'R_halo' : R_halo,
    'numsrc' : numsrc,
    'dens_ins' : dens_ins,
    'dens_out' : dens_out
}

if args.o is not None:
    outfn = str(args.o)
    print("Saving result in " + outfn + " ...")
    with open(outfn,"wb") as f:
        pkl.dump(res,f)

if args.plot:
    fig1,ax1 = plt.subplots()
    im1 = ax1.imshow(sim.phi_ion[:,:,center].T,cmap='plasma',origin='lower',norm='linear')
    plt.colorbar(im1)
    ax1.set_title("Photoionization Rate [1/s]")

if args.plotdens:
    fig2,ax2 = plt.subplots()
    im2 = ax2.imshow(ndens[:,:,center].T,norm='log',origin='lower')
    ax2.scatter(x_rand-1,y_rand-1,s=20,marker='*',c='orangered')
    plt.colorbar(im2)
    ax2.set_title("H Density [atom/cm$^3$]")

plt.show()

"""
# NB: this is another interesting test: place sources all on the left side of the box
# x_rand = 1
# y_rand = np.linspace(center + 1 - R_halo, center + 1 + R_halo,numsrc) # Whole side of the box corresp. to the area where the halo is
# z_rand = 1+center
"""