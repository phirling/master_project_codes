import sys
sys.path.append("../pyc2ray_pdm/")
import pyc2ray as pc2r
import numpy as np
import time
import argparse
import pickle as pkl
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-numsrc",type=int,default=1,help="Number of sources to use (isotropically)")
parser.add_argument("-Rsrc",type=float,default=0,help="Distance of the sources from the center of the box")
parser.add_argument("--gpu",action='store_true')
parser.add_argument("--plot",action='store_true')
parser.add_argument("--debug",action='store_true')
parser.add_argument("-o",type=str,default=None)
args = parser.parse_args()

# Global parameters
paramfile = "parameters.yml"
N = 256
use_octa = args.gpu
center = N//2-1 # Center of the box (in C-indexing from 0)
R_halo = 30


# Create C2Ray object
sim = pc2r.C2Ray_Minihalo(paramfile, N, use_octa)

if args.debug:
    numsrc = 1
    R_src = 0
    x_rand = 1
    y_rand = 1 + center
    z_rand = 1 + center
else:
    # We use sources distributed isotropically at a fixed radius R_src of the halo
    numsrc = int(args.numsrc)
    R_src = args.Rsrc

    # Create random generator with fixed seed and generate angles
    gen = np.random.default_rng(100)
    phi_rand = gen.uniform(0.0, 2 * np.pi, numsrc)
    theta_rand = np.arccos(gen.uniform(-1.0, 1.0, numsrc))

    # We must add 1 because source positions are given in 1-indexing (Fortran style)
    x_rand = 1 + center + R_src * np.sin(theta_rand) * np.cos(phi_rand)
    y_rand = 1 + center + R_src * np.sin(theta_rand) * np.sin(phi_rand)
    z_rand = 1 + center + R_src * np.cos(theta_rand)

srcpos = np.empty((3,numsrc))
srcpos[0,:] = x_rand
srcpos[1,:] = y_rand
srcpos[2,:] = z_rand
tot_power = 1
source_power = tot_power / numsrc
srcflux = source_power * np.ones(numsrc)

# NB: this is another interesting test: place sources all on the left side of the box
# x_rand = 1
# y_rand = np.linspace(center + 1 - R_halo, center + 1 + R_halo,numsrc) # Whole side of the box corresp. to the area where the halo is
# z_rand = 1+center

# Density field: homogeneous sphere of fixed radius R_halo with high density
# outside the sphere, the density is very low (optically thin IGM)
dens_ins = 0.01
dens_out = 2e-10
xi = np.arange(0,N)
halopos = np.array([center,center,center])
X,Y,Z = np.meshgrid(xi,xi,xi)
rr = np.sqrt((X - halopos[0])**2 + (Y - halopos[1])**2 + (Z - halopos[2])**2)
sim.ndens = np.where(rr <= R_halo,dens_ins,dens_out)
#sim.ndens = 2e-9 * np.ones((N,N,N))

# Call raytracing once to compute the Gamma field everywhere
sim.do_raytracing(srcflux,srcpos)

# Save the output (ionization rate + metadata)
res = {
    'Gamma' : sim.phi_ion,
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
    plt.imshow(sim.phi_ion[:,:,center].T,cmap='plasma',origin='lower',norm='log')
    plt.colorbar()
    plt.show()