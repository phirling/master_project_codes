import sys
sys.path.append("../pyc2ray_pdm/")
import pyc2ray as pc2r
from pyc2ray.c2ray_base import YEAR
import numpy as np
import argparse
import pickle as pkl
import matplotlib.pyplot as plt
import astropy.constants as ac

# ======================================================================
parser = argparse.ArgumentParser()

parser.add_argument("file", nargs="+", help="Nbody file")

# General script parameters
parser.add_argument("-dt",default=None,type=float,help="Timestep to use if solving chemistry. If none, only raytrace once")
parser.add_argument("-nsteps",default=1,type=int,help="Number of timesteps for evolve")
parser.add_argument("-nsteps_output",default=5,type=int,help="Number of timesteps between outputs")
parser.add_argument("--gpu",action='store_true')
parser.add_argument("-o",type=str,default=None)
args = parser.parse_args()

# ======================================================================

# Run Parameters
paramfile = "parameters.yml"
num_steps_between_output = int(args.nsteps_output)

# ============
# LOAD DENSITY
# ============
fname = args.file[0]
abu_h =  0.926
abu_he = 0.074
mean_molecular = abu_h + 4.0*abu_he
m_p = ac.m_p.cgs
mdens_to_ndens = 1.0 / (mean_molecular * m_p)
with open(fname,"rb") as f:
    data = pkl.load(f)
ndens = data["dens_cgs"] * mdens_to_ndens
N = ndens.shape[0]
boxsize_Mpc = data["boxsize"]

# Convert
center = N//2 - 1
boxsize_kpc = boxsize_Mpc * 1000
dr = boxsize_kpc / N
extent = (0,boxsize_kpc,0,boxsize_kpc)

# Create C2Ray object
sim = pc2r.C2Ray_Minihalo(paramfile, N, args.gpu,boxsize_Mpc)
output_dir = sim.results_basename
sim.ndens = ndens

# =======
# SOURCES
# =======
numsrc = 1
R_src = 0
x_rand = np.array([1])
y_rand = np.array([1 + center])
z_rand = np.array([1 + center])
srcpos = np.empty((3,numsrc))
srcpos[0,:] = x_rand
srcpos[1,:] = y_rand
srcpos[2,:] = z_rand
tot_power = 1
source_power = tot_power / numsrc
srcflux = source_power * np.ones(numsrc)

# ==================
# RAYTRACE OR EVOLVE
# ==================

if args.dt is None:
    sim.do_raytracing(srcflux,srcpos)
    plt.imshow(sim.phi_ion[:,:,center].T,cmap='plasma',origin='lower',norm='log',extent=extent)
    plt.colorbar()
    plt.title(fr"Ionization Rate [1/s] ($dr={dr*1000:.3f}$ pc)")
    plt.xlabel("$x$ [kpc]")
    plt.ylabel("$y$ [kpc]")
    plt.show()
else:
    dt = args.dt * YEAR
    nsteps = int(args.nsteps)
    for i in range(nsteps):
        sim.evolve3D(dt,srcflux,srcpos)
        if i % num_steps_between_output == 0:
            fn = output_dir + f"xfrac_{i:n}.pkl"
            with open(fn,"wb") as f:
                pkl.dump(sim.xh,f)
            fn = output_dir + f"ionrate_{i:n}.pkl"
            with open(fn,"wb") as f:
                pkl.dump(sim.phi_ion,f)
    plt.imshow(sim.xh[:,:,center].T,norm='log',origin='lower',cmap='Spectral_r',extent=extent)
    plt.colorbar()
    plt.title(fr"Ionized H Fraction ($dr={dr*1000:.3f}$ pc)")
    plt.xlabel("$x$ [kpc]")
    plt.ylabel("$y$ [kpc]")
    plt.show()