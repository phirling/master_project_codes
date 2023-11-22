import sys
sys.path.append("../pyc2ray_pdm/")
import pyc2ray as pc2r
from pyc2ray.c2ray_base import YEAR
import numpy as np
import argparse
import constants as cst
from my_format import GridSnapshot
import matplotlib.pyplot as plt
from pygrackle import chemistry_data, FluidContainer
from pygrackle.utilities.physical_constants import cm_per_mpc
from time import time as walltime

"""
Compare halo reionization result between Asora+C2Ray and Asora+Grackle
in the case without cooling and heating

Checklist
---------
1. Is temperature array equal ? (from internal energy)
2. How accurate is the xfrac for varying timesteps ?
3. Which timestep to choose for grackle ?

A1: There is a tiny difference (~ 0.1 K) in temperature between the
internally computed value by grackle, and the naive perfect gas calculation
taken from the AGORA paper. Let's use the grackle values

A3: For now, we use a fixed fraction of the cell-crossing time.

"""
# ======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="+", help="Grid IC file")
parser.add_argument("--grackle",action='store_true')
args = parser.parse_args()
# ======================================================================

# Global parameters (hardcoded)
final_time = 1e5 * YEAR     # Final time of the simulation
dt_c2ray = final_time / 10  # Time step to use for C2Ray time integration
dt_factor_grackle = 1     # Fraction of the cell crossing time to use as dt
dt_output = final_time / 5  # Time between saving snapshots
output_dir = "snap/"

# Use standard primordial hydrogen/helium mass fractions
X = 0.76
Y = 1.0 - X
initial_ionized_H_fraction = 2.0e-4 # Fraction OF hydrogen (not whole gas)

# =======================
# LOAD INITIAL CONDITIONS
# =======================
fname = args.file[0]
gs = GridSnapshot(fname)
N = gs.N
boxsize_Mpc = gs.boxsize
boxsize_cgs = boxsize_Mpc * cm_per_mpc
# Here we use the constant mu to convert mdens<->ndens since this was used to generate the ICs
ndens = gs.dens_cgs / (cst.mu * cst.mh_cgs)
initial_internal_energy = gs.u_cgs

# Helper functions to copy data to/from a grackle fluid container
def to_grackle(A):
    return np.copy(A.flatten())
def from_grackle(A):
    return np.asfortranarray(np.copy(np.reshape(A,(N,N,N))))

# =================
# CONFIGURE GRACKLE
# =================
chemistry_data = chemistry_data()
chemistry_data.use_grackle = 1
chemistry_data.with_radiative_cooling = 0 # No heating or cooling
chemistry_data.primordial_chemistry = 1
chemistry_data.metal_cooling = 0
chemistry_data.UVbackground = 0
chemistry_data.use_radiative_transfer = 1
chemistry_data.self_shielding_method = 0
chemistry_data.H2_self_shielding = 0
chemistry_data.CaseBRecombination = 1 # For consistency (C2Ray uses case B)
chemistry_data.grackle_data_file = bytearray("/Users/phirling/Program/grackle/input/CloudyData_UVB=HM2012.h5", 'utf-8')

# Set units (we use CGS internally, not so good...)
chemistry_data.comoving_coordinates = 0 # proper units
chemistry_data.a_units = 1.0
chemistry_data.a_value = 1.0
chemistry_data.density_units = cst.mh_cgs # rho = 1.0 is 1.67e-24 g
chemistry_data.length_units = 1.0 #cm
chemistry_data.time_units = 1.0 #s
chemistry_data.set_velocity_units()

chemistry_data.initialize()

# Set up fluid
fc = FluidContainer(chemistry_data, N*N*N)
fc["density"] = to_grackle(ndens)
fc["HI"]    = (1.0-initial_ionized_H_fraction) * X * fc["density"]
fc["HII"]   = initial_ionized_H_fraction * X * fc["density"]
fc["HeI"]   = Y * fc["density"]
fc["HeII"]  = 1e-20 * fc["density"]
fc["HeIII"] = 1e-20 * fc["density"]

# Set bulk velocity to zero
fc["x-velocity"][:] = 0.0
fc["y-velocity"][:] = 0.0
fc["z-velocity"][:] = 0.0

# Set internal specific energy [erg/g]
fc["energy"] = to_grackle(initial_internal_energy)


# Get initial mu (scalar)
fc.calculate_mean_molecular_weight()

# Calculate initial temperature
fc.calculate_temperature()

# Set timestep as fraction of cell-crossing time
dr_cgs = boxsize_cgs / N
cell_crossing_time = dr_cgs / cst.c_cgs
dt_grackle = dt_factor_grackle * cell_crossing_time

# =====================
# CONFIGURE ASORA/C2RAY
# =====================
paramfile = "parameters.yml"
sim = pc2r.C2Ray_Minihalo(paramfile, N, False, boxsize_Mpc)

# Set material properties
sim.ndens = ndens
sim.temp = from_grackle(fc["temperature"])
sim.xh = initial_ionized_H_fraction * np.ones((N,N,N),order='F')

# ================
# SET UP SOURCE(S)
# ================
center = N//2 - 1
numsrc = 1
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

# Print some info
print(f"Final time:                 {final_time/YEAR:.5e} yrs")
print(f"C2Ray time-step:            {dt_c2ray/YEAR:.5e} yrs")
print(f"Grackle time-step:          {dt_grackle/YEAR:.5e} yrs")
print(f"Initial mu:                 {fc['mean_molecular_weight'][0]:.5f}")
print(f"Initial mean temperature:   {fc['temperature'].mean():.4e} K")

# =============
# SET UP EVOLVE
# =============
def write_output(dir,output_number,u,x):
    gso = GridSnapshot(N=N,dens_cgs=gs.dens_cgs,u_cgs=u,xfrac=x,boxsize=boxsize_Mpc)
    fn = dir + f"snapshot_{output_number:04n}.hdf5"
    gso.write(fn)

current_time = 0.0
next_output_time = dt_output
walltime0 = walltime()

if args.grackle: DT = dt_grackle
else:            DT = dt_c2ray

# Write initial state and initialize output
write_output(output_dir,0,gs.u_cgs,sim.xh)
nsnap = 1

while (current_time < final_time):
    xh_prev = sim.xh

    if final_time - current_time < DT:
        actual_dt = final_time - current_time
    else:
        actual_dt = DT
    
    if (next_output_time <= final_time and current_time + actual_dt > next_output_time):
        actual_dt = next_output_time - current_time

    print(f"-- time: {current_time/YEAR:.3e} yrs, dt: {actual_dt/YEAR:.3e} yrs, wall-clock time: {walltime()-walltime0:.3e} seconds --")

    if args.grackle:
        # Find photo-ionization rates using Asora
        sim.do_raytracing(srcflux,srcpos)

        # Give rates to grackle
        fc["RT_HI_ionization_rate"] = to_grackle(sim.phi_ion)

        # Solve chemisry
        fc.solve_chemistry(actual_dt)

        # Copy the updated ionization fractions to sim for next asora call
        sim.xh = from_grackle(fc["HII"] / (fc["HI"] + fc["HII"]))
    else:
        # Use built-in method of pyC2Ray
        sim.evolve3D(actual_dt,srcflux,srcpos)

    current_time += actual_dt

    if current_time == next_output_time:
        write_output(output_dir,nsnap,gs.u_cgs,sim.xh)
        next_output_time += dt_output
        nsnap += 1

    mean_relative_change_x = np.abs( (sim.xh-xh_prev) / sim.xh ).mean()
    print("------- > Mean relative change in x: ",mean_relative_change_x)

print("done")

plt.imshow(sim.xh[:,:,center].T,norm='log',origin='lower',cmap='Spectral_r',vmin=2e-4,vmax=1)
plt.colorbar()
plt.xlabel("$x$ [kpc]")
plt.ylabel("$y$ [kpc]")
plt.show()