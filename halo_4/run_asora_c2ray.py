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
from pygrackle.utilities.physical_constants import cm_per_mpc, mass_hydrogen_cgs
from time import time as walltime

"""
Compare halo reionization result between Asora+C2Ray and Asora+Grackle
in the case without cooling and heating

Note: still use grackle in this script to ensure same temperature

Checklist
---------
1. Is temperature array equal ? (from internal energy)
2. How accurate is the xfrac for varying timesteps ?
3. Which timestep to choose for grackle ?

A1: There is a tiny difference (~ 0.1 K) in temperature between the
internally computed value by grackle, and the naive perfect gas calculation
taken from the AGORA paper. Let's use the grackle values

A3: For now, we use a fixed fraction of the cell-crossing time.

NOTE:
X AND Y ARE MASS FRACTIONS, NOT NUMBER FRACTIONS!!!
THE CONVERSION FROM MDENS TO NDENS IS WRONG (DONT USE MU)
THE ATTRIBUTION OF ABUNDANCES BETWEEN C2RAY AND GRACKLE IS WRONG
"""
# ======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="+", help="Grid IC file")
parser.add_argument("-numsrc",type=int,default=1,help="Number of sources to use (isotropically)")
parser.add_argument("--grackle",action='store_true')
args = parser.parse_args()
# ======================================================================

# Global parameters (hardcoded)
final_time = 2e6 * YEAR     # Final time of the simulation
dt_c2ray = 1e5 * YEAR  # Time step to use for C2Ray time integration
dt_factor_grackle = 1     # Fraction of the cell crossing time to use as dt
dt_output = 1e5 * YEAR  # Time between saving snapshots

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
initial_internal_energy = gs.u_cgs
mdens_gas = gs.dens_cgs # This is the mass density of the whole gas (H+He) in g/cm3
mdens_hydrogen = X * mdens_gas
ndens_hydrogen = mdens_hydrogen / mass_hydrogen_cgs

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
chemistry_data.density_units = mass_hydrogen_cgs # Density is in mh/cm3
chemistry_data.length_units = 1.0 #cm
chemistry_data.time_units = 1.0 #s
chemistry_data.set_velocity_units()

chemistry_data.initialize()

# Set up fluid
fc = FluidContainer(chemistry_data, N*N*N)
fc["density"] = to_grackle(mdens_gas / mass_hydrogen_cgs) # Since density unit is mh
fc["HI"]    = (1.0-initial_ionized_H_fraction) * X * fc["density"]
fc["HII"]   = initial_ionized_H_fraction * X * fc["density"]
fc["de"]   = initial_ionized_H_fraction * X * fc["density"] # Electron density is given in units me/mp * unit
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
dt_courant = dt_factor_grackle * cell_crossing_time

# =====================
# CONFIGURE ASORA/C2RAY
# =====================
paramfile = "parameters.yml"
sim = pc2r.C2Ray_Minihalo(paramfile, N, False, boxsize_Mpc)
output_dir = sim.results_basename

# Set material properties
sim.ndens = ndens_hydrogen
sim.temp = from_grackle(fc["temperature"])
sim.xh = initial_ionized_H_fraction * np.ones((N,N,N),order='F')

# ================
# SET UP SOURCE(S)
# ================
center = N//2 - 1

numsrc = int(args.numsrc)
srcpos = np.empty((3,numsrc))
srcflux = np.empty(numsrc)

# When using only one source, we place it at the left XY plane
if numsrc == 1:
    srcpos[0,0] = 1
    srcpos[1,0] = 1 + center
    srcpos[2,0] = 1 + center
    srcflux[0] = 1.0
else:
    # Create random generator with fixed seed and generate angles
    gen = np.random.default_rng(100)
    phi_rand = gen.uniform(0.0, 2 * np.pi, numsrc)
    theta_rand = np.arccos(gen.uniform(-1.0, 1.0, numsrc))
    
    R_src = N/2.0 - 0.5
    offset = N/2.0

    # Random positions from the center
    srcpos[0,:] = np.ceil( offset + R_src * np.sin(theta_rand) * np.cos(phi_rand)   )
    srcpos[1,:] = np.ceil( offset + R_src * np.sin(theta_rand) * np.sin(phi_rand)   )
    srcpos[2,:] = np.ceil( offset + R_src * np.cos(theta_rand)                      )
    
    srcflux[:] = 1.0 / numsrc

    # Error check
    if np.any(srcpos > N) or np.any(srcpos < 1):
        raise ValueError("Some sources are outside of the grid!")
    if np.any(srcflux < 0.0):
        print(srcflux)
        raise ValueError("Some sources have negative fluxes (reduce std)")

# Print some info
print(f"Final time:                 {final_time/YEAR:.5e} yrs")
print(f"C2Ray time-step:            {dt_c2ray/YEAR:.5e} yrs")
print(f"Courant time-step:          {dt_courant/YEAR:.5e} yrs")
print(f"Initial mu:                 {fc['mean_molecular_weight'][0]:.5f}")
print(f"Initial mean temperature:   {fc['temperature'].mean():.4e} K")

# =============
# SET UP EVOLVE
# =============
def write_output(dir,output_number,rho,u,x,t):
    gso = GridSnapshot(N=N,dens_cgs=rho,u_cgs=u,xfrac=x,boxsize=boxsize_Mpc,time=t)
    fn = dir + f"snapshot_{output_number:04n}.hdf5"
    gso.write(fn)

current_time = 0.0
next_output_time = dt_output
walltime0 = walltime()

if args.grackle: DT = dt_courant
else:            DT = dt_c2ray

# Write initial state and initialize output
write_output(output_dir,0,gs.dens_cgs,gs.u_cgs,sim.xh,0.0)
nsnap = 1

while (current_time < final_time):
    xh_prev = sim.xh

    if final_time - current_time < DT:
        actual_dt = final_time - current_time
    else:
        actual_dt = DT
    
    if (next_output_time <= final_time and current_time + actual_dt > next_output_time):
        actual_dt = next_output_time - current_time

    print(f"TIME: {current_time/YEAR:.3e} YRS, DT: {actual_dt/YEAR:.3e} YRS, WALL-CLOCK TIME: {walltime()-walltime0:.3e} S")

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
        #flat_x = to_grackle(sim.xh)
        #fc["HI"]    = (1.0-flat_x) * X * fc["density"]
        #fc["HII"]   = flat_x * X * fc["density"]
        #fc["de"]   = flat_x * X * fc["density"]
        #fc.calculate_temperature()
        #print(fc["temperature"].mean())
        #sim.temp = from_grackle(fc["temperature"])
        sim.evolve3D(actual_dt,srcflux,srcpos)
        print("Mean temperature:", sim.temp.mean())
    current_time += actual_dt

    if current_time == next_output_time:
        write_output(output_dir,nsnap,chemistry_data.density_units*from_grackle(fc["density"]),gs.u_cgs,sim.xh,current_time/(1e6*YEAR))
        next_output_time += dt_output
        nsnap += 1

    mean_relative_change_x = np.abs( (sim.xh-xh_prev) / sim.xh ).mean()
    print("-> Mean relative change in x: ",mean_relative_change_x)

print("done")

plt.imshow(sim.xh[:,:,center].T,norm='log',origin='lower',cmap='Spectral_r',vmin=2e-4,vmax=1)
plt.colorbar()
plt.xlabel("$x$ [kpc]")
plt.ylabel("$y$ [kpc]")
plt.show()