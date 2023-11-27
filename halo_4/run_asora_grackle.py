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
Halo reionization with photo-heating using Grackle
"""
# ======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="+", help="Grid IC file")
parser.add_argument("-numsrc",type=int,default=1,help="Number of sources to use (isotropically)")
parser.add_argument("--isothermal",action='store_true')
args = parser.parse_args()
# ======================================================================

# Global parameters (hardcoded)
final_time = 2e6 * YEAR     # Final time of the simulation
dt_its_factor = 0.05            # Fraction of the ionization time scale to use as time-step
dt_min = 0.1 * YEAR        # Minimum time-step
dt_max = 1e5 * YEAR         # Maximum time-step
dt_output = 1e5 * YEAR  # Time between saving snapshots
paramfile = "parameters2.yml"

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
mdens_gas = gs.dens_cgs
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
chemistry_data.with_radiative_cooling = 0 #(not args.isothermal)
chemistry_data.primordial_chemistry = 1
chemistry_data.metal_cooling = 0
chemistry_data.UVbackground = 0
chemistry_data.use_radiative_transfer = 1
chemistry_data.radiative_transfer_hydrogen_only = 1
chemistry_data.self_shielding_method = 0
chemistry_data.H2_self_shielding = 0
chemistry_data.CaseBRecombination = 1 # For consistency (C2Ray uses case B)
chemistry_data.grackle_data_file = bytearray("/Users/phirling/Program/grackle/input/CloudyData_UVB=HM2012.h5", 'utf-8')

# Set units (we use CGS internally, not so good...)
chemistry_data.comoving_coordinates = 0 # proper units
chemistry_data.a_units = 1.0
chemistry_data.a_value = 1.0
chemistry_data.density_units = mass_hydrogen_cgs # rho = 1.0 is 1.67e-24 g
chemistry_data.length_units = 1.0 #cm
chemistry_data.time_units = 1.0 #s
chemistry_data.set_velocity_units()

chemistry_data.initialize()

# Set up fluid
fc = FluidContainer(chemistry_data, N*N*N)
fc["density"] = to_grackle(mdens_gas / mass_hydrogen_cgs)
fc["HI"]    = (1.0-initial_ionized_H_fraction) * X * fc["density"]
fc["HII"]   = initial_ionized_H_fraction * X * fc["density"]
fc["de"]   = initial_ionized_H_fraction * X * fc["density"]
fc["HeI"]   = Y * fc["density"]
fc["HeII"]  = 1e-50 * fc["density"]
fc["HeIII"] = 1e-50 * fc["density"]

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

# Cell-crossing time
dr_cgs = boxsize_cgs / N
cell_crossing_time = dr_cgs / cst.c_cgs

# =====================
# CONFIGURE ASORA/C2RAY
# =====================
sim = pc2r.C2Ray_Minihalo(paramfile, N, False, boxsize_Mpc)
output_dir = sim.results_basename

# Set material properties
sim.ndens = ndens_hydrogen
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
print(f"Minimum asora time-step:    {dt_min/YEAR:.5e} yrs")
print(f"Maximum asora time-step:    {dt_max/YEAR:.5e} yrs")
print(f"Initial mu:                 {fc['mean_molecular_weight'][0]:.5f}")
print(f"Initial mean temperature:   {fc['temperature'].mean():.4e} K")

# =============
# SET UP EVOLVE
# =============
def write_output(dir,output_number,rho,u,x,t):
    gso = GridSnapshot(N=N,dens_cgs=rho,u_cgs=u,xfrac=x,boxsize=boxsize_Mpc,time=t)
    fn = dir + f"snapshot_{output_number:04n}.hdf5"
    gso.write(fn)

# Rate for HII + e --> HI + photon
def recombination_rate(temp):
    return 4.881357e-6*temp**(-1.5) * (1.0 + 1.14813e2*temp**(-0.407))**(-2.242)

current_time = 0.0
next_output_time = dt_output
walltime0 = walltime()

# Write initial state and initialize output
write_output(output_dir,0,gs.dens_cgs,gs.u_cgs,sim.xh,0.0)
nsnap = 1

# Save timestep values for stats
timesteps = []

while (current_time < final_time):
    #print(fc["metal"].mean())
    #print(fc["HeI"].mean())
    #print(fc["HeII"].mean())
    #print(fc["de"].mean())
    #fc.calculate_hydrogen_number_density()
    #hdiff = from_grackle(fc["nH"]) -  sim.ndens
    #print(fc["nH"].mean())
    #print(sim.ndens.mean())

    # Save mean ionized fraction
    prev_mean_x = sim.xh.mean()

    # Find photo-ionization rates using Asora
    sim.do_raytracing(srcflux,srcpos)
    
    # Give rates to grackle & compute cooling time
    fc["RT_HI_ionization_rate"] = to_grackle(sim.phi_ion)
    if not args.isothermal:
        fc["RT_heating_rate"] = to_grackle(sim.phi_heat)
        fc.calculate_cooling_time()
        min_cooling_time = fc["cooling_time"].min()

    # "Adaptive timestep": Fraction of ionization timescale Δx/x ~ x/(1-x) Gamma^-1
    fc.calculate_temperature()
    Tnow = from_grackle(fc["temperature"])
    xnow = sim.xh
    photo_rate = (1.0 - xnow) * sim.phi_ion
    recom_rate = xnow**2*ndens_hydrogen*recombination_rate(Tnow)
    ion_timescale_grid = xnow / np.abs(photo_rate - recom_rate)

    #ion_timescale_grid = sim.xh / ((1.0-sim.xh) * sim.phi_ion)
    ion_timescale_min = ion_timescale_grid.min()
    dt_grackle = max(dt_min,min(dt_max,dt_its_factor*ion_timescale_min))

    # If 10% of cooling time is shorter, rather use that as time-step
    if args.isothermal:
        DT = dt_grackle
    else:
        DT = min(0.1*min_cooling_time,dt_grackle)
    
    # DT = min(DT,0.4*cell_crossing_time)

    # Adjust time-step for output & final time
    if final_time - current_time < DT:
        actual_dt = final_time - current_time
    else:
        actual_dt = DT
    
    if (next_output_time <= final_time and current_time + actual_dt > next_output_time):
        actual_dt = next_output_time - current_time

    print(f"TIME: {current_time/YEAR:.3e} YRS, DT: {actual_dt/YEAR:.3e} YRS, WALL-CLOCK TIME: {walltime()-walltime0:.3e} S")

    # Solve chemisry
    print("Solving chemistry...")
    fc.solve_chemistry(actual_dt)

    # Copy the updated ionization fractions to sim for next asora call
    sim.xh = from_grackle(fc["HII"] / (fc["HI"] + fc["HII"]))

    # Save relative change in xfrac
    mean_x = sim.xh.mean()
    rel_change_x = (mean_x - prev_mean_x) / mean_x

    fc.calculate_temperature()

    # Print info
    print("-> Min ion timescale value: ",ion_timescale_min / YEAR,"yrs")
    print(f"-> Relative change in ionized H fraction: {rel_change_x:.4e}")
    print(f"-> Mean temperature: {fc['temperature'].mean():.4f}")

    # Increment simulation time
    current_time += actual_dt

    # Save output if at output time
    if current_time == next_output_time:
        write_output(output_dir,nsnap,chemistry_data.density_units*from_grackle(fc["density"]),gs.u_cgs,sim.xh,current_time/(1e6*YEAR))
        next_output_time += dt_output
        nsnap += 1

    # Save timestep
    ts_ = np.array([DT,actual_dt])
    timesteps.append(ts_)

print("done")

timesteps = np.array(timesteps)
np.savetxt(output_dir + "timesteps.txt",timesteps)

plt.imshow(sim.xh[:,:,center].T,norm='log',origin='lower',cmap='Spectral_r',vmin=2e-4,vmax=1)
plt.colorbar()
plt.xlabel("$x$ [kpc]")
plt.ylabel("$y$ [kpc]")
plt.show()
