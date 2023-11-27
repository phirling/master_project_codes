import sys
sys.path.append("../pyc2ray_pdm/")
from pyc2ray.c2ray_base import YEAR, Mpc, ev2k
from pyc2ray.chemistry import global_pass
import numpy as np
import matplotlib.pyplot as plt
from pygrackle import chemistry_data, FluidContainer
from pygrackle.utilities.physical_constants import mass_hydrogen_cgs

N = 1
dr = 0.001 * Mpc
mdens = 1e-27
u = 4e+10
X = 0.76
Y = 1.0 - X
initial_ionized_H_fraction = 2.0e-4
gamma = 1e-16 # 1/s

final_time = 10e6 * YEAR     # Final time of the simulation
dt_its_factor = 0.05            # Fraction of the ionization time scale to use as time-step
dt_min = 0.1 * YEAR        # Minimum time-step
dt_max = 1e5 * YEAR         # Maximum time-step
#dt_c2ray = 1e3 * YEAR

title = rf"$t_f$ = {final_time/YEAR/1e6:.2f} Myr, $\Gamma$ = {gamma:.2e} s$^{{-1}}$, $\varepsilon_\mathrm{{ion}}={dt_its_factor:.3f}$"

# =================
# CONFIGURE GRACKLE
# =================
chemistry_data = chemistry_data()
chemistry_data.use_grackle = 1
chemistry_data.with_radiative_cooling = 0
chemistry_data.primordial_chemistry = 1
chemistry_data.metal_cooling = 0
chemistry_data.UVbackground = 0
chemistry_data.use_radiative_transfer = 1
chemistry_data.radiative_transfer_hydrogen_only = 1
chemistry_data.self_shielding_method = 0
chemistry_data.H2_self_shielding = 0
chemistry_data.CaseBRecombination = 1 # For consistency (C2Ray uses case B)
chemistry_data.grackle_data_file = bytearray("/Users/phirling/Program/grackle/input/CloudyData_UVB=HM2012.h5", 'utf-8')

chemistry_data.HydrogenFractionByMass = 0.76

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
fc = FluidContainer(chemistry_data, 1)
fc["density"][0] = mdens / mass_hydrogen_cgs
fc["HI"]    = (1.0-initial_ionized_H_fraction) * X * fc["density"]
fc["HII"]   = initial_ionized_H_fraction * X * fc["density"]
#fc["de"]   = initial_ionized_H_fraction * X * fc["density"]
fc["HeI"]   = Y * fc["density"]
fc["HeII"]  = 1e-50 * fc["density"]
fc["HeIII"] = 1e-50 * fc["density"]

# Set bulk velocity to zero
fc["x-velocity"][:] = 0.0
fc["y-velocity"][:] = 0.0
fc["z-velocity"][:] = 0.0

# Set internal specific energy [erg/g]
fc["energy"][0] = u

# Get initial mu (scalar)
fc.calculate_mean_molecular_weight()

# Calculate initial temperature
fc.calculate_temperature()

print("Mu: ",fc["mean_molecular_weight"][0])
print("Temperature [K]: ",fc["temperature"][0])

fc["RT_HI_ionization_rate"][0] = gamma
fc["RT_HeI_ionization_rate"][0] = 0.0
fc["RT_HeII_ionization_rate"][0] = 0.0

def x_HII():
    return (fc["HII"] / (fc["HI"] + fc["HII"]))[0]

# ===============
# CONFIGURE C2RAY
# ===============
ndens = np.copy(fc["HI"],order='F')
xh = np.copy(x_HII(),order='F')
temp = np.copy(fc["temperature"],order='F')
phi_ion = np.asfortranarray(gamma)
eth0 = 13.598
temph0 = eth0 * ev2k
colh0 = 1.3e-8 * 0.83 / eth0**2
bh00 = 1.26e-14
albpow = -0.75
abu_c = 7.1e-7

xnow_grackle = x_HII()
its = xnow_grackle / ((1.0 - xnow_grackle)*gamma)
print("Ionization time-scale [yr] = ",its/YEAR)
dt_grackle = max(dt_min,min(dt_max,dt_its_factor*its))
print("dt [yr]= ",dt_grackle/YEAR)

print("")
current_time = 0.0

xfrac_t_grackle = []
xfrac_t_c2ray = []
temp_t_grackle = []
temp_t_c2ray = []
time_t = []
dt_t = []
while (current_time < final_time):
    
    # SCratch
    fc.calculate_temperature()
    temp_p = fc["temperature"][0]
    
    #print(temp_p)
    
    # DETERMINE TIMESTEP
    
    xnow_grackle = x_HII()
    xnow_c2ray = xh

    brech0 = 4.881357e-6*temp_p**(-1.5) * (1.0 + 1.14813e2*temp_p**(-0.407))**(-2.242)
    phot_term = (1.0 - xnow_grackle)*gamma
    rec_term = xnow_grackle*fc["de"][0]*brech0
    #print(phot_term,rec_term)
    # its = xnow_grackle / np.abs(phot_term - 10000*rec_term)
    its = xnow_grackle / ((1.0-xnow_grackle) * gamma)
    #its = (1.0-xnow_grackle)*ndens[0] / np.abs((1.0-xnow_grackle)*ndens[0]*gamma - (xnow_grackle*ndens[0])**2*brech0)
    dt_grackle = max(dt_min,min(dt_max,dt_its_factor*its))

    if final_time - current_time < dt_grackle:
        actual_dt = final_time - current_time
    else:
        actual_dt = dt_grackle

    time_t.append(1.0 + current_time/YEAR)
    xfrac_t_grackle.append(xnow_grackle)
    xfrac_t_c2ray.append(xnow_c2ray)
    temp_t_grackle.append(temp_p)
    temp_t_c2ray.append(temp[0])
    dt_t.append(actual_dt/YEAR)
    # C2RAY
    #temp = np.copy(fc["temperature"],order='F')
    xh_new = global_pass(actual_dt,ndens,temp,xh,phi_ion,bh00,albpow,colh0,temph0,abu_c)
    xh = xh_new

    # GRACKLE
    fc.solve_chemistry(actual_dt)

    #print(f"x_HII = {xh_new:.5e}")
    current_time += actual_dt

    print(rf"t={current_time/YEAR:.3e} yrs   dt={actual_dt/YEAR:.3e} yrs    T(Grackle)={temp_p:.6f}  T(C2Ray)={temp[0]:.6f}   xHII(Grackle)={xnow_grackle:.6f}   xHII(C2Ray)={xnow_c2ray:.6f}")
print("done")

xfrac_t_c2ray = np.array(xfrac_t_c2ray)
xfrac_t_grackle = np.array(xfrac_t_grackle)
temp_t_c2ray = np.array(temp_t_c2ray)
temp_t_grackle = np.array(temp_t_grackle)
time_t = np.array(time_t)
dt_t = np.array(dt_t)

fig, ax = plt.subplots(2,1,constrained_layout=True,figsize=(12,5.5),
                       sharex=True,height_ratios=[5,2],squeeze=False)

ax[0,0].loglog(time_t,xfrac_t_grackle,label="Grackle")
ax[0,0].loglog(time_t,xfrac_t_c2ray,label="C2Ray")

fracerr = xfrac_t_grackle/xfrac_t_c2ray
#print(fracerr)

ax[1,0].semilogx(time_t,fracerr)

ax[0,0].legend()
ax[1,0].set_xlabel("1+t [years]")
ax[0,0].set_ylabel("Ionized Fraction")
ax[1,0].set_ylabel("Fractional Error")

# ============================================

fig2, ax2 = plt.subplots(2,1,constrained_layout=True,figsize=(12,5.5),
                       sharex=True,height_ratios=[5,2],squeeze=False)

ax2[0,0].loglog(time_t,temp_t_grackle,label="Grackle")
ax2[0,0].loglog(time_t,temp_t_c2ray,label="C2Ray")

fracerr_temp = temp_t_grackle/temp_t_c2ray

ax2[1,0].semilogx(time_t,fracerr_temp)

ax2[0,0].legend()
ax2[1,0].set_xlabel("1+t [years]")
ax2[0,0].set_ylabel("Temperature [K]")
ax2[1,0].set_ylabel("Fractional Error")

# ============================================
ax[0,0].set_title(title)
ax2[0,0].set_title(title)

fig3,ax3 = plt.subplots()
ax3.loglog(time_t,dt_t)
ax3.set_title("Timestep [yrs]")
plt.show()