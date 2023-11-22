from numpy import pi

# Internal Unit System
UnitLength_in_cm         = 3.085678e24    # Mpc
UnitMass_in_g            = 1.989e33       # Solar Mass
UnitVelocity_in_cm_per_s = 1e5            # km/sec 
UnitTime_in_s            = UnitLength_in_cm / UnitVelocity_in_cm_per_s
UnitEnergy_in_cgs        = UnitMass_in_g * UnitVelocity_in_cm_per_s**2
#UnitLength_in_cm         = 3.085678e21    # Mpc
#UnitMass_in_g            = 1.989e43       # Solar Mass

# Constants
gamma = 5/3.                    # Adiabatic index
#mu    = 0.58822635              # Mean gas weight (assuming full ionization and H (76%) He (24%) mixture)
mu    = 1.2195                   # Mean gas weight (assuming neutral H (76%) He (24%) mixture)
mh_cgs    = 1.6726e-24          # Hydrogen mass (CGS)
kb_cgs    = 1.3806e-16          # Boltzmann constant (CGS)
G_cgs     = 6.672e-8            # Gravitational constant (CGS)
c_cgs = 2.99792458e+10          # Speed of light (CGS)
H0          = 72 #67.7
HUBBLE= 3.2407789e-18           # Hubble constant in h/sec (= 100km/s/Mpc)
HubbleParam = H0 / 100.0     # Hubble parameter

# Convert constants to internal units
kb = kb_cgs / UnitEnergy_in_cgs
mh = mh_cgs / UnitMass_in_g
G  = G_cgs  / pow(UnitLength_in_cm, 3) * UnitMass_in_g * pow(UnitTime_in_s, 2)

# Cosmology
rho_c = pow(HubbleParam*HUBBLE*UnitTime_in_s,2)*3/(8*pi*G)

# Convert between specific internal energy and temperature
u2T = (gamma-1)*mu*mh/kb
u2T_cgs = (gamma-1)*mu*mh_cgs/kb_cgs