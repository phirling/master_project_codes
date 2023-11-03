import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from pNbody import profiles
from pNbody import ic
#from astropy.cosmology import FlatLambdaCDM
import astropy.constants as ac
import astropy.units as u
from matplotlib.patches import Circle

"""
IC script to generate a NFW gas halo with variable parameters
"""

# Parse Options
parser = argparse.ArgumentParser("IC for AGORA NFW halo")
parser.add_argument("-o",dest="output",type=str,default="nfw.hdf5",help="output file name")
parser.add_argument("--plot",action="store_true")
parser.add_argument("-H0",type=float,default=67.7,help="Hubble Constant")
parser.add_argument("-Om0",type=float,default=0.310,help="Matter density parameter")
parser.add_argument("-M200",type=float,default=3.14799e+08,help="Virial mass of halo (total mass)")
parser.add_argument("-c",type=float,default=17.21,help="NFW Concentration of halo")
parser.add_argument("-fb",type=float,default=0.15,help="Baryonic fraction of halo (gas fraction)")
parser.add_argument("-Ngas",type=int,default=100000,help="Number of gas particles to sample")
parser.add_argument("-boxsize",type=float,default=0.05,help="Box size in Mpc")
parser.add_argument("-boxsize_factor",type=float,default=1,help="Ratio of half-boxsize to r200")
args = parser.parse_args()

# =================
# Set up Parameters
# =================

# Internal Unit System
UnitLength_in_cm         = 3.085678e24    # Mpc
UnitMass_in_g            = 1.989e33       # Solar Mass
UnitVelocity_in_cm_per_s = 1e5            # km/sec 
UnitTime_in_s            = UnitLength_in_cm / UnitVelocity_in_cm_per_s
UnitEnergy_in_cgs        = UnitMass_in_g * UnitVelocity_in_cm_per_s**2

# Constants
gamma = 5/3.                    # Adiabatic index
mu    = 0.58822635              # Mean gas weight (assuming full ionization and H (76%) He (24%) mixture)
mh_cgs    = 1.6726e-24          # Hydrogen mass (CGS)
kb_cgs    = 1.3806e-16          # Boltzmann constant (CGS)
G_cgs     = 6.672e-8            # Gravitational constant (CGS)
HUBBLE= 3.2407789e-18           # Hubble constant in h/sec (= 100km/s/Mpc)
HubbleParam = args.H0 / 100     # Hubble parameter

# Convert constants to internal units
kb = kb_cgs / UnitEnergy_in_cgs
mh = mh_cgs / UnitMass_in_g
G  = G_cgs  / pow(UnitLength_in_cm, 3) * UnitMass_in_g * pow(UnitTime_in_s, 2)

# Cosmology
#cosmo = FlatLambdaCDM(args.H0,args.Om0,Tcmb0=2.7255)
#rho_c  = cosmo.critical_density0.cgs.value * UnitLength_in_cm**3 / UnitMass_in_g #pow(HubbleParam*HUBBLE*UnitTime_in_s,2)*3/(8*np.pi*G)
#print(rho_c)
rho_c = pow(HubbleParam*HUBBLE*UnitTime_in_s,2)*3/(8*np.pi*G)

# Parameters of the halo
fb = args.fb                       # Baryonic fraction
M200 = args.M200
c = args.c

'''
Recall:
Virial radius r200 == radius s.t. mean dens inside r200 is 200*rho_c
Virial mass M200 == total mass inside r200 == 4/3πr200^3 * 200 * rho_c
'''

# Derived NFW parameters
delta_c = 200/3. * c**3/( np.log(1+c) - c/(1+c) )
rho_0 = delta_c * rho_c
r_s = 1/c * (M200/(4/3.*np.pi*rho_c*200))**(1/3.)
r200 = c*r_s

# Gas GAS density
def Density(r):
    return fb*rho_0/((r/r_s)*(1+r/r_s)**2)

# Total GAS mass inside the radius r
def Mr(r):
    return fb*4*np.pi*rho_0*r_s**3 * ( np.log(1+r/r_s) - (r/r_s)/(1+r/r_s) )

# Circular velocity
def Vcirc(r):
    return  np.sqrt(G*Mr(r)/r)

# Box size: chosen s.t. r200 exactly fits inside
L = args.boxsize_factor * 2*r200 #args.boxsize                 # Box size in internal units

# The model is truncated at some radius rmax. Here, we want the box to be "full",
# so we set 2*rmax = box diagonal = sqrt3*L
rmax = np.sqrt(3) * L / 2

# Numerical parameters
Ngas = int(args.Ngas)
Mgas = Mr(rmax) # Since the model is truncated at arbitrary rmax, the total particle mass must be = M(rmax)
mgas = Mgas / Ngas
#mgas = fb*M200 / Ngas
# mgas = 5.65e4 #/1e10              # AGORA gas resolution
# Ngas = int(fb*M200/mgas)        # The model is truncated at r200

# We want r200 to be within the box
if r200 > L:
    print("WARNING: r200 > Box Size")

# Estimate of dynamical time in internal coordinates (= period of circular orbit at r_s)
vcirc_at_rs = Vcirc(r_s)
tdyn = 2*np.pi*r_s / vcirc_at_rs

Myr = u.Myr.to('s')

# Print info
print("In Internal Units: (Mpc, Msun, km/s), time in Myr")
print(f"G                = {G           : .5e}")
print(f"Boltzmann        = {kb          : .5e}")
print(f"ProtonMass       = {mh          : .5e}")
print(f"mumh             = {mh*mu       : .5e}")
print(f"Critical density = {rho_c       : .5e}")
print(f"r200             = {r200        : .5e}")
print(f"r_s              = {r_s         : .5e}")
print(f"Ngas             = {Ngas        : n}")
print(f"M200 (Tot)       = {M200        : .5e}")
print(f"M200 (Gas)       = {fb*M200     : .5e}")
print(f"M200 (DM)        = {(1-fb)*M200 : .5e}")
print(f"mgas             = {mgas        : .5e}")
print(f"Estimate of tdyn = {tdyn*UnitTime_in_s/Myr: .5e}")
print(f"rmax             = {rmax        : .5e}")
print(f"boxsize          = {L           : .5f}")
print(f"boxsize / 2*r200 = {L/(2*r200)  : .2f}")

# ================================
# Create Nbody object and write IC
# ================================
addargs = (r_s,)
pr_fct = profiles.nfw_profile
mr_fct = profiles.nfw_mr

Neps_des = 10.  # number of des. points in eps
ng = 256  # number of division to generate the model
rc = 0.1  # default rc (if automatic rc fails) length scale of the grid
dR = 0.1

Rqs, rc, eps, Neps, g, gm = ic.ComputeGridParameters(Ngas, addargs, rmax, M200, pr_fct, mr_fct, Neps_des, rc, ng)    

nb = ic.nfw(Ngas, r_s, rmax, dR, Rqs, name=args.output, ftype='swift')
nb.verbose = True

# Set gas mass
nb.mass = mgas * np.ones(Ngas)

# Units
nb.UnitLength_in_cm         = UnitLength_in_cm        
nb.UnitMass_in_g            = UnitMass_in_g           
nb.UnitVelocity_in_cm_per_s = UnitVelocity_in_cm_per_s
nb.Unit_time_in_cgs         = UnitLength_in_cm/UnitVelocity_in_cm_per_s

# Set box size & shift particles
nb.boxsize = L
nb.pos += L / 2

# Write output
nb.write()

# ===============
# Plot
# ===============
if args.plot:
    fig2,ax2 = plt.subplots()
    hist = np.histogram2d(nb.pos[:,0],nb.pos[:,1],800,range=[[0,L],[0,L]])[0]
    ax2.imshow(hist.T,norm='log',extent=(0,L,0,L),origin='lower')
    ax2.set_xlabel("$x$ Mpc")
    ax2.set_ylabel("$y$ Mpc")
    ax2.set_title("Projected Surface Density")
    circle = Circle((L/2,L/2),r200,fill=False,ls='--',color='magenta')
    ax2.add_patch(circle)
    tf = 0.8
    ax2.text(L/2+tf*r200,L/2+tf*r200,"$r_{200}$",color='magenta')
    plt.show()