import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from pNbody import profiles
from pNbody import ic
import constants as cst
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
parser.add_argument("-logM200",type=float,default=7,help="log10 of Virial mass of halo (total mass)")
parser.add_argument("-c",type=float,default=17,help="NFW Concentration of halo")
parser.add_argument("-fb",type=float,default=0.15,help="Baryonic fraction of halo (gas fraction)")
parser.add_argument("-Ngas",type=int,default=100000,help="Number of gas particles to sample")
parser.add_argument("-boxsize_factor",type=float,default=1,help="Ratio of half-boxsize to r200")
args = parser.parse_args()

# Parameters of the halo
fb = args.fb                       # Baryonic fraction
M200 = 10.0**(float(args.logM200))
c = args.c

'''
Recall:
Virial radius r200 == radius s.t. mean dens inside r200 is 200*rho_c
Virial mass M200 == total mass inside r200 == 4/3πr200^3 * 200 * rho_c
'''

# Derived NFW parameters
delta_c = 200/3. * c**3/( np.log(1+c) - c/(1+c) )
rho_0 = delta_c * cst.rho_c
r_s = 1/c * (M200/(4/3.*np.pi*cst.rho_c*200))**(1/3.)
r200 = c*r_s

# Gas GAS density
def Density(r):
    return fb*rho_0/((r/r_s)*(1+r/r_s)**2)

# Total GAS mass inside the radius r
def Mr(r):
    return 4*np.pi*rho_0*r_s**3 * ( np.log(1+r/r_s) - (r/r_s)/(1+r/r_s) )

# Circular velocity
def Vcirc(r):
    return  np.sqrt(cst.G*Mr(r)/r)

# Integrand for the pressure
def integrand(r):
    return Density(r) * cst.G * Mr(r) /  r**2

# Pressure
def P(r,rmax):
    Pr = quad(integrand, r, rmax, args=())
    return Pr[0]

# Specific energy
def U(P,r):
    u = P/(cst.gamma-1)/Density(r)
    return u

# Temperature
def T(P,r):
    u = U(P,r)
    T = (cst.gamma-1)*cst.mu*cst.mh/cst.kb *u
    return T

# Box size: chosen s.t. r200 exactly fits inside
L = args.boxsize_factor * 2*r200 #args.boxsize                 # Box size in internal units

# The model is truncated at some radius rmax. Here, we want the box to be "full",
# so we set 2*rmax = box diagonal = sqrt3*L
rmax = np.sqrt(3) * L / 2
#rmax = r200
rmin = 1e-3*r_s
nr   = 1000 
rr = 10**np.linspace(np.log10(rmin),np.log10(rmax),nr)

# Numerical parameters
Ngas = int(args.Ngas)
Mgas = fb * Mr(rmax) # Since the model is truncated at arbitrary rmax, the total particle mass must be = M(rmax)
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
print("=== Internal Unit System === ")
print(f"UnitLength_in_cm         = {cst.UnitLength_in_cm        : .4e}")
print(f"UnitMass_in_g            = {cst.UnitMass_in_g           : .4e}")
print(f"UnitVelocity_in_cm_per_s = {cst.UnitVelocity_in_cm_per_s: .4e}") 
print(f"UnitTime_in_s            = {cst.UnitTime_in_s           : .4e}")
print(f"UnitEnergy_in_cgs        = {cst.UnitEnergy_in_cgs       : .4e}")
print("")
print("=== Parameters in Internal Units: (Mpc, Msun, km/s), time in Myr === ")
print(f"G                = {cst.G           : .5e}")
print(f"Boltzmann        = {cst.kb          : .5e}")
print(f"ProtonMass       = {cst.mh          : .5e}")
print(f"mumh             = {cst.mh*cst.mu   : .5e}")
print(f"Critical density = {cst.rho_c       : .5e}")
print(f"r200             = {r200            : .5e}")
print(f"r_s              = {r_s             : .5e}")
print(f"Ngas             = {Ngas            : n}")
print(f"M200 (Tot)       = {M200            : .5e}")
print(f"M200 (Gas)       = {fb*M200         : .5e}")
print(f"M200 (DM)        = {(1-fb)*M200     : .5e}")
print(f"mgas             = {mgas            : .5e}")
print(f"Estimate of tdyn = {tdyn*cst.UnitTime_in_s/Myr: .5e}")
print(f"rmax             = {rmax        : .5e}")
print(f"boxsize          = {L           : .5f}")
print(f"boxsize / 2*r200 = {L/(2*r200)  : .2f}")

# ================================================
# Compute physical quantities in integration range
# ================================================
Vc = Vcirc(rr)
rho = Density(rr) 
Ps = np.zeros(len(rr))
for i in range(len(rr)):
    Ps[i] = P(rr[i],10*rmax) # Integral is to infty, here, use 10rmax
Ts = T(Ps,rr)
us = U(Ps,rr)

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
nb.UnitLength_in_cm         = cst.UnitLength_in_cm        
nb.UnitMass_in_g            = cst.UnitMass_in_g           
nb.UnitVelocity_in_cm_per_s = cst.UnitVelocity_in_cm_per_s
nb.Unit_time_in_cgs         = cst.UnitLength_in_cm/cst.UnitVelocity_in_cm_per_s

# interpolate the specific energy
u_interpolator = interp1d(rr, us, kind='linear')
nb.u_init = u_interpolator(nb.rxyz())

# Set box size & shift particles
nb.boxsize = L
nb.pos += L / 2

# Write output
nb.write()

# ===============
# Plot
# ===============
if args.plot:
    fig2,ax2 = plt.subplots(1,2,constrained_layout=True,figsize=(12,5.5))
    N = 50
    internal_energy = nb.u_init * nb.mass
    U_hist = np.histogram2d(nb.pos[:,0],nb.pos[:,1],N,range=[[0,L],[0,L]],weights=internal_energy)[0]
    M_hist = np.histogram2d(nb.pos[:,0],nb.pos[:,1],N,range=[[0,L],[0,L]],weights=nb.mass)[0]
    dA = (L / N)**2
    dens2cgs = cst.UnitMass_in_g / cst.UnitLength_in_cm**3
    ndens2cgs = 1 / cst.UnitLength_in_cm**3
    T_hist = (cst.gamma-1)*cst.mu*cst.mh/cst.kb * U_hist / M_hist
    n_hist = M_hist / (cst.mu*cst.mh) / dA
    rho_hist = M_hist / dA
    #ax2.imshow(hist.T,norm='log',extent=(0,L,0,L),origin='lower',interpolation='gaussian')
    im1 = ax2[0].imshow(ndens2cgs * n_hist.T,norm='log',extent=(0,L,0,L),origin='lower',interpolation='gaussian',cmap='viridis')
    #im1 = ax2[0].imshow(dens2cgs * rho_hist.T,norm='log',extent=(0,L,0,L),origin='lower',interpolation='gaussian',cmap='viridis')
    im2 = ax2[1].imshow(T_hist.T,norm='log',extent=(0,L,0,L),origin='lower',interpolation='gaussian',cmap='jet')
    plt.colorbar(im1)
    plt.colorbar(im2)
    ax2[0].set_xlabel("$x$ Mpc")
    ax2[0].set_ylabel("$y$ Mpc")
    ax2[1].set_xlabel("$x$ Mpc")
    ax2[1].set_ylabel("$y$ Mpc")
    ax2[0].set_title("Projected Surface Density [atom/cm3]")
    ax2[1].set_title("Projected Surface Temperature [K]")
    circle = Circle((L/2,L/2),r200,fill=False,ls='--',color='magenta')
    ax2[0].add_patch(circle)
    tf = 0.8
    ax2[0].text(L/2+tf*r200,L/2+tf*r200,"$r_{200}$",color='magenta')
    plt.show()