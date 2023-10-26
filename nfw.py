import numpy as np
from scipy.integrate import quad

# Internal Unit System
UnitLength_in_cm         = 3.085678e24    # Mpc
UnitMass_in_g            = 1.989e33       # solar mass
UnitVelocity_in_cm_per_s = 1e5            # km/sec 
UnitTime_in_s            = UnitLength_in_cm / UnitVelocity_in_cm_per_s
UnitEnergy_in_cgs        = UnitMass_in_g * UnitVelocity_in_cm_per_s**2
UnitDensity_in_cgs       = UnitMass_in_g / UnitLength_in_cm**3

# Constants
gamma = 5/3.                    # Adiabatic index
mu    = 0.58822635              # Mean gas weight (assuming full ionization and H (76%) He (24%) mixture)
mh_cgs    = 1.6726e-24          # Hydrogen mass (CGS)
kb_cgs    = 1.3806e-16          # Boltzmann constant (CGS)
G_cgs     = 6.672e-8            # Gravitational constant (CGS)
HUBBLE= 3.2407789e-18           # Hubble constant in h/sec (= 100km/s/Mpc)
HubbleParam = 0.72              # Reduced hubble parameter (small h)
m_p = 1.672661e-24
abu_h =  0.926
abu_he = 0.074
mean_molecular = abu_h + 4.0*abu_he

# Convert constants to internal units
kb = kb_cgs / UnitEnergy_in_cgs
mh = mh_cgs / UnitMass_in_g
G  = G_cgs  / pow(UnitLength_in_cm, 3) * UnitMass_in_g * pow(UnitTime_in_s, 2)

# Critical density of the universe
rho_c_global  = pow(HubbleParam*HUBBLE*UnitTime_in_s,2)*3/(8*np.pi*G)

class NFW:
    def __init__(self,M200,c,eps,rho_c = None) -> None:
        """
        M200 : float
            Mass of the NFW halo, in solar masses
        c : float
            Concentration of the NFW halo
        eps : float
            Gravitational softening, in Mpc
        """
        self.m200 = M200
        self.c = c
        self.eps = eps
        if rho_c is None:
            self.rho_c = rho_c_global
        else:
            self.rho_c = rho_c

        # Derived NFW parameters
        self.delta_c = 200/3. * c**3/( np.log(1+c) - c/(1+c) )
        self.rho_0 = self.delta_c * self.rho_c
        self.r_s = 1/c * (M200/(4/3.*np.pi*self.rho_c*200))**(1/3.)
        self.r200 = c*self.r_s
        print("rho_c [atoms/cm3]  = ",self.rho_c * UnitDensity_in_cgs / (mean_molecular * m_p))
        print("r_s [Mpc]  = ",self.r_s)
        print("r200 [Mpc] = ",self.r200)

    # Mass Density
    def density(self,r):
        return self.rho_0/(((r + self.eps)/self.r_s)*(1+(r + self.eps)/self.r_s)**2)
    
    def density_gridded(self,N,boxsize):
        xi = np.arange(0,N)
        dr = boxsize / N
        halopos = np.array([N//2-1,N//2-1,N//2-1])
        X,Y,Z = np.meshgrid(xi,xi,xi)
        R = dr * np.sqrt((X - halopos[0])**2 + (Y - halopos[1])**2 + (Z - halopos[2])**2)
        ndens = self.density(R) * UnitDensity_in_cgs / (mean_molecular * m_p)
        return ndens