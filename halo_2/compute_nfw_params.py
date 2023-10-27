from scipy.optimize import root_scalar
import astropy.units as u
import astropy.constants as ac
import numpy as np

def delta_c_of_c(c):
    return 200./3 * c**3 / (np.log(1+c) - c/(1+c))

def find_concentration(delta_c):
    f = lambda c : delta_c_of_c(c) - delta_c
    res = root_scalar(f,x0=5,x1=20)
    print(res)
    return res.root

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-rho0",default=2.5e16,help="rho0 density, in msun/Mpc^3")
    parser.add_argument("-rs",default=8.0e-4,help="NFW scale radius r_s, in Mpc")
    parser.add_argument("-H0",default=67.7)
    args = parser.parse_args()

    H0 = args.H0 * u.km / u.s / u.Mpc
    G = ac.G
    rho_crit = 3*H0**2 / (8*np.pi*G)
    rho_0 = args.rho0 * u.Msun / u.Mpc**3
    r_s = args.rs * u.Mpc

    delta_c = rho_0 / rho_crit

    c = find_concentration(delta_c)

    r_200 = (c * r_s).to('Mpc')
    M_200 = (4.0 / 3 * np.pi * rho_crit * 200 * r_200**3).to('Msun')

    print("Found the following parameters:")
    print(f"r200 = {r_200.value : .5e} Mpc")
    print(f"M200 = {M_200.value : .5e} Msun")
    print(f"c    = {c:.2f}")
    