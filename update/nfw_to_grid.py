from particle_to_grid import swift_to_grid
import numpy as np
import argparse
import unyt

H0 = 72.0 * unyt.km / unyt.s / unyt.Mpc
rho_c = 3*H0**2 / (8*np.pi*unyt.G)
rho_c = rho_c.to(unyt.Solar_Mass/unyt.kpc**3)
gamma = 5.0 / 3.0
G = (unyt.G.to(unyt.kpc / (1e10*unyt.Solar_Mass) * (unyt.km/ unyt.s)**2)).value

# Parse user input
parser = argparse.ArgumentParser(description="Project a SWIFT-formatted NFW halo onto a regular cartesian grid")
parser.add_argument("file", nargs="+", help="Nbody file")
parser.add_argument("-N", type=int, default=256, help="Grid Size")
parser.add_argument("-xfrac0", type=float, default=2e-4, help="Initial (homogeneous) ionized fraction")
parser.add_argument("-frsp", type=float, default=3, help="Factor by which to multiply RSP for SPH smoothing")
parser.add_argument("-o", type=str, default="grid_ic.hdf5", help="Output file")
parser.add_argument("-logM200",type=float,default=7,help="log10 of Virial mass of halo in Msun")
parser.add_argument("-c",type=float,default=20,help="NFW Concentration of halo")
args = parser.parse_args()

M200 = 10.0**args.logM200 * unyt.Solar_Mass
c = float(args.c)
fname = str(args.file[0])
N =int(args.N)

delta_c = 200/3. * c**3/( np.log(1+c) - c/(1+c) )
r200 = (M200/(4/3.*np.pi*rho_c*200))**(1/3.)

swift_to_grid(fname,N,r200.value,args.frsp,args.o,args.xfrac0)