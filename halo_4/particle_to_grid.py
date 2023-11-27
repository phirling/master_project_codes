import numpy as np
from matplotlib import pyplot as plt
import argparse
from pNbody import Nbody
from pNbody import mapping
from scipy.ndimage import gaussian_filter
from my_format import GridSnapshot
import constants as cst

fontsz = 11
plt.rcParams.update({"font.size": fontsz})
figsize = (7,5)

# Parse user input
parser = argparse.ArgumentParser(
    description="Grid a particle distribution"
)
parser.add_argument("file", nargs="+", help="Nbody file")
parser.add_argument("-method", type=str, default='SPH', help="Method to use")
parser.add_argument("-N", type=int, default=256, help="Grid Size")
parser.add_argument("-fwhm", type=float, default=None, help="STD of Gaussian kernel")
parser.add_argument("-nn", type=int, default=48, help="Number of neighbours to calculate RSP")
parser.add_argument("-xfrac0", type=float, default=2e-4, help="Initial (homogeneous) ionized fraction")
parser.add_argument("-frsp", type=float, default=3, help="Factor by which to multiply RSP for SPH smoothing")
parser.add_argument("-o", type=str, default="grid_ic.hdf5", help="Output file")
parser.add_argument("--plot",action="store_true")
args = parser.parse_args()

fname = args.file
N = int(args.N)

nb = Nbody(fname)
mass = nb.mass
pos = nb.pos
internal_energy = nb.u_init
boxsize = nb.boxsize[0]
dr = boxsize / N
dV = dr*dr*dr
npart = pos.shape[0]

UnitDensity_in_cgs = cst.UnitMass_in_g / (cst.UnitLength_in_cm**3)

def indices_particles_in_box(pos,boxsize):
    in_x = np.logical_and(pos[:,0] >= 0, pos[:,0] <= boxsize)
    in_y = np.logical_and(pos[:,1] >= 0, pos[:,1] <= boxsize)
    in_z = np.logical_and(pos[:,2] >= 0, pos[:,2] <= boxsize)
    in_all = np.logical_and(np.logical_and(in_x,in_y),in_z)
    return np.where(in_all == 1)[0]

print("Number of particles: ",npart)
print("Box size [Mpc]:      ",boxsize)
print("Cell size [Mpc]:     ",dr)

idx = indices_particles_in_box(pos,boxsize)

total_mass_in_box = mass[idx].sum()

if args.method == 'gaussian':
    if args.fwhm is None:
        fwhm = boxsize / 10
    else:
        fwhm = args.fwhm
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # in cMpc
    sigma_px = sigma / dr

    M_hist = np.histogramdd(pos,bins=N,range=[[0,boxsize],[0,boxsize],[0,boxsize]],weights = mass)[0]
    U_hist = np.histogramdd(pos,bins=N,range=[[0,boxsize],[0,boxsize],[0,boxsize]],weights = mass * internal_energy)[0]
    rho_hist = M_hist / dV

    M_filtered = gaussian_filter(M_hist,sigma_px)
    U_filtered = gaussian_filter(U_hist,sigma_px)
    rho_filtered = gaussian_filter(rho_hist,sigma_px)
    #T_filtered = (cst.gamma-1)*cst.mu*cst.mh/cst.kb * U_filtered / M_filtered
    u_filtered = U_filtered / M_filtered
    dens_cgs = rho_filtered * UnitDensity_in_cgs
    #temp_cgs = T_filtered
    u_cgs = u_filtered * cst.UnitEnergy_in_cgs / cst.UnitMass_in_g

elif args.method == 'SPH':
    # We normalize the positions to the box size, i.e. boxsize = 1. Any particle outside
    # the box size will be excluded from the gaussian map
    xmin = 0
    xmax = boxsize
    ymin = 0
    ymax = boxsize
    zmin = 0
    zmax = boxsize
    pos[:,0] = ((nb.pos[:,0]-xmin) / (xmax-xmin))
    pos[:,1] = ((nb.pos[:,1]-ymin) / (ymax-ymin))
    pos[:,2] = ((nb.pos[:,2]-zmin) / (zmax-zmin))

    nnb = int(args.nn)
    frsp = float(args.frsp)
    print(f"INFO: Doing SPH smoothing with {nnb:n} neighbours and rsp factor {frsp:.3f}")
    nb.ComputeRsp(nnb)
    print("Smoothing mass...")
    M_filtered = mapping.mkmap3dksph(pos,mass,np.ones(nb.nbody,np.float32),frsp*nb.rsp, (N, N, N),verbose=1)
    print("Smoothing internal energy...")
    U_filtered = mapping.mkmap3dksph(pos,mass * internal_energy,np.ones(nb.nbody,np.float32),frsp*nb.rsp, (N, N, N),verbose=1)
    u_filtered = U_filtered / M_filtered

    #T_filtered = (cst.gamma-1)*cst.mu*cst.mh/cst.kb * U_filtered / M_filtered
    #data = mapping.mkmap3dsph(pos,mass,np.ones(nb.nbody,np.float32),nb.rsp, (N, N, N))
    dens_cgs = M_filtered/dV * UnitDensity_in_cgs
    #temp_cgs = T_filtered
    u_cgs = u_filtered * cst.UnitEnergy_in_cgs / cst.UnitMass_in_g

mass_cons_err = ( (dens_cgs * dV / UnitDensity_in_cgs).sum() - total_mass_in_box) / total_mass_in_box
print("Mass conservation error (relative):", mass_cons_err)

temp_cgs = (cst.gamma-1)*cst.mu*cst.mh_cgs/cst.kb_cgs * u_cgs
if args.plot:
    fig2,ax2 = plt.subplots(1,2,constrained_layout=True,figsize=(12,5.5))
    im1 = ax2[0].imshow(dens_cgs.T[:,:,N//2-1],norm='log',origin='lower',interpolation='gaussian',cmap='viridis')
    im2 = ax2[1].imshow(temp_cgs.T[:,:,N//2-1],norm='log',origin='lower',interpolation='gaussian',cmap='jet')
    plt.colorbar(im1)
    plt.colorbar(im2)
    ax2[0].set_xlabel("$x$ Mpc")
    ax2[0].set_ylabel("$y$ Mpc")
    ax2[1].set_xlabel("$x$ Mpc")
    ax2[1].set_ylabel("$y$ Mpc")
    ax2[0].set_title("Density [g/cm3]")
    ax2[1].set_title("Temperature [K]")
    plt.show()

# Save result
xfrac = float(args.xfrac0) * np.ones((N,N,N))
gs = GridSnapshot(N=N,dens_cgs=dens_cgs,u_cgs=u_cgs,xfrac=xfrac,boxsize=boxsize,time=0.0)
gs.write(str(args.o))