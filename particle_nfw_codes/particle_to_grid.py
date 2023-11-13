import numpy as np
from matplotlib import pyplot as plt
import argparse
from pNbody import Nbody
from pNbody import mapping
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import pickle as pkl

fontsz = 11
plt.rcParams.update({"font.size": fontsz})
figsize = (7,5)

# Parse user input
parser = argparse.ArgumentParser(
    description="Plot multiple density profiles against theoretical prediction"
)
parser.add_argument("file", nargs="+", help="Nbody file")
parser.add_argument("-method", type=str, default='histogram', help="Method to use")
parser.add_argument("-N", type=int, default=256, help="Grid Size")
parser.add_argument("-fwhm", type=float, default=None, help="STD of Gaussian kernel")
parser.add_argument("-nn", type=int, default=48, help="Number of neighbours to calculate RSP")
parser.add_argument("-frsp", type=float, default=2, help="Factor by which to multiply RSP for SPH smoothing")
parser.add_argument("-o", type=str, default="gridded_density.pkl", help="Output file")
parser.add_argument("--plot",action="store_true")
args = parser.parse_args()

Msun_to_g = 1.989e33
Mpc_to_cm = 3.085678e24
Msun_per_Mpc3_to_g_per_cm3 = Msun_to_g / (Mpc_to_cm**3)
fname = args.file
N = int(args.N)

nb = Nbody(fname)
mass = nb.mass
pos = nb.pos
boxsize = nb.boxsize[0]
dr = boxsize / N
dV = dr*dr*dr
npart = pos.shape[0]

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

def cgs_dens_from_histogram(pos,mass,N):
    H, edges = np.histogramdd(pos,bins=N,range=[[0,boxsize],[0,boxsize],[0,boxsize]],weights=mass)
    return H/dV * Msun_per_Mpc3_to_g_per_cm3
if args.method == 'histogram':
    dens_cgs = cgs_dens_from_histogram(pos,mass,N)

elif args.method == 'gaussian':
    grid = np.zeros((N,N,N))
    if args.fwhm is None:
        fwhm = boxsize / 10
    else:
        fwhm = args.fwhm
    # #fwhm = 0.001                                   # std of the Gaussian kernel in cMpc
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # in cMpc
    sigma_px = sigma / dr                # in grid unit
    f_kernel = sigma_px*3                        # kernel size in grid unit
    print(f_kernel)
    print(sigma, sigma_px, f_kernel)
# 
    dens_hist = cgs_dens_from_histogram(pos,mass,N)
    dens_cgs = gaussian_filter(dens_hist,sigma_px)
    # idr = dr * (0.5 + np.arange(0,N))
    # grid_cells_pos = np.array(np.meshgrid(idr,idr,idr))
    # 
    # for i in tqdm(range(npart)):
    #     #dist = np.linalg.norm((grid_cells_pos.T - pos[i,:]).T,axis=0)
    #     #grid += mass[i] * np.exp(-dist**2 / (2*sigma_px**2))
    #     dist = 1
    #     grid[1:40,1:40,1:40] += mass[i] * np.exp(-dist**2 / (2*sigma_px**2))
    # for i in tqdm(range(N)):
    #     for j in range(N):
    #         for k in range(N):
    #             dist = np.linalg.norm(pos - grid_cells_pos[:,i,j,k], axis=1)
    #             dist_mask = dist <= f_kernel
    #             mass_amplitude = mass[dist_mask]
    #             grid[i,j,k] = np.sum(mass_amplitude*np.exp(-dist[dist_mask]**2 / (2*sigma_px**2)))
    #             # grid[i,j,k] = np.sum(np.linspace(0,1,100))
    #     #print(dist.shape)
    # dens_cgs = grid/dV * Msun_per_Mpc3_to_g_per_cm3
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
    data = mapping.mkmap3dksph(pos,mass,np.ones(nb.nbody,np.float32),frsp*nb.rsp, (N, N, N),verbose=1)
    #data = mapping.mkmap3dsph(pos,mass,np.ones(nb.nbody,np.float32),nb.rsp, (N, N, N))
    dens_cgs = data/dV * Msun_per_Mpc3_to_g_per_cm3

mass_cons_err = ( (dens_cgs * dV / Msun_per_Mpc3_to_g_per_cm3).sum() - total_mass_in_box) / total_mass_in_box
print("Mass conservation error (relative):", mass_cons_err)

if args.plot:
    logdens = np.log10(dens_cgs[:,:,N//2-1])
    plt.imshow(logdens)
    plt.colorbar()
    plt.show()

# Save result
res = {
    "dens_cgs" : dens_cgs,
    "boxsize" : boxsize,
    "mass_error" : mass_cons_err,
    "npart" : npart,
    "args" : args
}
with open(args.o,"wb") as f:
    pkl.dump(res,f)