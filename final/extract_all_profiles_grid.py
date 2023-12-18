import numpy as np
import matplotlib.pyplot as plt
import mpph
from tqdm import tqdm
import argparse
import h5py
import os

parser = argparse.ArgumentParser("Extract spherical profiles from snapshots")
parser.add_argument("files",nargs='+',help="Snapshot files")
parser.add_argument("-nbins",type=int,default=10,help="Number of radial bins to use")
parser.add_argument("-massfile",type=str,default=None,help="File to extract single density profile")
parser.add_argument("-o",type=str,default="profiles.hdf5",help="Output file")
parser.add_argument("-XH",type=float,default=1.0,help="Hydrogen mass fraction")
parser.add_argument("--verbose",action='store_true')
args = parser.parse_args()

nbins = int(args.nbins)
outfn = str(args.o)

# Create new output HDF5 file and add metadata
if args.verbose: print("Saving profiles to '" + outfn + "' ...")
with h5py.File(outfn,"w") as f:
    f.attrs['unit_distance'] = 'kpc'
    f.attrs['unit_time'] = 'Myr'
    f.attrs['unit_temperature'] = 'K'
    f.attrs['unit_number_density'] = '1/cm**3'
    # If a mass density file is provided, we extract the profile once
    # here and add it to the root of the output file
    if args.massfile is not None:
        rbc, pndens = mpph.get_density_profile_grid(args.massfile,nbins,args.XH)
        f.create_dataset('rbin_centers',data=rbc,dtype=np.float64)
        f.create_dataset('profile_ndens',data=pndens,dtype=np.float64)

# Now loop through snapshots and extract the profiles
for k,fn in enumerate(args.files):
    hname = fn.split(".")[-2]
    print(hname)
    rbc, ptemp = mpph.get_temperature_profile_grid(fn,nbins,args.XH)
    rbc, pxfrac = mpph.get_xfrac_profile_grid(fn,nbins,ionized=False)
    gs = mpph.GridSnapshot(fn)
    time_myr = gs.time
    boxsize_kpc = gs.boxsize * 1000
    N = gs.N
    # Append result to HDF5 file
    with h5py.File(outfn,"a") as f:
        hname = os.path.splitext(os.path.basename(fn))[0]
        if args.verbose: print("Create group '" + hname + "'")
        f.create_group(hname)
        f[hname].attrs['time'] = time_myr
        f[hname].attrs['boxsize'] = boxsize_kpc
        f[hname].attrs['filename_full'] = fn
        f[hname].create_dataset('rbin_centers',data=rbc,dtype=np.float64)
        f[hname].create_dataset('profile_temperature',data=ptemp,dtype=np.float64)
        f[hname].create_dataset('profile_xHI',data=pxfrac,dtype=np.float64)